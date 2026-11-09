"""Native RapidChiplet bridge, independent of the Legacy/INDM adapter."""
from dataclasses import dataclass
import contextlib
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import sys


@dataclass(frozen=True)
class Network:
    bandwidth_Bps: dict
    latency_s: dict
    length_mm: dict
    area_mm2: float
    idle_power_w: float
    source_hashes: dict
    inputs: dict


def evaluate_network(hardware, native_root=None):
    hardware.validate()
    root = Path(native_root) if native_root else Path(__file__).resolve().parents[2]/"third_party/rapidchiplet-main"
    lock_path = Path(__file__).resolve().parents[2]/"rapidchiplet.lock.json"
    expected = json.loads(lock_path.read_text())["reference_sha256"]
    hashes = {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in expected}
    if hashes != expected:
        raise ValueError("native RapidChiplet source hash differs from lock")
    for name in ("rapidchiplet","helpers","validation","booksim_wrapper"):
        loaded = sys.modules.get(name)
        if loaded is not None and Path(loaded.__file__).resolve().parent != root.resolve():
            raise RuntimeError("native module collision: use a separate process")
    sys.path.insert(0,str(root.resolve()))
    try:
        rc = importlib.import_module("rapidchiplet")
        val = importlib.import_module("validation")
    finally:
        sys.path.pop(0)
    indices = {n.id:i for i,n in enumerate(hardware.nodes)}
    ports = {n.id:[] for n in hardware.nodes}
    edges = []
    for l in hardware.links:
        endpoints = []
        for name in (l.a,l.b):
            n = hardware.node_map[name]
            port = len(ports[name])
            wires = math.ceil(16*l.bandwidth_Bps/hardware.network_clock_hz)+2
            fraction = wires*hardware.bump_pitch_mm**2/(n.width_mm*n.height_mm*0.5)
            ports[name].append({"x":n.width_mm/2,"y":n.height_mm/2,"fraction_bump_area":fraction})
            endpoints.append({"type":"chiplet","outer_id":indices[name],"inner_id":port})
        edges.append({"ep1":endpoints[0],"ep2":endpoints[1]})
    chiplets = {}
    for n in hardware.nodes:
        if sum(p["fraction_bump_area"] for p in ports[n.id]) > 1:
            raise ValueError(f"PHY bump budget exceeded on {n.id}")
        chiplets[n.id] = {"dimensions":{"x":n.width_mm,"y":n.height_mm},"type":n.kind,
            "phys":ports[n.id],"fraction_power_bumps":0.5,"technology":"exploratory",
            "power":n.static_power_w,"relay":True,"internal_latency":hardware.router_latency_cycles,
            "unit_count":n.slots}
    table = {}
    paths = {}
    for a in indices:
        table[("chiplet",indices[a])] = {}
        for b in indices:
            route = hardware.route(a,b)
            paths[(a,b)] = route
            table[("chiplet",indices[a])][("chiplet",indices[b])] = (
                ("chiplet",indices[route[0][2]]) if route else None)
    inputs = {"design":{},"verbose":False,"validate":True,"chiplets":chiplets,
        "placement":{"chiplets":[{"name":n.id,"position":{"x":n.x_mm,"y":n.y_mm},"rotation":0}
                                   for n in hardware.nodes],"interposer_routers":[]},
        "topology":edges,"technologies":{"exploratory":{"phy_latency":hardware.phy_latency_cycles,
            "wafer_radius":150,"wafer_cost":1000,"defect_density":0.01}},
        "packaging":{"link_routing":"manhattan","link_latency_type":"constant","link_latency":1,
            "link_power_type":"constant","link_power":0,"packaging_yield":0.9,
            "bump_pitch":hardware.bump_pitch_mm,"non_data_wires":0,"is_active":False,
            "latency_irouter":0,"power_irouter":0,"has_interposer":False},
        "routing_table":{"type":"default","table":table},"traffic_by_chiplet":{}}
    report = io.StringIO()
    with contextlib.redirect_stdout(report):
        for check in (val.validate_chiplets,val.validate_placement,val.validate_topology,val.validate_routing_table):
            check(inputs)
    if "VALIDATION ERROR" in report.getvalue():
        raise ValueError(report.getvalue())
    intermediate = {}
    lengths = rc.compute_link_lengths(inputs,intermediate)
    intermediate["link_lengths"] = lengths
    bandwidths = rc.compute_link_bandwidths(inputs,intermediate)
    latencies = {}
    for l,edge in zip(hardware.links,edges):
        pair = (("chiplet",indices[l.a]),("chiplet",indices[l.b]))
        subset = {**inputs,"topology":[edge],"packaging":{**inputs["packaging"],
                  "link_latency":math.ceil(lengths[pair]*l.wire_cycles_per_mm)}}
        latencies.update(rc.compute_link_latencies(subset,intermediate))
    intermediate["link_latencies"] = latencies
    inverse = {i:n for n,i in indices.items()}
    bandwidth = {(inverse[a[1]],inverse[b[1]]):v*hardware.network_clock_hz/8 for (a,b),v in bandwidths.items()}
    configured = {(a,b):l.bandwidth_Bps for l in hardware.links for a,b in ((l.a,l.b),(l.b,l.a))}
    bandwidth = {k:min(v,configured[k]) for k,v in bandwidth.items()}
    delays = {}
    for a,b in paths:
        if a == b:
            delays[(a,b)] = 0.0
        else:
            query = {**inputs,"traffic_by_chiplet":{(indices[a],indices[b]):1.0}}
            delays[(a,b)] = rc.compute_latency(query,intermediate)["avg"]/hardware.network_clock_hz
    area = rc.compute_area_summary(inputs,intermediate)["total_chiplet_area"]
    power = rc.compute_power_summary(inputs,intermediate)["total_power"]
    # External services reuse the physical route between attached compute
    # gateways. Off-chip service is separately charged by the oracle.
    attachments = {p.id:p.attached_to for p in hardware.memory_ports}
    physical_delays = delays.copy()
    for a in hardware.node_map:
        for b in hardware.node_map:
            delays[(a,b)] = physical_delays[(attachments.get(a,a),attachments.get(b,b))]
    return Network(bandwidth,delays,{(inverse[a[1]],inverse[b[1]]):v for (a,b),v in lengths.items()},
                   area,power,hashes,inputs)
