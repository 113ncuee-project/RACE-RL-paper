"""Gemmini-inspired WS service model with explicit storage and movement.

Configuration facts come from pinned RTL. Cycle formulas, bank allocation and
the chiplet mesh are research assumptions, not cycle-accurate Gemmini behavior.
This foundation returns traffic/cycles, never fabricated power/area/rewards.
"""
from dataclasses import dataclass, asdict, field
import hashlib
import math
from pathlib import Path
import numpy as np
from .gemmini_config import default_config
from .memory_hierarchy import MemoryHierarchy
from ..mesh import mesh_shape
from ..serialization import digest


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def ceildiv(a, b): return (a + b - 1) // b


def padded(value, dim=16): return ceildiv(value, dim) * dim


@dataclass(frozen=True)
class GemmShape:
    m: int
    n: int
    k: int

    def __post_init__(self):
        for name, value in asdict(self).items(): positive_int(value, name)

    @property
    def logical_macs(self): return self.m * self.n * self.k


@dataclass(frozen=True)
class GemminiAction:
    """A complete selected blocking/residency/placement transaction."""
    m_tile: int = 16
    n_tile: int = 16
    k_tile: int = 16
    retain_weights: bool = True
    double_buffer: bool = True
    placement: tuple[int, ...] = ()
    dataflow: str = "weight_stationary"

    def __post_init__(self):
        object.__setattr__(self, "placement", tuple(self.placement))
        for name in ("m_tile", "n_tile", "k_tile"):
            value = getattr(self, name)
            positive_int(value, name)
            if value % 16: raise ValueError("blocking dimensions must be multiples of 16; problem tails are supported")
        if type(self.retain_weights) is not bool or type(self.double_buffer) is not bool:
            raise ValueError("residency and buffering must be booleans")
        if self.dataflow != "weight_stationary":
            raise ValueError("OS exists in public hardware but is not yet implemented in this executor")
        if any(type(x) is not int or x < 0 for x in self.placement):
            raise ValueError("placement must contain nonnegative chiplet indices")


@dataclass(frozen=True)
class PlatformScenario:
    """Explicit scenario parameters; none is claimed as Gemmini silicon data."""
    dram_capacity_bytes: int = 1024**3
    dram_bytes_per_cycle: int = 16
    dram_first_response_cycles: int = 20
    mesh_bytes_per_cycle: int = 16
    mesh_hop_cycles: int = 2
    clock_hz: float | None = None

    def __post_init__(self):
        for name in ("dram_capacity_bytes", "dram_bytes_per_cycle", "mesh_bytes_per_cycle"):
            positive_int(getattr(self, name), name)
        for name in ("dram_first_response_cycles", "mesh_hop_cycles"):
            value = getattr(self, name)
            if type(value) is not int or value < 0: raise ValueError(f"{name} must be a nonnegative integer")
        if self.clock_hz is not None and (isinstance(self.clock_hz, bool) or not math.isfinite(self.clock_hz) or self.clock_hz <= 0):
            raise ValueError("scenario clock must be positive and finite")

    @classmethod
    def from_record(cls, record):
        return cls(**{key: record[key] for key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class CycleJob:
    id: str
    dependencies: tuple[str, ...]
    resources: tuple[str, ...]
    cycles: int
    kind: str
    metadata: dict = field(default_factory=dict)


def schedule_cycles(jobs):
    """Nonpreemptive integer-cycle calendars; no packet or RTL simulation."""
    pending = {job.id: job for job in jobs}
    if len(pending) != len(jobs): raise ValueError("duplicate job identity")
    if any(job.cycles < 0 or type(job.cycles) is not int or len(set(job.resources)) != len(job.resources)
           or any(dep not in pending or dep == job.id for dep in job.dependencies) for job in jobs):
        raise ValueError("invalid job duration, resources or dependency")
    calendars, done, events = {}, {}, []
    while pending:
        choices = []
        for name, job in pending.items():
            if not all(dep in done for dep in job.dependencies): continue
            start = max((done[dep] for dep in job.dependencies), default=0)
            while True:
                conflicts = [end for resource in job.resources for begin, end in calendars.get(resource, ())
                             if begin < start + job.cycles and end > start]
                if not conflicts or job.cycles == 0: break
                start = max(conflicts)
            choices.append((start, name))
        if not choices: raise ValueError("cyclic dependencies")
        start, name = min(choices)
        job = pending.pop(name)
        end = start + job.cycles
        done[name] = end
        for resource in job.resources:
            if job.cycles: calendars.setdefault(resource, []).append((start, end))
        events.append({"job": asdict(job), "start_cycle": start, "finish_cycle": end})
    return events


def xy_route(chiplets, source, destination):
    rows, columns = mesh_shape(chiplets)
    if any(type(x) is not int or not 0 <= x < chiplets for x in (source, destination)):
        raise ValueError("unknown mesh endpoint")
    path, node = [], source
    while node % columns != destination % columns:
        other = node + (1 if destination % columns > node % columns else -1)
        path.append((node, other)); node = other
    while node // columns != destination // columns:
        other = node + (columns if destination // columns > node // columns else -columns)
        path.append((node, other)); node = other
    return tuple(path)


class GemminiModel:
    def __init__(self, *, chiplets=1, config=None, platform=None, memory=None):
        positive_int(chiplets, "chiplets")
        if chiplets > 16: raise ValueError("foundation mesh supports 1–16 compute chiplets")
        self.chiplets = chiplets
        self.config = config or default_config()
        self.platform = platform or PlatformScenario()
        self.memory = memory or MemoryHierarchy()

    def compile(self, shape, action):
        """Only compile the supplied action; no candidate search or PPA predictor."""
        c, p, h = self.config, self.platform, self.memory
        groups = [(m, min(action.m_tile, shape.m-m), n, min(action.n_tile, shape.n-n))
                  for n in range(0, shape.n, action.n_tile) for m in range(0, shape.m, action.m_tile)]
        if len(action.placement) != len(groups) or any(node >= self.chiplets for node in action.placement):
            raise ValueError("action must supply exactly one valid chiplet for every output block")
        a_rows, b_rows = padded(shape.m), padded(shape.k)
        dram_allocated = a_rows*padded(shape.k) + b_rows*padded(shape.n) + a_rows*padded(shape.n)*4
        if dram_allocated > p.dram_capacity_bytes:
            raise ValueError("packed A/B/C allocations exceed external capacity")
        jobs, flows, capacities, weight_cache = [], [], [], {}
        last_group = {}
        def job(name, dependencies, resources, cycles, kind, metadata=None):
            jobs.append(CycleJob(name, tuple(dict.fromkeys(dependencies)), tuple(resources),
                                 int(cycles), kind, metadata or {}))
            return name

        def sp_resources(node, banks): return tuple(f"c{node}.gb.bank{b}" for b in banks)

        def movement(source, destination, amount, operand):
            return {"source": source, "destination": destination, "bytes": amount, "operand": operand}

        def access(movements):
            counts = {level: 0 for level in ("gb", "lb", "rf", "accumulator")}
            for move in movements:
                for endpoint in (move["source"], move["destination"]):
                    level = endpoint.split(".", 1)[0]
                    if level in counts: counts[level] += move["bytes"]
            return counts

        if h.rf_a_bytes < 256 or h.rf_b_bytes < 256 or h.rf_psum_bytes < 1024:
            raise ValueError("RF cannot hold the 16x16 INT8/INT32 microtiles")
        if h.lb_output_bytes < 1024:
            raise ValueError("output LB cannot hold one INT32 microtile")

        def dma(name, node, rows, row_bytes, dependencies, direction, operand, banks=()):
            # Packed rows are aligned to DIM. No implicit cache/coalescing is assumed.
            amount = rows * row_bytes
            beats = rows * ceildiv(row_bytes, c.dma_beat_bytes)
            transactions = rows * ceildiv(row_bytes, c.dma_max_bytes)
            source, destination = (0, node) if direction == "read" else (node, 0)
            route = xy_route(self.chiplets, source, destination)
            bandwidth = min(p.dram_bytes_per_cycle, c.dma_beat_bytes, h.gb_bank_bytes_per_cycle*max(1,len(banks)),
                            p.mesh_bytes_per_cycle if route else c.dma_beat_bytes)
            # Streaming through the path, with atomic resource reservations.
            cycles = ceildiv(amount, bandwidth) + transactions*p.dram_first_response_cycles + len(route)*p.mesh_hop_cycles
            resources = ("dram.channel0", f"c{node}.dma.{direction}",
                         *sp_resources(node, (0,1) if direction=="write" else banks),
                         *(f"d2d.c{a}>c{b}" for a, b in route))
            metadata = {"chiplet":node,"gb_banks":list(banks) if direction=="read" else [0,1],
                        "operand": operand, "bytes": amount, "dma_beats": beats,
                        "transactions": transactions, "route": route, "direction": direction,
                        "timing_assumption": "serial transaction responses; streamed path with atomic reservations",
                        "movements": [movement("dram" if direction=="read" else "gb.C",
                            "gb."+operand if direction=="read" else "dram", amount, operand)],
                        "storage_access_bytes": {"gb":amount,"lb":0,"rf":0,"accumulator":0}}
            flows.append({"job": name, "chiplet": node, **metadata})
            return job(name, dependencies, resources, cycles, "dma_"+direction, metadata)

        def load(name, node, rows, row_bytes, banks, dependencies, operand):
            return dma(name+".external", node, rows, row_bytes, dependencies, "read", operand, banks)

        def local_copy(name, node, amount, gb_banks, lb_bank, dependencies, operand):
            moves = [movement("gb."+operand, "lb."+operand+str(lb_bank), amount, operand)]
            return job(name, dependencies, (*sp_resources(node, gb_banks), f"c{node}.lb.bank{lb_bank}"),
                       ceildiv(amount, min(h.gb_bank_bytes_per_cycle*len(gb_banks), h.lb_bank_bytes_per_cycle)),
                       "gb_to_lb", {"chiplet":node,"gb_banks":list(gb_banks),"bytes":amount,"movements":moves,
                                    "storage_access_bytes":access(moves)})

        for index, ((m, mm, n, nn), node) in enumerate(zip(groups, action.placement)):
            mp, np_, kp = padded(mm), padded(nn), padded(min(action.k_tile, shape.k))
            a_size = mp*kp
            b_size = padded(shape.k)*np_ if action.retain_weights else kp*np_
            slots = 2 if action.double_buffer else 1
            b_capacity = 2*c.scratchpad_bank_bytes if action.retain_weights else c.scratchpad_bank_bytes
            if a_size > c.scratchpad_bank_bytes or b_size > b_capacity:
                raise ValueError("A panel or resident B exceeds allocated GB scratchpad banks; choose a different action")
            if a_size > h.lb_operand_slot_bytes or kp*np_ > h.lb_operand_slot_bytes:
                raise ValueError("A or B panel exceeds local buffer slot capacity; choose smaller M/N/K blocking")
            acc_size = mp*np_*4
            if acc_size > c.accumulator_bytes:
                raise ValueError("output partial sums exceed accumulator capacity")
            if acc_size > 2*c.scratchpad_bank_bytes:
                raise ValueError("output exceeds GB staging capacity")
            capacities.append({"output_block": index, "chiplet": node,
                "a_slot_bytes": a_size, "a_slots": slots, "a_bank_bytes": c.scratchpad_bank_bytes,
                "b_bytes": b_size, "b_slots": 1 if action.retain_weights else slots,
                "b_allocation_bytes": b_capacity, "accumulator_bytes": acc_size,
                "accumulator_capacity_bytes": c.accumulator_bytes,
                "gb_output_bytes":acc_size, "gb_output_capacity_bytes":2*c.scratchpad_bank_bytes,
                "lb_a_slot_bytes":a_size, "lb_b_slot_bytes":kp*np_,
                "lb_operand_slot_capacity_bytes":h.lb_operand_slot_bytes,
                "lb_a_slots":slots,"lb_b_slots":slots,
                "lb_output_peak_bytes":1024,"lb_output_capacity_bytes":h.lb_output_bytes,
                "rf_a_peak_bytes":256,"rf_b_peak_bytes":256,"rf_psum_peak_bytes":1024})
            prefix = f"g{index}.c{node}"
            base = (last_group[node],) if node in last_group else ()
            # One resident N-block per chiplet. A change of N invalidates it.
            if action.retain_weights:
                if weight_cache.get(node, (None,))[0] != n:
                    full_b = load(prefix+".B.resident", node, padded(shape.k), np_, (2, 3), base, "B")
                    weight_cache[node] = (n, full_b)
                else:
                    full_b = weight_cache[node][1]
            slot_release, compute_previous = {}, None
            for kk, k in enumerate(range(0, shape.k, action.k_tile)):
                kk_actual = min(action.k_tile, shape.k-k)
                kpad = padded(kk_actual)
                slot = kk % slots
                banks_a = (slot,)
                banks_b = (2, 3) if action.retain_weights else (2+slot,)
                admission = (*base, *((slot_release[slot],) if slot in slot_release else ()))
                a_ready = load(prefix+f".k{kk}.A", node, mp, kpad, banks_a, admission, "A")
                b_ready = full_b if action.retain_weights else load(
                    prefix+f".k{kk}.B", node, kpad, np_, banks_b, admission, "B")
                a_ready = local_copy(prefix+f".k{kk}.A.lb", node, mp*kpad, banks_a, slot, (a_ready,), "A")
                b_ready = local_copy(prefix+f".k{kk}.B.lb", node, kpad*np_, banks_b, 2+slot,
                                     (b_ready,*admission), "B")
                mb, nb, kb = mp//16, np_//16, kpad//16
                # Nested WS microprogram: N16, K16, then M16. Weights stay in
                # RF for all M16 uses; A stays in LB across N16 uses. Partial
                # sums are explicitly saved/reloaded through LB to/from ACC.
                lb_a_reads, lb_b_reads = mp*kpad*nb, kpad*np_
                acc_writes = kb*mp*np_*4
                acc_reads = (kb-(1 if k == 0 else 0))*mp*np_*4
                moves = [movement(f"lb.A{slot}", "rf.A", lb_a_reads, "A"),
                         movement(f"lb.B{2+slot}", "rf.B", lb_b_reads, "B"),
                         movement("accumulator", "lb.C", acc_reads, "psum"),
                         movement("lb.C", "rf.psum", acc_reads, "psum"),
                         movement("rf.psum", "lb.C", acc_writes, "psum"),
                         movement("lb.C", "accumulator", acc_writes, "psum")]
                moves = [move for move in moves if move["bytes"]]
                counts = access(moves)
                macs = mp*np_*kpad
                rf_compute = {"a_read_bytes":macs,"b_read_bytes":macs,
                              "psum_read_bytes":(macs-(mp*np_ if k==0 else 0))*4,
                              "psum_write_bytes":macs*4}
                counts["rf"] += sum(rf_compute.values())
                preload_cycles = nb*kb*16
                array_cycles = mb*nb*kb*(3*16-2)
                operand_cycles = ceildiv(lb_a_reads, min(h.lb_bank_bytes_per_cycle,h.rf_operand_bytes_per_cycle)) + ceildiv(lb_b_reads, min(h.lb_bank_bytes_per_cycle,h.rf_operand_bytes_per_cycle))
                psum_cycles = 2*ceildiv(acc_reads, min(h.lb_bank_bytes_per_cycle,h.rf_psum_bytes_per_cycle,h.accumulator_bank_bytes_per_cycle*c.accumulator_banks)) + 2*ceildiv(acc_writes, min(h.lb_bank_bytes_per_cycle,h.rf_psum_bytes_per_cycle,h.accumulator_bank_bytes_per_cycle*c.accumulator_banks))
                rf_cycles = max(ceildiv(macs,h.rf_operand_bytes_per_cycle),ceildiv(rf_compute["psum_read_bytes"],h.rf_psum_bytes_per_cycle),ceildiv(rf_compute["psum_write_bytes"],h.rf_psum_bytes_per_cycle))
                cycles = preload_cycles+max(array_cycles,rf_cycles)+operand_cycles+psum_cycles
                deps = (a_ready, b_ready, *((compute_previous,) if compute_previous else ()))
                acc_resources = tuple(f"c{node}.acc.bank{bank}.{port}" for bank in range(c.accumulator_banks)
                                      for port in ("read", "write"))
                rf_resources = tuple(f"c{node}.rf.{operand}.{port}" for operand in ("A","B","psum") for port in ("read","write"))
                metadata = {"chiplet": node, "output_block": index, "m_range": (m, m+mm),
                    "n_range": (n, n+nn), "k_range": (k, k+kk_actual),
                    "logical_macs": mm*nn*kk_actual, "padded_macs": macs,
                    "lb_a_read_bytes":lb_a_reads,"lb_b_read_bytes":lb_b_reads,
                    "acc_read_bytes": acc_reads, "acc_write_bytes": acc_writes,
                    "weight_preload_elements": kpad*np_, "preload_cycles": preload_cycles,
                    "array_service_cycles": array_cycles, "operand_copy_cycles":operand_cycles,
                    "psum_copy_cycles":psum_cycles,"rf_compute_service_cycles":rf_cycles,
                    "movements":moves,"rf_compute_access_bytes":rf_compute,"storage_access_bytes":counts,
                    "microprogram":"N16 -> K16 -> M16; preload B once per N16/K16, A per M16, save psum per K16",
                    "buffer_slot": slot, "cycle_formula_status": "UNVALIDATED_ANALYTICAL_ASSUMPTION"}
                compute_previous = job(prefix+f".k{kk}.execute", deps,
                    (f"c{node}.array", f"c{node}.lb.bank{slot}",f"c{node}.lb.bank{2+slot}",
                     f"c{node}.lb.C",*rf_resources,*acc_resources), cycles, "execute_ws", metadata)
                slot_release[slot] = compute_previous
            moves = [movement("accumulator","lb.C",acc_size,"C"),movement("lb.C","gb.C",acc_size,"C")]
            # Stream 16x16 output microtiles through LB; never allocate the
            # whole output tile in LB. GB banks 0/1 are free after final execute.
            drain = job(prefix+".C.stage", (compute_previous,),
                (*tuple(f"c{node}.acc.bank{bank}.read" for bank in range(c.accumulator_banks)),
                 f"c{node}.lb.C",*sp_resources(node,(0,1))),
                ceildiv(acc_size,min(h.accumulator_bank_bytes_per_cycle*c.accumulator_banks,h.lb_bank_bytes_per_cycle))+
                ceildiv(acc_size,min(h.lb_bank_bytes_per_cycle,2*h.gb_bank_bytes_per_cycle)), "acc_lb_gb_output",
                {"chiplet":node,"bytes":acc_size,"chunk_bytes":1024,"movements":moves,
                 "storage_access_bytes":access(moves),"format":"INT32 full-width output; no requantization"})
            last_group[node] = dma(prefix+".C.external", node, mp, np_*4, (drain,), "write", "C")
        return jobs, flows, capacities, dram_allocated

    def evaluate(self, shape, action):
        jobs, flows, capacities, dram_allocated = self.compile(shape, action)
        events = schedule_cycles(jobs)
        executes = [event["job"]["metadata"] for event in events if event["job"]["kind"] == "execute_ws"]
        cycles = max((event["finish_cycle"] for event in events), default=0)
        totals = {key: sum(item[key] for item in executes) for key in (
            "logical_macs", "padded_macs", "lb_a_read_bytes", "lb_b_read_bytes", "acc_read_bytes", "acc_write_bytes")}
        if totals["logical_macs"] != shape.logical_macs:
            raise ValueError("logical operation coverage mismatch")
        bytes_by_operand = {operand: sum(flow["bytes"] for flow in flows if flow["operand"] == operand)
                            for operand in ("A", "B", "C")}
        interconnect_bytes = sum(flow["bytes"]*len(flow["route"]) for flow in flows)
        storage_accesses = {level:sum(event["job"]["metadata"].get("storage_access_bytes",{}).get(level,0) for event in events)
                            for level in ("gb","lb","rf","accumulator")}
        movements = {}
        for event in events:
            for move in event["job"]["metadata"].get("movements",()):
                boundary = move["source"].split(".",1)[0]+"->"+move["destination"].split(".",1)[0]
                movements[boundary] = movements.get(boundary,0)+move["bytes"]
        provenance = self.config.provenance()
        model_source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return {"model": "race_gemmini_hierarchy_ws_v2", "shape": asdict(shape), "action": asdict(action),
            "chiplets": self.chiplets, "mesh_shape": mesh_shape(self.chiplets),
            "identity": digest({"shape": shape, "action": action, "chiplets": self.chiplets,
                                "configuration": provenance, "platform": self.platform, "model_version": 2, "memory_hierarchy":self.memory.record(),
                                "model_source_sha256": model_source_hash}),
            "configuration": asdict(self.config), "source_provenance": provenance,
            "model_source_sha256": model_source_hash, "memory_hierarchy":self.memory.record(),
            "platform_scenario": asdict(self.platform), "platform_status": "RESEARCH_ASSUMPTION",
            "modeled_cycles": cycles, "scenario_latency_s": cycles/self.platform.clock_hz if self.platform.clock_hz else None,
            "traffic": {"external_bytes": sum(bytes_by_operand.values()), "by_operand": bytes_by_operand,
                "directed_link_byte_hops": interconnect_bytes,
                "dma_transactions": sum(flow["transactions"] for flow in flows),
                "storage_access_bytes":storage_accesses, "hierarchy_movement_bytes":movements},
            "operations": totals, "array_padding_utilization": totals["logical_macs"]/totals["padded_macs"],
            "storage": {"blocks": capacities, "external_allocation_bytes": dram_allocated},
            "events": events, "transfers": flows,
            "ppa": {"average_power_w": None, "energy_j": None, "area_mm2": None,
                "status": "UNAVAILABLE_WITHOUT_CALIBRATION", "eligible_for_formal_rl_scoring": False},
            "coverage": {"executor": "INT8 GEMM, INT32 accumulated output, WS only",
                "timing": "analytical service calendars; not RTL cycle accuracy",
                "whole_cnn": False, "output_stationary": False, "activation_residency": False,
                "peer_to_peer_activation": False, "reduction_across_chiplets": False,
                "queue_tlb_dram_protocol": False, "cross_layer_fusion": False},
            "assumptions": ["A/B/C in external memory are zero-padded packed matrices aligned to 16 elements.",
                "INT32 C is exported; requantization, bias and nonlinear operators are excluded.",
                "GB uses pinned scratchpad capacity, with A bank 0/1 and retained B bank 2/3; output stages in free A banks.",
                "LB/RF organization is a RACE-RL research extension, not the original Gemmini RTL.",
                "Each LB A/B slot must hold the entire selected panel; RF holds one microtile. Oversize actions are rejected, not implicitly retiled.",
                "Weights reuse in GB across M groups and in RF across M16; A reuses in LB across N16.",
                "RF and SRAM accumulator are distinct: partial sums traverse RF -> LB -> ACC and back after each K16.",
                "Output microtiles stream ACC -> LB -> GB -> DRAM. There is no direct DRAM/PE bypass.",
                "Single-port GB/LB banks are exclusive; copies reserve both endpoints; RF/ACC read/write ports are separate.",
                "Output groups on each chiplet serialize. K-panel copies overlap computation only in different slots.",
                "Microprogram aggregated service phases serialize operand/psum copies; analytical conservative reservations need RTL validation.",
                "External channel is shared by reads/writes; no implicit DRAM cache or multicast.",
                "Mesh is a project extension. Gemmini itself is not a chiplet architecture."]}

    @staticmethod
    def require_ppa(record):
        if not record["ppa"]["eligible_for_formal_rl_scoring"]:
            raise ValueError("Gemmini foundation lacks calibrated area/energy; formal PPA scoring is blocked")


def int8_array(array):
    value = np.asarray(array)
    if value.dtype.kind not in "iu" or value.size == 0 or value.min() < -128 or value.max() > 127:
        raise ValueError("explicit signed INT8-compatible integers required; no implicit CNN quantization")
    return value.astype(np.int8)


def execute_gemm(a, b, action):
    """Independent arithmetic executor; not RTL, accuracy metrics or a timing simulator."""
    a, b = int8_array(a), int8_array(b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("GEMM matrix dimensions do not match")
    result = np.zeros((a.shape[0], b.shape[1]), dtype=np.int64)
    for m in range(0, a.shape[0], action.m_tile):
        for n in range(0, b.shape[1], action.n_tile):
            for k in range(0, a.shape[1], 16):
                result[m:m+action.m_tile, n:n+action.n_tile] += (
                    a[m:m+action.m_tile, k:k+16].astype(np.int64) @ b[k:k+16, n:n+action.n_tile].astype(np.int64))
                block = result[m:m+action.m_tile, n:n+action.n_tile]
                if block.min() < -(2**31) or block.max() >= 2**31:
                    raise ValueError("INT32 partial-sum overflow; wrapping is not modeled")
    return result.astype(np.int32)


def lower_conv2d(inputs, weights, *, stride=1, padding=0):
    """Explicit im2col reference, dense convolution only; preprocessing cost excluded."""
    x, w = int8_array(inputs), int8_array(weights)
    if x.ndim != 4 or w.ndim != 4 or x.shape[1] != w.shape[1]:
        raise ValueError("dense NCHW/OIHW convolution required; grouped/depthwise not covered")
    positive_int(stride, "stride")
    if type(padding) is not int or padding < 0: raise ValueError("padding must be a nonnegative integer")
    batches, channels, height, width = x.shape
    outputs, _, kh, kw = w.shape
    oh, ow = (height+2*padding-kh)//stride+1, (width+2*padding-kw)//stride+1
    if oh < 1 or ow < 1: raise ValueError("kernel does not fit input")
    padded_input = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    rows = [padded_input[batch, :, y*stride:y*stride+kh, z*stride:z*stride+kw].reshape(-1)
            for batch in range(batches) for y in range(oh) for z in range(ow)]
    a, b = np.stack(rows), w.reshape(outputs, -1).T.copy()
    return a, b, {"output_shape": (batches, outputs, oh, ow), "gemm_shape": asdict(GemmShape(*(
        batches*oh*ow, outputs, channels*kh*kw))), "im2col_bytes": a.nbytes,
        "original_input_bytes": x.nbytes, "lowering": "explicit_im2col",
        "preprocessing_latency_energy": "NOT_MODELED; GEMM-stage results are not whole-convolution PPA"}
