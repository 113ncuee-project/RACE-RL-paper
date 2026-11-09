"""Declared synthetic CNN evaluation fixture, not a calibrated accelerator profile."""
from dataclasses import replace
import math
from .examples import toy_hardware
from .tiles import partition,requirements,parameter_regions
from .mapping import Mapping
from .oracle import Oracle
from .serialization import digest


def cnn_hardware(compute_count=2,topology="mesh"):
    if topology == 'mesh':
        from .mesh import mesh_hardware
        return mesh_hardware(compute_count)
    h = toy_hardware(compute_count,topology)
    h = replace(h,nodes=tuple(replace(n,memory_bytes=32*1024**2 if n.kind == 'compute' else 1024**3)
                              for n in h.nodes))
    return h


def evaluate_cnn_fixture(workload,compute_count=2,topology="mesh",guarded=False):
    h = cnn_hardware(compute_count,topology)
    if guarded:
        from .initialization import conservative_initial_mapping
        mapping,compiler = conservative_initial_mapping(workload,h)
        oracle = Oracle(workload,h,budget=1,scheduling_policy='serial_node_tiles')
        record = oracle.evaluate(mapping)
        return {"status":"SYNTHETIC_GUARDED_ANALYTICAL_VALIDATION_ONLY","hardware":h,"mapping":mapping,
            "record":record,"compiler":compiler,"query_accounting":oracle.request_log,
            "oracle_evaluations":oracle.evaluations,"native_setup_time_s":oracle.setup_time_s,
            "coefficient_origin":"same synthetic capacities/geometry/rates as the unrestricted fixture; not calibrated"}
    # Deterministic initial tiling based only on structural working-set bytes,
    # with no PPA calls or parameter optimization. NF Linear may split output
    # features; Conv splits height and retains full per-tile channel weights.
    counts = {}
    for op in workload.operators:
        shape = workload.tensor_map[op.output].shape
        if op.kind == 'Linear':
            full_params = sum(math.prod(s) for s in workload.parameter_shapes(op).values())*workload.tensor_map[op.output].bytes_per_element
            input_bytes = sum(workload.tensor_map[name].bytes for name in op.inputs)
            available = h.node_map[h.compute_ids[0]].memory_bytes-input_bytes-workload.tensor_map[op.output].bytes
            if available <= 0:
                raise ValueError("declared fixture cannot fit Linear activations")
            counts[op.id] = min(shape[1],max(1,math.ceil(full_params/available)))
        else:
            counts[op.id] = min(2,shape[2] if len(shape) == 4 else shape[1])
    tiles = partition(workload,counts)
    stores = tuple(sorted((name,h.memory_ids[0]) for name in (*workload.tensor_map,*workload.parameter_map)))
    # Keep same-layer tiles on one chiplet to share identical constant reads.
    locations = {op.id:h.compute_ids[i%len(h.compute_ids)] for i,op in enumerate(workload.operators)}
    mapping = Mapping(tiles,tuple((t.id,locations[t.operator]) for t in tiles),stores)
    oracle = Oracle(workload,h,budget=1)
    record = oracle.evaluate(mapping)
    return {"status":"SYNTHETIC_ANALYTICAL_VALIDATION_ONLY","hardware":h,"mapping":mapping,
        "mapping_hash":digest(mapping),"tiling_counts":counts,"record":record,
        "query_accounting":oracle.request_log,"oracle_evaluations":oracle.evaluations,
        "native_setup_time_s":oracle.setup_time_s,
        "coefficient_origin":"all hardware/energy values deliberately synthetic; no signoff or silicon accuracy",
        "initial_mapping_rule":"fixed operator-index placement; structural Linear footprint split; no PPA selection"}
