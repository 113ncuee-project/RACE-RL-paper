"""Conservative feasible-start construction with no PPA oracle or cost ranking."""
from math import prod,ceil
from .tiles import partition,requirements,parameter_regions
from .mapping import Mapping


def conservative_initial_mapping(workload,hardware):
    """Bound per-tile local footprint and worst-case full external residency.

    Must be used with serial_node_tiles scheduling. This is a deterministic
    compiler contract; it does not guarantee success for arbitrary hardware.
    """
    workload.validate()
    hardware.validate()
    local_capacity = min(hardware.node_map[name].memory_bytes for name in hardware.compute_ids)
    counts = {o.id:1 for o in workload.operators}

    def footprint(tile):
        op = workload.operator_map[tile.operator]
        return (sum(r.elements*workload.tensor_map[name].bytes_per_element for name,r in requirements(workload,tile))+
                sum(r.elements*workload.tensor_map[op.output].bytes_per_element for _,r in parameter_regions(workload,tile))+
                tile.output.elements*workload.tensor_map[op.output].bytes_per_element)

    # Validate each operator independently instead of repartitioning the full
    # graph repeatedly. All tiles still receive final global coverage checking.
    for op in workload.operators:
        shape = workload.tensor_map[op.output].shape
        axis = 2 if len(shape) == 4 else 1
        # Candidate count enumeration is structural; all choices are considered
        # in ascending count, never ranked by energy or delay.
        for count in range(1,shape[axis]+1):
            from .tiles import Tile,Region,whole
            group = []
            for i in range(count):
                bounds = list(whole(shape).bounds)
                bounds[axis] = (i*shape[axis]//count,(i+1)*shape[axis]//count)
                group.append(Tile(f'{op.id}:{i}',op.id,Region(tuple(bounds))))
            if all(footprint(t) <= local_capacity for t in group):
                counts[op.id] = count
                break
        else:
            raise ValueError(f"no supported output partition fits local capacity: {op.id}; channel reduction is not supplied for free")
    items = {t.id:t.bytes for t in workload.tensors}
    items.update({name:prod(shape)*workload.tensor_map[workload.operator_map[name.split(':',1)[1]].output].bytes_per_element
                  for name,shape in workload.parameter_map.items()})
    remaining = {name:hardware.node_map[name].memory_bytes for name in hardware.memory_ids}
    store = {}
    for name,amount in sorted(items.items(),key=lambda item:(-item[1],item[0])):
        endpoint = next((node for node in hardware.memory_ids if remaining[node] >= amount),None)
        if endpoint is None:
            raise ValueError(f"worst-case external residency cannot fit {name}: {amount} bytes")
        store[name] = endpoint
        remaining[endpoint] -= amount
    tiles = partition(workload,counts)
    location = {o.id:hardware.compute_ids[i%len(hardware.compute_ids)] for i,o in enumerate(workload.operators)}
    mapping = Mapping(tiles,tuple((t.id,location[t.operator]) for t in tiles),tuple(sorted(store.items())))
    mapping.validate(workload,hardware)
    metadata = {"rule":"ascending structural tile-count footprint; first-fit-decreasing worst-case external bytes; operator-index placement",
        "scheduling_policy_required":"serial_node_tiles","oracle_calls_for_compilation":0,
        "tile_counts":counts,"max_tile_footprint_bytes":max(footprint(t) for t in tiles),
        "local_capacity_bytes":local_capacity,"external_reserved_bytes":{
            name:hardware.node_map[name].memory_bytes-remaining[name] for name in hardware.memory_ids},
        "limitations":"no exact packet timing, cross-destination multicast, manufacture-density calibration or optimal packing guarantee"}
    return mapping,metadata
