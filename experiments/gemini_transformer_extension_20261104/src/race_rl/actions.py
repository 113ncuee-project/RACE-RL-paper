"""Dependency-connected coupled atomic transactions; no PPA proposal ranking."""
from dataclasses import dataclass
from itertools import combinations
from .mapping import Mapping
from .tiles import partition, requirements


@dataclass(frozen=True)
class Action:
    root: str
    scope: tuple[str, ...]
    kind: str
    target: str
    count: int = 1
    compute_target: str = ""


def dependency_edges(workload,mapping):
    owners = {}
    for t in mapping.tiles:
        owners.setdefault(workload.operator_map[t.operator].output,[]).append(t)
    edges = set()
    for t in mapping.tiles:
        for name,region in requirements(workload,t):
            for s in owners.get(name,()):
                if s.output.intersection(region) is not None:
                    edges.add((s.id,t.id))
    return tuple(sorted(edges))


def scopes(workload,mapping,max_scope=4):
    """Bounded connected expansion from each root, independent of all oracle metrics."""
    edges = dependency_edges(workload,mapping)
    neighbors = {t.id:set() for t in mapping.tiles}
    for a,b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    result = set()
    for root in sorted(neighbors):
        frontier = {(root,)}
        for _ in range(max_scope):
            next_frontier = set()
            for scope in sorted(frontier):
                result.add((root,scope))
                candidates = set().union(*(neighbors[t] for t in scope))-set(scope)
                for t in sorted(candidates):
                    next_frontier.add(tuple(sorted((*scope,t))))
            frontier = next_frontier
    return tuple(sorted(result))


def apply(workload,hardware,mapping,action):
    by_id = {t.id:t for t in mapping.tiles}
    if action.root not in action.scope or not set(action.scope) <= set(by_id):
        raise ValueError("invalid action scope")
    selected = set(action.scope)
    if len(selected) != len(action.scope):
        raise ValueError("duplicate tile in action scope")
    visited = {action.root}
    edges = dependency_edges(workload,mapping)
    while True:
        expanded = visited | {b for a,b in edges if a in visited and b in selected} | {a for a,b in edges if b in visited and a in selected}
        if expanded == visited:
            break
        visited = expanded
    if visited != selected:
        raise ValueError("scope is not dependency connected")
    return _edit_connected(workload,hardware,mapping,action)


def _edit_connected(workload,hardware,mapping,action):
    """Apply a structurally generated connected scope without re-enumerating all roots."""
    by_id = {t.id:t for t in mapping.tiles}
    placement,storage = mapping.placements.copy(),mapping.storages.copy()
    tiles = mapping.tiles
    if action.kind == "cut_migrate":
        if action.target not in hardware.compute_ids:
            raise ValueError("invalid compute target")
        for t in action.scope: placement[t] = action.target
    elif action.kind == "memory_affinity":
        if action.target not in hardware.memory_ids:
            raise ValueError("invalid storage target")
        if action.compute_target not in hardware.compute_ids:
            raise ValueError("memory affinity must jointly select a compute target")
        for name in action.scope:
            placement[name] = action.compute_target
            op = workload.operator_map[by_id[name].operator]
            for tensor in (*op.inputs,op.output): storage[tensor] = action.target
            for parameter in workload.parameter_shapes(op): storage[parameter] = action.target
    elif action.kind == "work_share":
        # Rebuild whole selected operators; partial layer retile would lose ownership.
        operators = {by_id[t].operator for t in action.scope}
        if set(action.scope) != {t.id for t in mapping.tiles if t.operator in operators}:
            raise ValueError("work_share requires full operator ownership in scope")
        if action.target not in hardware.compute_ids:
            raise ValueError("invalid work_share target")
        counts = {o.id:sum(t.operator == o.id for t in mapping.tiles) for o in workload.operators}
        for o in operators: counts[o] = action.count
        rebuilt = partition(workload,counts)
        tiles = tuple(t for t in mapping.tiles if t.operator not in operators)+tuple(t for t in rebuilt if t.operator in operators)
        placement = {t.id:placement[t.id] for t in mapping.tiles if t.operator not in operators}
        start = hardware.compute_ids.index(action.target)
        for i,t in enumerate(t for t in tiles if t.operator in operators):
            placement[t.id] = hardware.compute_ids[(start+i)%len(hardware.compute_ids)]
    else:
        raise ValueError("unknown transaction kind")
    candidate = Mapping(tuple(sorted(tiles,key=lambda t:t.id)),tuple(sorted(placement.items())),tuple(sorted(storage.items())))
    candidate.validate(workload,hardware)
    return candidate


def legal_actions(workload,hardware,mapping,max_scope=4):
    """Structural masks without rebuilding or validating a whole design per leaf.

    The selected transaction still goes through apply()+Mapping.validate().
    No oracle is used here. Keep the exact action support of the exhaustive
    compiler, including its canonical tuple ordering.
    """
    result = []
    by_id = {t.id:t for t in mapping.tiles}
    owners = {o.id:{t.id for t in mapping.tiles if t.operator==o.id} for o in workload.operators}
    placement,storage = mapping.placements,mapping.storages
    canonical = (tuple(sorted(mapping.tiles,key=lambda t:t.id))==mapping.tiles
        and tuple(sorted(mapping.placement))==mapping.placement
        and tuple(sorted(mapping.storage))==mapping.storage)
    groups = {}
    from .architecture import operator_tiles
    for root,scope in scopes(workload,mapping,max_scope):
        for dst in hardware.compute_ids:
            if not canonical or any(placement[t]!=dst for t in scope):
                result.append(Action(root,scope,'cut_migrate',dst))
        objects=set()
        for name in scope:
            op=workload.operator_map[by_id[name].operator]
            objects.update((*op.inputs,op.output,*workload.parameter_shapes(op)))
        for dst in hardware.memory_ids:
            moved_storage=any(storage[name]!=dst for name in objects)
            for compute in hardware.compute_ids:
                if not canonical or moved_storage or any(placement[t]!=compute for t in scope):
                    result.append(Action(root,scope,'memory_affinity',dst,compute_target=compute))
        operators={by_id[t].operator for t in scope}
        if set(scope)!=set().union(*(owners[o] for o in operators)):continue
        for dst in hardware.compute_ids:
            for count in (1,2):
                try:
                    selected=[]
                    # _edit_connected preserves workload order for round-robin.
                    for op in workload.operators:
                        if op.id not in operators:continue
                        key=(op.id,count)
                        if key not in groups:groups[key]=operator_tiles(workload,op.id,count)
                        selected.extend(groups[key])
                except ValueError:continue
                start=hardware.compute_ids.index(dst)
                changed=(not canonical or set(t.id for t in selected)!=set(scope)
                    or any(by_id[t.id]!=t or placement[t.id]!=hardware.compute_ids[(start+i)%len(hardware.compute_ids)]
                        for i,t in enumerate(selected)))
                if changed:result.append(Action(root,scope,'work_share',dst,count))
    return tuple(result)
