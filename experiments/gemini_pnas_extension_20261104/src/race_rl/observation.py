"""Versioned graph features and dependency-to-link attribution from traffic IDs."""
from dataclasses import dataclass
from math import log1p, log
from .actions import dependency_edges
from .workload import OPERATOR_KINDS

TILE_FEATURE_SIZE = len(OPERATOR_KINDS)+6


@dataclass(frozen=True)
class GraphObservation:
    schema_version: int
    tile_ids: tuple
    hardware_ids: tuple
    tile_features: tuple
    hardware_features: tuple
    dependency_edge_index: tuple
    dependency_edge_features: tuple
    hardware_edge_index: tuple
    hardware_edge_features: tuple
    placement_edge_index: tuple
    dependency_link_relations: tuple
    global_features: tuple
    legal_actions: tuple
    legal_action_masks: tuple
    feature_valid_masks: dict
    cached_mapping_hashes: tuple = ()
    best_mapping_hash: str = ""
    memory_port_edge_index: tuple = ()
    task_features: tuple = (0.0,)*16


def observe(workload,hardware,mapping,evaluation,actions,*,current,best,remaining,budget,
            remaining_requests=1.0,cached_mapping_hashes=(),best_mapping_hash="",last_reward=0.0):
    tiles = tuple(sorted(mapping.tiles,key=lambda t:t.id))
    nodes = tuple(sorted(hardware.service_nodes,key=lambda n:n.id))
    ti = {t.id:i for i,t in enumerate(tiles)}
    hi = {n.id:i for i,n in enumerate(nodes)}
    measured = evaluation is not None
    trace = {e["job"]["id"]:e for e in evaluation.schedule_trace} if measured else {}
    diagnostics = evaluation.diagnostics.get("criticality",{}) if measured else {}
    makespan = evaluation.latency_s if measured else 1.0
    types = OPERATOR_KINDS
    features = []
    for t in tiles:
        op = workload.operator_map[t.operator]
        if measured:
            e = trace[f"compute:{t.id}"]
            meta = e["job"]["metadata"]
            diag = diagnostics[e["job"]["id"]]
            feedback = ((e['finish_s']-e['start_s'])/makespan,diag['slack_s']/makespan,float(diag['critical']))
        else:
            from .tiles import operations,requirements
            meta = {'operations':operations(workload,t),
                'input_bytes':sum(r.elements*workload.tensor_map[name].bytes_per_element for name,r in requirements(workload,t)),
                'output_bytes':t.output.elements*workload.tensor_map[op.output].bytes_per_element}
            feedback = (0.0,0.0,0.0)
        features.append((*[float(op.kind == kind) for kind in types],
            log1p(meta["operations"]),log1p(meta["input_bytes"]),log1p(meta["output_bytes"]),
            *feedback))
    hw_features = []
    for n in nodes:
        spans = [e for e in trace.values() if any(r.startswith(f"compute:{n.id}:") or
            (n.kind == 'memory' and r == f'memory:{n.id}') for r in e['job']['resources'])]
        busy = sum(e["finish_s"]-e["start_s"] for e in spans)/(makespan*n.slots)
        hw_features.append((float(n.kind == "compute"),float(n.kind == "memory"),float(n.kind == "io"),
            log1p(n.operations_per_s),log1p(n.memory_bytes),log1p(n.bandwidth_Bps),n.x_mm,n.y_mm,busy))
    dependencies = dependency_edges(workload,mapping)
    edges = tuple((ti[a],ti[b]) for a,b in dependencies)
    links = tuple((l,a,b) for l in sorted(hardware.links,key=lambda l:l.id) for a,b in ((l.a,l.b),(l.b,l.a)))
    link_index = {(l.id,a,b):i for i,(l,a,b) in enumerate(links)}
    traffic_by_edge = {key:0 for key in link_index}
    relation = []
    for flow in evaluation.traffic_ledger if measured else ():
        for route in flow["route"]:
            key = tuple(route)
            traffic_by_edge[key] += flow["bytes"]
            # Read depends on spill-write; producer identity stays explicit.
            source_ids = [dep.split("write:",1)[1] for dep in flow.get("data_dependencies",flow["dependencies"]) if dep.startswith("write:")]
            for source in source_ids:
                for dest in flow["consumer_tiles"]:
                    if (source,dest) in dependencies:
                        relation.append((dependencies.index((source,dest)),link_index[key],flow["id"],"read"))
            if flow["producer_tile"]:
                source = flow["producer_tile"]
                for i,(a,b) in enumerate(dependencies):
                    if a == source: relation.append((i,link_index[key],flow["id"],"spill_write"))
    edge_features = []
    for l,a,b in links:
        resource = f"link:{l.id}:{a}>{b}"
        using = [e for e in trace.values() if resource in e["job"]["resources"]]
        busy = sum(e["finish_s"]-e["start_s"] for e in using)/makespan
        critical = any(diagnostics[e["job"]["id"]]["critical"] for e in using)
        edge_features.append((log1p(l.bandwidth_Bps),log1p(traffic_by_edge[(l.id,a,b)]),busy,float(critical)))
    placements = tuple((ti[t.id],hi[mapping.placements[t.id]]) for t in tiles)
    global_features = (log(current),log(best),remaining/budget,1-remaining/budget,
                       evaluation.average_power_w if measured else 0.0,
                       evaluation.latency_s if measured else 0.0,1.0,remaining_requests,
                       float(measured),last_reward)
    return GraphObservation(5,tuple(ti),tuple(hi),tuple(features),tuple(hw_features),edges,
        tuple((1.0,) for _ in edges),tuple((hi[a],hi[b]) for _,a,b in links),tuple(edge_features),
        placements,tuple(sorted(set(relation))),global_features,tuple(actions),tuple(True for _ in actions),
        {"tile":tuple((True,)*(len(f)-3)+(measured,)*3 for f in features),
         "hardware":tuple((True,)*8+(measured,) for _ in hw_features),
         "link":tuple((True,)+(measured,)*3 for _ in edge_features),
         "global":(measured,measured,True,True,measured,measured,True,True,True,True),
         "task":(True,)*9+(measured,)*6+(True,)},tuple(sorted(cached_mapping_hashes)),best_mapping_hash,
         tuple(edge for p in hardware.memory_ports for edge in ((hi[p.id],hi[p.attached_to]),(hi[p.attached_to],hi[p.id]))))
