"""Synthetic fixtures for verification, never real-hardware benchmark parameters."""
from .workload import Workload,Tensor,Operator
from .hardware import Hardware,Node,Link
from .mapping import Mapping
from .tiles import partition


def toy_workload(spatial=4):
    shape = (1,2,spatial,spatial)
    return Workload(tuple(Tensor(name,shape) for name in ("x","a","b","y")),
        (Operator("conv_a","Conv",("x",),"a",(3,3),(1,1),(1,1)),
         Operator("conv_b","Conv",("a",),"b",(3,3),(1,1),(1,1)),
         Operator("residual","Add",("a","b"),"y")),("y",))


def toy_hardware(compute_count=2,topology="chain"):
    if compute_count < 1 or topology not in {"chain","star","ring"}:
        raise ValueError("invalid fixture topology")
    nodes = tuple(Node(f"c{i}","compute",i*2,0,1,1,65536,1e9,(i+1)*1e9,1,0.001)
                  for i in range(compute_count))
    nodes += (Node("m0","memory",compute_count*2,0,1,1,1048576,1e9,static_power_w=0.001),
              Node("m1","memory",compute_count*2+2,0,1,1,1048576,0.75e9,static_power_w=0.001))
    ids = [n.id for n in nodes]
    pairs = list(zip(ids,ids[1:])) if topology != "star" else [(ids[0],name) for name in ids[1:]]
    if topology == "ring": pairs += [(ids[-1],ids[0])]
    return Hardware(nodes,tuple(Link(f"l{i}",a,b,1e9) for i,(a,b) in enumerate(pairs)))


def initial_mapping(workload,hardware):
    tiles = partition(workload)
    storage = [(t.id,hardware.memory_ids[0]) for t in workload.tensors]
    storage += [(name,hardware.memory_ids[0]) for name in workload.parameter_map]
    return Mapping(tiles,tuple((t.id,hardware.compute_ids[0]) for t in tiles),tuple(sorted(storage)))
