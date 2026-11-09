"""Homogeneous synthetic chiplets in a complete rectangular mesh."""
from math import isqrt
from .hardware import Hardware, Node, Link,MemoryPort


def mesh_shape(compute_count):
    if type(compute_count) is not int or compute_count < 1:
        raise ValueError("compute count must be a positive integer")
    total = compute_count
    rows = max(d for d in range(1,isqrt(total)+1) if total % d == 0)
    return rows,total//rows


def mesh_hardware(compute_count,*,local_memory_bytes=32*1024**2,
                  external_memory_bytes=1024**3):
    """Compute-only mesh, with two external-memory service gateways.

    Closest-to-square exact factorization; prime compute counts are 1 x N.
    No missing vertices or diagonal/wraparound edges. All compute chiplets
    have identical specs across counts. Capacities are synthetic, not density
    calibrated. External ports attach to c0/cN-1 and have no physical die
    geometry or relay role. Area is compute dies, not DRAM/package area.
    """
    rows,cols = mesh_shape(compute_count)
    nodes = []
    for i in range(compute_count):
        nodes.append(Node(f'c{i}','compute',2*(i%cols),2*(i//cols),1,1,
            local_memory_bytes,1e9,operations_per_s=1e9,static_power_w=0.001))
    links = []
    for i,node in enumerate(nodes):
        if i%cols+1 < cols:
            links.append(Link(f'l{len(links)}',node.id,nodes[i+1].id,1e9))
        if i//cols+1 < rows:
            links.append(Link(f'l{len(links)}',node.id,nodes[i+cols].id,1e9))
    memory_ports = (MemoryPort('dram0','c0',external_memory_bytes,1e9,0.001),
        MemoryPort('dram1',f'c{compute_count-1}',external_memory_bytes,1e9,0.001))
    hardware = Hardware(tuple(nodes),tuple(links),memory_ports=memory_ports)
    hardware.validate()
    return hardware
