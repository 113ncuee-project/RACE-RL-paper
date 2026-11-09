"""Generic bidirectional physical topology with explicit SI service parameters."""
from dataclasses import dataclass
from collections import deque
import math
from functools import cached_property
from types import MappingProxyType


@dataclass(frozen=True)
class Node:
    id: str
    kind: str
    x_mm: float
    y_mm: float
    width_mm: float
    height_mm: float
    memory_bytes: int
    bandwidth_Bps: float
    operations_per_s: float = 0.0
    slots: int = 1
    static_power_w: float = 0.0


@dataclass(frozen=True)
class Link:
    id: str
    a: str
    b: str
    bandwidth_Bps: float
    wire_cycles_per_mm: float = 1.0


@dataclass(frozen=True)
class MemoryPort:
    """External-memory service, not a die, router or physical mesh vertex."""
    id: str
    attached_to: str
    memory_bytes: int
    bandwidth_Bps: float
    static_power_w: float = 0.0


@dataclass(frozen=True)
class Hardware:
    nodes: tuple[Node, ...]
    links: tuple[Link, ...]
    network_clock_hz: float = 1e9
    bump_pitch_mm: float = 0.01
    phy_latency_cycles: float = 1.0
    router_latency_cycles: float = 1.0
    memory_ports: tuple[MemoryPort,...] = ()

    def __post_init__(self):
        object.__setattr__(self,"nodes",tuple(self.nodes))
        object.__setattr__(self,"links",tuple(self.links))
        object.__setattr__(self,"memory_ports",tuple(self.memory_ports))

    @cached_property
    def service_nodes(self):
        physical = {n.id:n for n in self.nodes}
        logical = []
        for p in self.memory_ports:
            if p.attached_to not in physical:
                raise ValueError('external memory attachment is not a physical node')
            anchor = physical[p.attached_to]
            # Zero-size compatibility view; never supplied to native geometry.
            logical.append(Node(p.id,'memory',anchor.x_mm,anchor.y_mm,0,0,
                p.memory_bytes,p.bandwidth_Bps,static_power_w=p.static_power_w))
        return self.nodes+tuple(logical)

    @cached_property
    def node_map(self):
        return MappingProxyType({n.id:n for n in self.service_nodes})

    @property
    def compute_ids(self):
        return tuple(n.id for n in self.nodes if n.kind == "compute")

    @property
    def memory_ids(self):
        return tuple(n.id for n in self.service_nodes if n.kind == "memory")

    def validate(self):
        if not self.compute_ids or not self.memory_ids or len(self.node_map) != len(self.service_nodes):
            raise ValueError("missing compute/memory nodes or duplicate IDs")
        def positive(x): return math.isfinite(x) and x > 0
        for p in self.memory_ports:
            if not p.id or p.attached_to not in self.compute_ids:
                raise ValueError('invalid external memory attachment')
            if type(p.memory_bytes) is not int or p.memory_bytes < 1 or not positive(p.bandwidth_Bps):
                raise ValueError('invalid external memory capacity/bandwidth')
            if not math.isfinite(p.static_power_w) or p.static_power_w < 0:
                raise ValueError('invalid external memory static power')
        if not all(positive(x) for x in (self.network_clock_hz,self.bump_pitch_mm,
                                       self.phy_latency_cycles,self.router_latency_cycles)):
            raise ValueError("invalid network/PHY parameters")
        for n in self.nodes:
            if n.kind not in {"compute","memory","io"} or not n.id:
                raise ValueError("invalid node role")
            if not all(positive(x) for x in (n.width_mm,n.height_mm,n.bandwidth_Bps)):
                raise ValueError("invalid geometry/bandwidth")
            if not math.isfinite(n.x_mm) or not math.isfinite(n.y_mm) or not math.isfinite(n.static_power_w) or n.static_power_w < 0:
                raise ValueError("invalid position/power")
            if type(n.memory_bytes) is not int or n.memory_bytes < 0 or type(n.slots) is not int or n.slots < 1:
                raise ValueError("invalid capacity/slots")
            if n.kind == "compute" and not positive(n.operations_per_s):
                raise ValueError("compute throughput must be positive")
        for i,n in enumerate(self.nodes):
            for m in self.nodes[i+1:]:
                if n.x_mm < m.x_mm+m.width_mm and m.x_mm < n.x_mm+n.width_mm and n.y_mm < m.y_mm+m.height_mm and m.y_mm < n.y_mm+n.height_mm:
                    raise ValueError("overlapping chiplets")
        seen = set()
        if len({l.id for l in self.links}) != len(self.links):
            raise ValueError("duplicate link ID")
        for l in self.links:
            key = frozenset((l.a,l.b))
            if l.a == l.b or not {l.a,l.b} <= {n.id for n in self.nodes} or key in seen:
                raise ValueError("invalid/parallel link")
            if not positive(l.bandwidth_Bps) or not math.isfinite(l.wire_cycles_per_mm) or l.wire_cycles_per_mm < 0:
                raise ValueError("invalid link service")
            seen.add(key)
        return True

    def route(self, source, destination):
        if source not in self.node_map or destination not in self.node_map:
            raise ValueError("unknown route endpoint")
        attachments = {p.id:p.attached_to for p in self.memory_ports}
        source,destination = attachments.get(source,source),attachments.get(destination,destination)
        queue = deque([(source, ())])
        visited = {source}
        while queue:
            node,path = queue.popleft()
            if node == destination:
                return path
            for l in sorted(self.links, key=lambda l:l.id):
                other = l.b if l.a == node else l.a if l.b == node else None
                if other is not None and other not in visited:
                    visited.add(other)
                    queue.append((other,(*path,(l.id,node,other))))
        raise ValueError(f"disconnected endpoints: {source} -> {destination}")
