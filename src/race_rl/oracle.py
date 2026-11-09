"""Spill-all analytical E2E oracle with native network and traceable accounting.

All intermediate tensors are written to the selected memory node. Network
transfers atomically reserve directed path and endpoint resources. This is a
conservative service model, not a packet simulator or silicon measurement.
"""
from collections import defaultdict
from dataclasses import dataclass, asdict
import math
import time
from .network import evaluate_network
from .parameters import exploratory_profile
from .schedule import Job, schedule, criticality, peak_bytes
from .serialization import digest, code_hashes, plain
from .tiles import requirements, operations, parameter_regions, macs


class BudgetExhausted(RuntimeError):
    pass


class ActionOracle:
    """Defer all native setup and PPA evaluation until an environment action."""
    def __init__(self,workload,hardware,*,budget,scheduling_policy):
        if type(budget) is not int or budget < 1: raise ValueError('positive integer oracle budget required')
        self.workload,self.hardware,self.budget = workload,hardware,budget
        self.scheduling_policy = scheduling_policy
        self.backend = None
        self.identity = digest({'workload':workload,'hardware':hardware,'budget':budget,
            'source':code_hashes(),'scheduling_policy':scheduling_policy,'contract':'action-triggered-v1'})

    def __getattr__(self,name):
        if self.backend is not None: return getattr(self.backend,name)
        if name in ('evaluations','requests','cache_hits','invalid_proposals','physical_failures','setup_time_s'): return 0
        if name == 'cache': return {}
        if name == 'request_log': return []
        raise AttributeError(name)

    def evaluate(self,mapping):
        if self.backend is None:
            self.backend = Oracle(self.workload,self.hardware,budget=self.budget,scheduling_policy=self.scheduling_policy)
        return self.backend.evaluate(mapping)


@dataclass(frozen=True)
class Evaluation:
    workload_hash: str
    hardware_hash: str
    mapping_hash: str
    evaluator_version: str
    parameter_profile: dict
    evaluation_status: str
    feasible: bool
    violations: tuple[str, ...]
    energy_j: float | None
    latency_s: float | None
    area_mm2: float
    average_power_w: float | None
    schedule_trace: tuple
    traffic_ledger: tuple
    energy_ledger: tuple
    buffer_ledger: dict
    diagnostics: dict
    source_hashes: dict
    wall_time_s: float


class Oracle:
    def __init__(self, workload, hardware, *, budget=64, profile=None, native_root=None,
                 scheduling_policy="shared_reads"):
        setup_started = time.perf_counter()
        if type(budget) is not int or budget < 1:
            raise ValueError("positive integer oracle budget required")
        workload.validate()
        hardware.validate()
        self.workload,self.hardware,self.budget = workload,hardware,budget
        if scheduling_policy not in {"shared_reads","serial_node_tiles"}:
            raise ValueError("unknown scheduling policy")
        self.scheduling_policy = scheduling_policy
        self.profile = profile or exploratory_profile()
        self.profile.validate()
        self.network = evaluate_network(hardware,native_root)
        self.hashes = code_hashes()
        self.identity = digest({"workload":workload,"hardware":hardware,"profile":self.profile,
                                "native":self.network.source_hashes,"code":self.hashes,"scheduling_policy":scheduling_policy})
        self.requests = self.evaluations = self.cache_hits = self.invalid_proposals = 0
        self.cache,self.request_log = {},[]
        self.physical_failures = 0
        self.setup_time_s = time.perf_counter()-setup_started

    def evaluate(self, mapping):
        key = digest(mapping)
        if key not in self.cache and self.evaluations >= self.budget:
            raise BudgetExhausted("official analytical evaluation budget exhausted")
        self.requests += 1
        if key in self.cache:
            self.cache_hits += 1
            self.request_log.append({"mapping_hash":key,"cached":True,"evaluations":self.evaluations})
            return self.cache[key]
        self.evaluations += 1  # includes every failed uncached validation/evaluation
        started = time.perf_counter()
        try:
            mapping.validate(self.workload,self.hardware)
            result = self._run(mapping,started)
        except ValueError as exc:
            self.invalid_proposals += 1
            result = self._record(mapping,started,"INVALID",False,(str(exc),))
        if result.evaluation_status == "EVALUATED" and not result.feasible:
            self.physical_failures += 1
        self.cache[key] = result
        self.request_log.append({"mapping_hash":key,"cached":False,"evaluations":self.evaluations,
                                 "feasible":result.feasible,"status":result.evaluation_status})
        return result

    def _record(self,mapping,started,status,feasible,violations,energy=None,latency=None,
                events=(),traffic=(),ledger=(),buffers=None,diagnostics=None):
        return Evaluation(digest(self.workload),digest(self.hardware),digest(mapping),self.identity,
            plain(self.profile),status,feasible,tuple(violations),energy,latency,self.network.area_mm2,
            energy/latency if energy is not None and latency else None,
            tuple(plain(e) for e in events),tuple(plain(t) for t in traffic),tuple(ledger),
            buffers or {},diagnostics or {},{**self.hashes,"native":self.network.source_hashes},
            time.perf_counter()-started)

    def _run(self,mapping,started):
        w,h,p,net = self.workload,self.hardware,self.profile,self.network
        place,store = mapping.placements,mapping.storages
        jobs,traffic = {},{}
        owners = defaultdict(list)
        for t in mapping.tiles:
            owners[w.operator_map[t.operator].output].append(t)
        reads_by_tile = defaultdict(list)
        writes = {t.id:f"write:{t.id}" for t in mapping.tiles}
        order = {o.id:i for i,o in enumerate(w.operators)}
        ordered_tiles = sorted(mapping.tiles,key=lambda t:(order[t.operator],t.output.bounds,t.id))
        tile_predecessors = {}
        if self.scheduling_policy == "serial_node_tiles":
            previous = {}
            for t in ordered_tiles:
                node = place[t.id]
                if node in previous:
                    tile_predecessors[t.id] = writes[previous[node]]
                previous[node] = t.id

        def transfer(name,tensor,region,amount,src,dst,deps,consumer=None,producer=None,data_dependencies=None):
            route = h.route(src,dst)
            resources = [f"memory:{src if h.node_map[src].kind == 'memory' else dst}",
                         f"dma:{dst if h.node_map[dst].kind == 'compute' else src}"]
            compute_endpoint = dst if h.node_map[dst].kind == "compute" else src
            resources.append(f"local_memory:{compute_endpoint}")
            resources += [f"link:{lid}:{a}>{b}" for lid,a,b in route]
            memory = src if h.node_map[src].kind == "memory" else dst
            mem_s = amount/h.node_map[memory].bandwidth_Bps
            local_s = amount/h.node_map[compute_endpoint].bandwidth_Bps
            network_s = net.latency_s[(src,dst)] + (amount/min(net.bandwidth_Bps[(a,b)] for _,a,b in route) if route else 0)
            energy = (("external_memory",amount*p.external_byte_j.value),
                      ("local_memory",amount*p.local_byte_j.value),
                      ("interconnect",8*amount*len(route)*p.link_bit_j.value))
            entry = {"id":name,"tensor":tensor,"region":plain(region),"bytes":amount,
                     "source":src,"destination":dst,"route":route,"dependencies":tuple(sorted(set(deps))),
                     "data_dependencies":tuple(sorted(set(deps if data_dependencies is None else data_dependencies))),
                     "consumer_tiles":[] if consumer is None else [consumer],"producer_tile":producer,
                     "memory_service_s":mem_s,"network_service_s":network_s,
                     "local_service_s":local_s,
                     "sharing":"none; per-tile admission" if self.scheduling_policy == "serial_node_tiles" else "exact same region and destination only"}
            if name in traffic:
                if consumer and consumer not in traffic[name]["consumer_tiles"]:
                    traffic[name]["consumer_tiles"].append(consumer)
                return
            traffic[name] = entry
            jobs[name] = Job(name,entry["dependencies"],tuple(resources),mem_s+network_s+local_s,
                             "memory_read" if consumer else "memory_write",energy,entry)

        for t in mapping.tiles:
            op = w.operator_map[t.operator]
            output_bytes = t.output.elements*w.tensor_map[op.output].bytes_per_element
            transfer(writes[t.id],op.output,t.output,output_bytes,place[t.id],store[op.output],
                     (f"compute:{t.id}",),producer=t.id)

        for t in ordered_tiles:
            op = w.operator_map[t.operator]
            reqs = list(requirements(w,t))
            for tensor,region in reqs:
                source_tiles = [s for s in owners[tensor] if s.output.intersection(region) is not None]
                data_deps = tuple(writes[s.id] for s in source_tiles)
                deps = data_deps
                if t.id in tile_predecessors:
                    deps += (tile_predecessors[t.id],)
                amount = region.elements*w.tensor_map[tensor].bytes_per_element
                owner = t.id if self.scheduling_policy == "serial_node_tiles" else None
                name = "read:"+digest((tensor,region,store[tensor],place[t.id],deps,owner))[:24]
                transfer(name,tensor,region,amount,store[tensor],place[t.id],deps,consumer=t.id,data_dependencies=data_deps)
                reads_by_tile[t.id].append(name)
            constant_bytes = 0
            activation_ready = tuple(reads_by_tile[t.id])
            for tensor,region in parameter_regions(w,t):
                amount = region.elements*w.tensor_map[op.output].bytes_per_element
                constant_bytes += amount
                owner = t.id if self.scheduling_policy == "serial_node_tiles" else None
                name = "read:"+digest((tensor,region,store[tensor],place[t.id],owner))[:24]
                # Constants are loaded after the first consumer's activation reads.
                # Later identical same-destination consumers share that live copy.
                transfer(name,tensor,region,amount,store[tensor],place[t.id],activation_ready,consumer=t.id)
                reads_by_tile[t.id].append(name)
            node = h.node_map[place[t.id]]
            # Static round-robin slot assignment has no oracle lookahead.
            local_index = sum(1 for other in mapping.tiles[:mapping.tiles.index(t)] if place[other.id] == place[t.id])
            input_bytes = sum(r.elements*w.tensor_map[name].bytes_per_element for name,r in reqs)+constant_bytes
            output_bytes = t.output.elements*w.tensor_map[op.output].bytes_per_element
            count = operations(w,t)
            local_bytes = input_bytes+output_bytes
            duration = count/node.operations_per_s+local_bytes/node.bandwidth_Bps
            jobs[f"compute:{t.id}"] = Job(f"compute:{t.id}",tuple(reads_by_tile[t.id]),
                (f"compute:{node.id}:{local_index%node.slots}",f"local_memory:{node.id}"),duration,"compute",
                (("compute",count*p.operation_j.value),("local_memory",local_bytes*p.local_byte_j.value)),
                {"tile":t.id,"operations":count,"macs":macs(w,t),"parameter_bytes":constant_bytes,
                 "input_bytes":input_bytes,"output_bytes":output_bytes})
        result = schedule(jobs.values())
        events = result.event_map
        intervals = defaultdict(list)
        for entry in traffic.values():
            e = events[entry["id"]]
            src,dst,amount = entry["source"],entry["destination"],entry["bytes"]
            if entry["consumer_tiles"]:
                release = max(events[f"compute:{t}"].finish_s for t in entry["consumer_tiles"])
                intervals[dst].append((e.start_s,release,amount))
            else:
                t = entry["producer_tile"]
                compute = events[f"compute:{t}"]
                intervals[src].append((compute.start_s,e.finish_s,amount))
                consumers = [events[r["id"]].finish_s for r in traffic.values() if entry["id"] in r["data_dependencies"]]
                release = max(consumers,default=e.finish_s)
                if entry["tensor"] in w.outputs:
                    release = result.makespan_s
                # Reserve destination bytes during write, then until the final read.
                intervals[dst].append((e.start_s,release,amount))
        for tensor,endpoint in store.items():
            if tensor in w.producers:
                continue
            if tensor in w.parameter_map:
                op = w.operator_map[tensor.split(":",1)[1]]
                amount = math.prod(w.parameter_map[tensor])*w.tensor_map[op.output].bytes_per_element
            else:
                amount = w.tensor_map[tensor].bytes
            users = [events[r["id"]].finish_s for r in traffic.values() if r["tensor"] == tensor and r["consumer_tiles"]]
            intervals[endpoint].append((0,max(users,default=result.makespan_s),amount))
        peaks = {n.id:peak_bytes(intervals[n.id]) for n in h.service_nodes}
        violations = [f"buffer capacity {n.id}: {peaks[n.id]} B > {n.memory_bytes} B" for n in h.service_nodes if peaks[n.id] > n.memory_bytes]
        ledger = [{"job_id":j.id,"component":c,"energy_j":e} for j in jobs.values() for c,e in j.energy]
        ledger += [{"job_id":"inference","component":"static","energy_j":
            (net.idle_power_w+sum(port.static_power_w for port in h.memory_ports))*result.makespan_s}]
        components = {c:sum(e["energy_j"] for e in ledger if e["component"] == c)
                      for c in ("compute","local_memory","external_memory","interconnect","static")}
        energy = sum(components.values())
        if not math.isclose(energy,sum(e["energy_j"] for e in ledger),rel_tol=1e-12):
            raise ValueError("energy ledger mismatch")
        return self._record(mapping,started,"EVALUATED",not violations,violations,energy,result.makespan_s,
            result.events,traffic.values(),ledger,{"peak_bytes":peaks,"intervals":dict(intervals)},
            {"criticality":criticality(result),"energy_components":components,
             "scheduling_policy":self.scheduling_policy,"tile_control_predecessors":tile_predecessors,
             "external_memory_ports":plain(h.memory_ports),
             "memory_interface_assumption":"effective external service bandwidth/energy includes off-mesh access; gateway mesh hops are separately charged; no physical HBM/IO geometry claim",
             "timing_model":"spill-all; serialized memory plus atomic route service; serialized compute plus local service",
             "calibration_status":"UNCALIBRATED","native_backend":"official_rapidchiplet"})
