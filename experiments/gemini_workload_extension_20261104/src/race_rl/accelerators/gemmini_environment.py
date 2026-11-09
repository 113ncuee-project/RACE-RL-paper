"""Gemmini GEMM design environment for the existing single REINFORCE trainer.

No reset or candidate enumeration invokes the hardware model. A complete
transaction jointly chooses count/blocking/residency/buffering/placement.
Uncalibrated feedback is inspectable, but never silently becomes PPA reward.
"""
from dataclasses import dataclass, asdict, replace
import math
from ..observation import GraphObservation
from ..workload import OPERATOR_KINDS
from ..requirements import PPARequirements
from ..rewards import reward
from ..serialization import digest, code_hashes
from ..mesh import mesh_shape
from ..oracle import BudgetExhausted
from .gemmini_config import default_config
from .memory_hierarchy import MemoryHierarchy
from .gemmini_model import GemminiModel, GemminiAction, GemmShape, PlatformScenario

BLOCK_PROFILES = ((16, 16, 16), (32, 16, 32), (16, 32, 32), (32, 32, 64), (64, 64, 64))


@dataclass(frozen=True)
class GemminiDesign:
    compute_count: int
    m_tile: int
    n_tile: int
    k_tile: int
    retain_weights: bool
    double_buffer: bool
    target: int
    spread: bool

    def __post_init__(self):
        if type(self.compute_count) is not int or not 1 <= self.compute_count <= 16:
            raise ValueError('Gemmini design count must be in 1–16')
        if type(self.target) is not int or not 0 <= self.target < self.compute_count:
            raise ValueError('Gemmini target must exist in proposed hardware')
        if type(self.spread) is not bool: raise ValueError('spread must be boolean')
        GemminiAction(self.m_tile,self.n_tile,self.k_tile,self.retain_weights,self.double_buffer)

    def materialize(self, shape):
        groups = ((shape.m+self.m_tile-1)//self.m_tile)*((shape.n+self.n_tile-1)//self.n_tile)
        placement = tuple((self.target+index)%self.compute_count if self.spread else self.target for index in range(groups))
        return GemminiAction(self.m_tile, self.n_tile, self.k_tile, self.retain_weights,
                             self.double_buffer, placement)


@dataclass(frozen=True)
class GemminiObservation:
    graph: GraphObservation
    legal_actions: tuple
    legal_action_masks: tuple
    action_features: tuple
    compute_count: int
    current_design: GemminiDesign
    model_feedback_valid: bool
    ppa_valid: bool
    last_reward_valid: bool
    backend: str = "race_gemmini_hierarchy_ws_v2"


@dataclass(frozen=True)
class GemminiEvaluation:
    design_hash: str
    physical_feasible: bool
    scorable: bool
    evaluation_status: str
    violations: tuple
    model_feedback: dict | None
    ppa_detail: dict | None
    energy_j: float | None = None
    latency_s: float | None = None
    average_power_w: float | None = None
    area_mm2: float | None = None

    @property
    def feasible(self): return self.physical_feasible and self.scorable


class GemminiDesignOracle:
    def __init__(self, shape, platform, cost_profile, budget, memory):
        self.memory = memory
        self.shape, self.platform, self.cost_profile, self.budget = shape, platform, cost_profile, budget
        self.requests = self.evaluations = self.cache_hits = 0
        self.cache, self.request_log = {}, []
        self.identity = digest({'backend':'race_gemmini_hierarchy_ws_v2', 'shape':shape, 'platform':platform,
            'memory_hierarchy':memory.record(), 'cost_profile':cost_profile, 'public_configuration':default_config().provenance(), 'source':code_hashes()})

    def evaluate(self, design):
        action = design.materialize(self.shape)
        key = digest({'oracle':self.identity, 'count':design.compute_count, 'execution':action})
        cached = key in self.cache
        if not cached and self.evaluations >= self.budget: raise BudgetExhausted('Gemmini evaluation budget exhausted')
        self.requests += 1
        if cached:
            self.cache_hits += 1
            result = self.cache[key]
        else:
            self.evaluations += 1
            try:
                feedback = GemminiModel(chiplets=design.compute_count, platform=self.platform,memory=self.memory).evaluate(self.shape, action)
            except ValueError as error:
                result = GemminiEvaluation(key, False, self.cost_profile is not None, 'PHYSICAL_INVALID', (str(error),), None, None)
            else:
                ppa = self.cost_profile.assess(feedback) if self.cost_profile is not None else None
                result = GemminiEvaluation(key, True, ppa is not None,
                    'EVALUATED' if ppa is not None else 'PPA_UNAVAILABLE', (), feedback, ppa,
                    **({name:ppa[name] for name in ('energy_j','latency_s','average_power_w','area_mm2')} if ppa else {}))
            self.cache[key] = result
        self.request_log.append({'design_hash':key, 'cached':cached, 'requests':self.requests,
            'evaluations':self.evaluations, 'physical_feasible':result.physical_feasible,
            'scorable':result.scorable, 'status':result.evaluation_status})
        return result


class GemminiDesignEnvironment:
    def __init__(self, shape, *, compute_counts=tuple(range(1,17)), initial_count=2,
                 budget=64, max_requests=None, requirements=None, platform=None,
                 cost_profile=None, allow_test_fixture=False, block_profiles=BLOCK_PROFILES, memory=None):
        if not isinstance(shape, GemmShape): raise ValueError('complete GEMM shape required; CNN graphs are not implicitly lowered')
        if not compute_counts or len(set(compute_counts)) != len(compute_counts) or any(
                type(n) is not int or not 1 <= n <= 16 for n in compute_counts):
            raise ValueError('distinct compute counts in 1–16 required')
        if initial_count not in compute_counts: raise ValueError('initial count must be in the declared set')
        if type(budget) is not int or budget < 1: raise ValueError('positive budget required')
        requests = budget if max_requests is None else max_requests
        if type(requests) is not int or requests < 1: raise ValueError('positive request ceiling required')
        self.shape, self.counts, self.initial_count = shape, tuple(compute_counts), initial_count
        self.platform = platform or PlatformScenario()
        self.memory = memory or MemoryHierarchy()
        self.cost_profile = cost_profile.validate(allow_test_fixture=allow_test_fixture) if cost_profile else None
        if cost_profile and cost_profile.memory_hierarchy_identity != self.memory.identity:
            raise ValueError('cost profile memory hierarchy does not match environment')
        if cost_profile and self.platform.clock_hz not in (None, cost_profile.clock_hz):
            raise ValueError('platform clock conflicts with PPA profile clock')
        self.requirements = requirements or PPARequirements()
        self.block_profiles = tuple(tuple(p) for p in block_profiles)
        if not self.block_profiles: raise ValueError('at least one structural block profile required')
        for profile in self.block_profiles:
            if len(profile)!=3: raise ValueError('M/N/K blocking triple required')
            GemminiAction(*profile)
        self.settings = {'backend':'race_gemmini_hierarchy_ws_v2', 'budget':budget, 'max_requests':requests,
            'counts':self.counts, 'block_profiles':self.block_profiles, 'requirements':self.requirements,
            'platform':self.platform, 'memory_hierarchy':self.memory.record(), 'cost_profile':self.cost_profile,
            'reward':'verified-best dense PPA; no cycle/traffic surrogate'}
        # Structural catalogue only. It does not call compile(), evaluate(), or a cost model.
        self.catalogue = tuple(GemminiDesign(n,*profile,retain,double,target,spread)
            for n in self.counts for profile in self.block_profiles for retain in (False,True)
            for double in (False,True) for target in range(n) for spread in ((False,True) if n>1 else (False,)))

    @property
    def experiment_identity(self):
        return digest({'shape':self.shape, 'settings':self.settings,
                       'public_configuration':default_config().provenance(), 'source':code_hashes()})

    def reset(self):
        self.oracle = GemminiDesignOracle(self.shape,self.platform,self.cost_profile,self.settings['budget'],self.memory)
        self.current_design = GemminiDesign(self.initial_count,*self.block_profiles[0],True,True,0,True)
        self.compute_count = self.initial_count
        self.current_record = self.best_record = self.current_feedback = self.best_design = None
        self.current_objective = self.best_objective = 1.0
        self.last_reward, self.last_reward_valid = 0.0, False
        self.scorer = self.requirements.bind()
        self.trace = []
        self.termination = None
        self.done = False
        self._refresh()
        return self.observation

    def feasible(self, record): return bool(record and record.scorable and self.scorer.assess(record)['feasible'])

    def _features(self, design):
        rows, columns = mesh_shape(design.compute_count)
        return (design.compute_count/16, math.log1p(design.m_tile)/math.log1p(max(self.shape.m,16)),
            math.log1p(design.n_tile)/math.log1p(max(self.shape.n,16)),
            math.log1p(design.k_tile)/math.log1p(max(self.shape.k,16)),
            float(design.retain_weights),float(design.double_buffer),design.target/max(design.compute_count-1,1),
            float(design.spread),rows/4,columns/16,
            math.log1p(((self.shape.m+design.m_tile-1)//design.m_tile)*((self.shape.n+design.n_tile-1)//design.n_tile)),1.0)

    def _refresh(self):
        self.actions = tuple(action for action in self.catalogue if action != self.current_design)
        if not self.termination:
            self.termination = ('budget_exhausted' if self.oracle.evaluations>=self.oracle.budget else
                'request_limit' if self.oracle.requests>=self.settings['max_requests'] else
                'no_legal_action' if not self.actions else None)
        self.done = self.termination is not None
        graph = self._graph()
        self.observation = GemminiObservation(graph,self.actions,tuple(True for _ in self.actions),
            tuple(self._features(action) for action in self.actions),self.compute_count,self.current_design,
            self.current_feedback is not None,self.current_record is not None,self.last_reward_valid)

    def _graph(self):
        c, d, shape, h = default_config(), self.current_design, self.shape, self.memory
        execution = d.materialize(shape)
        group_shapes = [(m,min(d.m_tile,shape.m-m),n,min(d.n_tile,shape.n-n))
                        for n in range(0,shape.n,d.n_tile) for m in range(0,shape.m,d.m_tile)]
        measured, scorable = self.current_feedback is not None, self.current_record is not None
        feedback = self.current_feedback or {}
        events = feedback.get('events',())
        total_cycles = feedback.get('modeled_cycles',1)
        tiles = ('A','B',*(f'g{i}' for i in range(len(group_shapes))),'C')
        tile_features = []
        def vector(kind,ops,inputs,outputs,elapsed,coordinates):
            return (*[float(kind==name) for name in OPERATOR_KINDS],math.log1p(ops),math.log1p(inputs),
                    math.log1p(outputs),elapsed,0.0,0.0,*coordinates)
        tile_features.append(vector('Identity',0,shape.m*shape.k,shape.m*shape.k,0,(0,1,0,1,0,1)))
        tile_features.append(vector('Identity',0,shape.k*shape.n,shape.k*shape.n,0,(0,1,0,1,0,1)))
        for index,(m,mm,n,nn) in enumerate(group_shapes):
            elapsed = sum(event['finish_cycle']-event['start_cycle'] for event in events
                if event['job']['kind']=='execute_ws' and event['job']['metadata']['output_block']==index)/total_cycles
            tile_features.append(vector('Linear',mm*nn*shape.k,mm*shape.k+shape.k*nn,mm*nn*4,elapsed,
                                       (m/shape.m,(m+mm)/shape.m,n/shape.n,(n+nn)/shape.n,0,1)))
        tile_features.append(vector('Identity',0,shape.m*shape.n*4,shape.m*shape.n*4,0,(0,1,0,1,0,1)))
        dependencies = tuple(edge for index in range(len(group_shapes)) for edge in
                             ((0,2+index),(1,2+index),(2+index,len(tiles)-1)))
        hw_ids, hw_features = [], []
        rows, columns = mesh_shape(self.compute_count)
        resource_names = []
        def add_node(name,compute,memory,io,throughput,capacity,bandwidth,x,y,resources):
            busy = sum(event['finish_cycle']-event['start_cycle'] for event in events
                if any(resource in event['job']['resources'] for resource in resources))/total_cycles
            hw_ids.append(name);resource_names.append(resources)
            hw_features.append((compute,memory,io,math.log1p(throughput),math.log1p(capacity),math.log1p(bandwidth),x,y,busy))
        for node in range(self.compute_count):
            x,y = (node%columns)/max(columns,1),(node//columns)/max(rows,1)
            add_node(f'c{node}.array',1,0,0,256,0,0,x,y,(f'c{node}.array',))
            for bank in range(4):
                add_node(f'c{node}.gb{bank}',0,1,0,0,c.scratchpad_bank_bytes,h.gb_bank_bytes_per_cycle,x,y,(f'c{node}.gb.bank{bank}',))
                add_node(f'c{node}.lb{bank}',0,1,0,0,h.lb_operand_slot_bytes,h.lb_bank_bytes_per_cycle,x,y,(f'c{node}.lb.bank{bank}',))
            add_node(f'c{node}.lbC',0,1,0,0,h.lb_output_bytes,h.lb_bank_bytes_per_cycle,x,y,(f'c{node}.lb.C',))
            for operand,capacity,bw in (('A',h.rf_a_bytes,h.rf_operand_bytes_per_cycle),('B',h.rf_b_bytes,h.rf_operand_bytes_per_cycle),('psum',h.rf_psum_bytes,h.rf_psum_bytes_per_cycle)):
                add_node(f'c{node}.rf{operand}',0,1,0,0,capacity,bw,x,y,
                         (f'c{node}.rf.{operand}.read',f'c{node}.rf.{operand}.write'))
            for bank in range(2):
                add_node(f'c{node}.acc{bank}',0,1,0,0,c.accumulator_bank_bytes,h.accumulator_bank_bytes_per_cycle,x,y,
                         (f'c{node}.acc.bank{bank}.read',f'c{node}.acc.bank{bank}.write'))
            add_node(f'c{node}.dma',0,0,1,0,0,c.dma_beat_bytes,x,y,(f'c{node}.dma.read',f'c{node}.dma.write'))
        add_node('dram0',0,1,0,0,self.platform.dram_capacity_bytes,self.platform.dram_bytes_per_cycle,0,0,('dram.channel0',))
        hi = {name:index for index,name in enumerate(hw_ids)}
        edges, edge_resources, edge_bandwidth = [], [], []
        def connect(a,b,bw,resource=None):
            for source,destination in ((a,b),(b,a)):
                edges.append((hi[source],hi[destination]));edge_bandwidth.append(bw)
                edge_resources.append((resource if source==a else None))
        for node in range(self.compute_count):
            for bank in range(4):
                connect(f'c{node}.dma',f'c{node}.gb{bank}',min(c.dma_beat_bytes,h.gb_bank_bytes_per_cycle))
                # Retained B spans GB banks 2/3 and feeds either B LB slot.
                for lb in ((bank,) if bank<2 else (2,3)):
                    connect(f'c{node}.gb{bank}',f'c{node}.lb{lb}',min(h.gb_bank_bytes_per_cycle,h.lb_bank_bytes_per_cycle))
                connect(f'c{node}.lb{bank}',f'c{node}.rf'+('A' if bank<2 else 'B'),min(h.lb_bank_bytes_per_cycle,h.rf_operand_bytes_per_cycle))
            for bank in range(2):
                connect(f'c{node}.acc{bank}',f'c{node}.lbC',min(h.accumulator_bank_bytes_per_cycle,h.lb_bank_bytes_per_cycle))
                connect(f'c{node}.lbC',f'c{node}.gb{bank}',min(h.lb_bank_bytes_per_cycle,h.gb_bank_bytes_per_cycle))
            connect(f'c{node}.lbC',f'c{node}.rfpsum',min(h.lb_bank_bytes_per_cycle,h.rf_psum_bytes_per_cycle))
            for operand in ('A','B','psum'):
                connect(f'c{node}.rf{operand}',f'c{node}.array',h.rf_psum_bytes_per_cycle if operand=='psum' else h.rf_operand_bytes_per_cycle)
        # Only actual mesh edges. Internal paths above are storage connectivity,
        # not extra chiplets or invented silicon geometry.
        d2d_index = {}
        for node in range(self.compute_count):
            for other in (node+1 if node%columns+1<columns else None,
                          node+columns if node//columns+1<rows else None):
                if other is None: continue
                for a,b in ((node,other),(other,node)):
                    d2d_index[a,b]=len(edges)
                    edges.append((hi[f'c{a}.dma'],hi[f'c{b}.dma']))
                    edge_bandwidth.append(self.platform.mesh_bytes_per_cycle)
                    edge_resources.append(f'd2d.c{a}>c{b}')
        flow_bytes = {index:0 for index in range(len(edges))}
        edge_activity = {index:set() for index in range(len(edges))}
        relations = []
        for flow in feedback.get('transfers',()):
            index = int(flow['job'].split('.',1)[0][1:])
            group_indices = [index]
            if flow['operand']=='B' and '.resident.' in flow['job']:
                group_indices = [j for j,g in enumerate(group_shapes) if g[2]==group_shapes[index][2]
                                 and execution.placement[j]==flow['chiplet']]
            for a,b in flow['route']:
                edge = d2d_index[a,b];flow_bytes[edge]+=flow['bytes']
                for group_index in group_indices:
                    dependency = (0 if flow['operand']=='A' else 1,2+group_index) if flow['direction']=='read' else (2+group_index,len(tiles)-1)
                    relations.append((dependencies.index(dependency),edge,flow['job'],'read' if flow['direction']=='read' else 'write'))
        edge_lookup = {(hw_ids[a],hw_ids[b]):index for index,(a,b) in enumerate(edges)}
        def endpoint_nodes(endpoint,node,event):
            level,*rest = endpoint.split('.',1)
            suffix = rest[0] if rest else ''
            if level=='gb':
                banks = event['job']['metadata'].get('gb_banks',(0,1) if suffix=='C' else ())
                return [f'c{node}.gb{bank}' for bank in banks]
            if level=='lb': return [f'c{node}.lbC' if suffix=='C' else f'c{node}.lb{suffix[1:]}']
            if level=='rf': return [f'c{node}.rf{suffix}']
            if level=='accumulator': return [f'c{node}.acc{bank}' for bank in range(2)]
            return []
        for event in events:
            metadata = event['job']['metadata']
            node = metadata.get('chiplet',next((flow['chiplet'] for flow in feedback.get('transfers',()) if flow['job']==event['job']['id']),0))
            for move in metadata.get('movements',()):
                sources,destinations = endpoint_nodes(move['source'],node,event),endpoint_nodes(move['destination'],node,event)
                if move['source']=='dram': sources=[f'c{node}.dma']
                if move['destination']=='dram': destinations=[f'c{node}.dma']
                candidates = [edge_lookup[a,b] for a in sources for b in destinations if (a,b) in edge_lookup]
                if candidates:
                    for edge in candidates:
                        flow_bytes[edge] += move['bytes']/len(candidates)
                        edge_activity[edge].add(event['job']['id'])
            for operand,amount in metadata.get('rf_compute_access_bytes',{}).items():
                name = operand.split('_')[0]
                if name in ('a','b'): name = name.upper()
                direction = (f'c{node}.array',f'c{node}.rf{name}') if 'write' in operand else (f'c{node}.rf{name}',f'c{node}.array')
                flow_bytes[edge_lookup[direction]] += amount
                if amount: edge_activity[edge_lookup[direction]].add(event['job']['id'])
        edge_features = []
        for index,resource in enumerate(edge_resources):
            busy = sum(event['finish_cycle']-event['start_cycle'] for event in events
                       if (resource is not None and resource in event['job']['resources']) or
                          (resource is None and event['job']['id'] in edge_activity[index]))/total_cycles
            edge_features.append((math.log1p(edge_bandwidth[index]),math.log1p(flow_bytes[index]),busy,0.0))
        placements = ((0,hi['dram0']),(1,hi['dram0']),
                      *((2+i,hi[f'c{node}.array']) for i,node in enumerate(execution.placement)),
                      (len(tiles)-1,hi['dram0']))
        current_ppa = self.current_record
        global_base = (math.log(self.current_objective),math.log(self.best_objective),
            (self.oracle.budget-self.oracle.evaluations)/self.oracle.budget,self.oracle.evaluations/self.oracle.budget,
            current_ppa.average_power_w if scorable else 0,current_ppa.latency_s if scorable else 0,
            self.compute_count/16,(self.settings['max_requests']-self.oracle.requests)/self.settings['max_requests'],
            float(scorable),self.last_reward)
        global_extra = (math.log1p(shape.m),math.log1p(shape.n),math.log1p(shape.k),
            math.log1p(d.m_tile),math.log1p(d.n_tile),math.log1p(d.k_tile),float(d.retain_weights),
            float(d.double_buffer),d.target/max(d.compute_count-1,1),float(d.spread),float(measured),float(self.last_reward_valid))
        return GraphObservation(7,tiles,tuple(hw_ids),tuple(tile_features),tuple(hw_features),dependencies,
            tuple((1.0,) for _ in dependencies),tuple(edges),tuple(edge_features),placements,
            tuple(sorted(set(relations))),(*global_base,*global_extra),(),(),
            {'tile':tuple((True,)*14+(measured,)*3+(True,)*6 for _ in tile_features),
             'hardware':tuple((True,)*8+(measured,) for _ in hw_features),
             'link':tuple((True,)+(measured,)*2+(False,) for _ in edges),
             'task':(True,)*9+(scorable,)*6+(True,),
             'global':(scorable,scorable,True,True,scorable,scorable,True,True,True,self.last_reward_valid)+(True,)*12},
            tuple(self.oracle.cache),self.best_record.design_hash if self.best_record else '',
            ((hi['dram0'],hi['c0.dma']),(hi['c0.dma'],hi['dram0'])),
            self.scorer.features(current_ppa,self.feasible(self.best_record)))

    def step(self, action):
        if self.done: raise RuntimeError('Gemmini episode terminated')
        if action not in self.actions: raise ValueError('action is not in the structural catalogue')
        old_best = self.best_objective
        result = self.oracle.evaluate(action)
        if result.physical_feasible:
            self.current_design, self.compute_count, self.current_feedback = action,action.compute_count,result.model_feedback
            if result.scorable:
                self.current_record = result
                self.current_objective = self.scorer.objective(result)
                if self.scorer.better(result,self.best_record):
                    self.best_record,self.best_design,self.best_objective = result,action,self.current_objective
            else:
                self.current_record = None
        if not result.scorable:
            r = None
            self.last_reward,self.last_reward_valid = 0.0,False
            self.termination = 'ppa_unavailable'
        else:
            r = reward(1.0,old_best,self.best_objective,variant='dense')
            self.last_reward,self.last_reward_valid = r,True
        self._refresh()
        self.trace.append({'action':asdict(action),'design_hash':result.design_hash,
            'physical_feasible':result.physical_feasible,'scorable':result.scorable,
            'status':result.evaluation_status,'ppa_assessment':self.scorer.assess(result) if result.scorable else None,
            'model_cycles':result.model_feedback['modeled_cycles'] if result.model_feedback else None,
            'model_traffic':result.model_feedback['traffic'] if result.model_feedback else None,
            'current_objective':self.current_objective,'best_objective':self.best_objective,
            'reward':r,'requests':self.oracle.requests,'evaluations':self.oracle.evaluations,
            'cache_hits':self.oracle.cache_hits,'termination':self.termination})
        return self.observation,r,self.done
