"""One RL agent edits architecture, whole-operator partitions and mapping."""
from dataclasses import dataclass
import math
import time
from .actions import apply,legal_actions
from .mesh import mesh_hardware,mesh_shape
from .mapping import Mapping
from .initialization import conservative_initial_mapping
from .oracle import Oracle,BudgetExhausted
from .observation import observe
from .rewards import potential,reward
from .serialization import digest,save_json,code_hashes
from .requirements import PPARequirements
from .stopping import PlateauSettings,ObjectivePlateau


@dataclass(frozen=True)
class CountAction:
    compute_count: int
    kind: str = 'resize_mesh'


@dataclass(frozen=True)
class RetileAction:
    operator: str
    parts: int
    target: str
    distribute: bool
    kind: str = 'retile_operator'


@dataclass(frozen=True)
class IndependentAction:
    kind: str
    target: str = ''
    tile: str = ''
    storage: str = ''
    operator: str = ''
    parts: int = 0


DEFAULT_COUNTS = tuple(range(2,17,2))


def part_counts(length,current):
    return sorted({n for n in (1,2,4,8,16,32,current-1,current,current+1,
        max(1,current//2),min(length,2*current)) if 1 <= n <= length})


@dataclass(frozen=True)
class DesignObservation:
    graph: object
    legal_actions: tuple
    legal_action_masks: tuple
    count_features: tuple
    retile_features: tuple
    compute_count: int
    best_hardware_hash: str


def resized_mapping(mapping,hardware):
    # Expansion keeps placements. Shrinking remaps only removed IDs modulo N.
    return Mapping(mapping.tiles,tuple((tile,node if node in hardware.compute_ids else
        hardware.compute_ids[int(node[1:])%len(hardware.compute_ids)]) for tile,node in mapping.placement),mapping.storage)


def operator_tiles(workload,operator,count):
    # Only materialize the selected operator, retaining every other tile.
    from .tiles import Tile,Region,whole
    shape = workload.tensor_map[workload.operator_map[operator].output].shape
    axis = 2 if len(shape) == 4 else 1
    if type(count) is not int or not 1 <= count <= shape[axis]:
        raise ValueError('invalid output partition count')
    group = []
    for i in range(count):
        bounds = list(whole(shape).bounds)
        bounds[axis] = (i*shape[axis]//count,(i+1)*shape[axis]//count)
        group.append(Tile(f'{operator}:{i}',operator,Region(tuple(bounds))))
    return tuple(group)


def retiled_mapping(workload,hardware,mapping,action):
    group = operator_tiles(workload,action.operator,action.parts)
    tiles = tuple(t for t in mapping.tiles if t.operator != action.operator)+group
    old_ids = {t.id for t in mapping.tiles if t.operator == action.operator}
    offset = hardware.compute_ids.index(action.target)
    placement = tuple((tid,node) for tid,node in mapping.placement if tid not in old_ids)
    placement += tuple((t.id,hardware.compute_ids[(offset+i)%len(hardware.compute_ids)]
        if action.distribute else action.target) for i,t in enumerate(group))
    return Mapping(tiles,placement,mapping.storage)


def independent_mapping(workload,hardware,mapping,action):
    placement,storage = mapping.placements.copy(),mapping.storages.copy()
    tiles = mapping.tiles
    if action.kind == 'place_tile': placement[action.tile] = action.target
    elif action.kind == 'store_object': storage[action.storage] = action.target
    elif action.kind == 'partition_operator':
        old = [t for t in tiles if t.operator == action.operator]
        group = operator_tiles(workload,action.operator,action.parts)
        for t in old: placement.pop(t.id)
        for t in group:
            # Partition-only: inherit the node of greatest old output overlap.
            owner = max(old,key=lambda previous:(
                (previous.output.intersection(t.output).elements if previous.output.intersection(t.output) else 0),
                previous.id))
            placement[t.id] = mapping.placements[owner.id]
        tiles = tuple(t for t in tiles if t.operator != action.operator)+group
    else: raise ValueError('unknown independent edit')
    return Mapping(tiles,tuple(sorted(placement.items())),tuple(sorted(storage.items())))


class DesignOracle:
    """All hardware contexts share one evaluation budget; cache key is H+M."""
    def __init__(self,workload,budget):
        self.workload,self.budget = workload,budget
        self.evaluations = self.requests = self.cache_hits = 0
        self.cache,self.request_log,self.backends,self.designs = {},[],{},{}
        self.identity = digest({'workload':workload,'source':code_hashes(),'contract':'joint-mesh-v1'})

    @property
    def setup_time_s(self): return sum(o.setup_time_s for o in self.backends.values())
    @property
    def invalid_proposals(self): return sum(o.invalid_proposals for o in self.backends.values())
    @property
    def physical_failures(self): return sum(o.physical_failures for o in self.backends.values())

    def evaluate(self,hardware,mapping):
        hh = digest(hardware)
        key = digest({'hardware':hh,'mapping':mapping})
        if key not in self.cache and self.evaluations >= self.budget:
            raise BudgetExhausted('global design budget exhausted')
        self.requests += 1
        cached = key in self.cache
        if cached:
            self.cache_hits += 1
            record = self.cache[key]
        else:
            if hh not in self.backends:
                self.backends[hh] = Oracle(self.workload,hardware,budget=self.budget,scheduling_policy='serial_node_tiles')
            record = self.backends[hh].evaluate(mapping)
            self.evaluations += 1
            self.cache[key] = record
            self.designs[key] = (hardware,mapping)
        self.request_log.append({'design_hash':key,'hardware_hash':hh,'mapping_hash':record.mapping_hash,
            'compute_count':len(hardware.compute_ids),'cached':cached,'evaluations':self.evaluations,'feasible':record.feasible})
        return record


class DesignEnvironment:
    """Every step may resize/retile/remap; poor PPA never removes a count."""
    def __init__(self,workload,*,compute_counts=DEFAULT_COUNTS,initial_count=None,
                 budget=16,max_scope=2,max_area_mm2=None,max_requests=None,requirements=None,
                 action_mode='coupled',reward_variant='dense',lambda_b=1.0,lambda_c=0.0):
        if (not compute_counts or len(set(compute_counts)) != len(compute_counts)
            or any(type(n) is not int or n < 1 for n in compute_counts)):
            raise ValueError('distinct positive compute counts required')
        if type(budget) is not int or budget < 1 or not 1 <= max_scope <= 4:
            raise ValueError('budget>=1 and scope in 1..4 required')
        if max_area_mm2 is not None and (not math.isfinite(max_area_mm2) or max_area_mm2 <= 0):
            raise ValueError('positive finite area limit required')
        if action_mode not in ('coupled','independent') or reward_variant not in ('dense','terminal','potential'):
            raise ValueError('unknown action/reward ablation')
        if any(not math.isfinite(v) or v < 0 for v in (lambda_b,lambda_c)):
            raise ValueError('finite nonnegative potential coefficients required')
        if requirements is not None and max_area_mm2 is not None:
            raise ValueError("supply area in requirements or max_area_mm2, not both")
        self.requirements = requirements or PPARequirements(max_area_mm2=max_area_mm2)
        self.workload,self.counts = workload,tuple(compute_counts)
        self.initial_count = self.counts[0] if initial_count is None else initial_count
        if self.initial_count not in self.counts: raise ValueError('initial count not in candidate set')
        self.hardware_candidates = {n:mesh_hardware(n) for n in self.counts}
        self.settings = {'budget':budget,'max_scope':max_scope,'max_requests':max_requests or budget,
            'requirements':self.requirements,'scheduling_policy':'serial_node_tiles','reward_variant':reward_variant,
            'action_mode':action_mode,'lambda_b':lambda_b,'lambda_c':lambda_c}
        if type(self.settings['max_requests']) is not int or self.settings['max_requests'] < 1:
            raise ValueError('positive request ceiling required')

    def feasible(self,record):
        return self.scorer.assess(record)['feasible']

    def reset(self):
        self.oracle = DesignOracle(self.workload,self.settings['budget'])
        self.compute_count = self.initial_count
        self.hardware = self.hardware_candidates[self.compute_count]
        self.current_mapping,self.compiler = conservative_initial_mapping(self.workload,self.hardware)
        self.initial = self.current_mapping
        self.current_record = self.best_record = None
        self.scorer = self.requirements.bind()
        self.reference = 1.0  # Unmeasured archive sentinel, never an initial PPA score.
        self.last_reward = 0.0
        self.current_objective = self.best_objective = 1.0
        self.best_mapping = self.best_hardware = None
        self.trace,self.structural_validation_time_s = [],0.0
        self._refresh()
        return self.observation

    def _retile_actions(self):
        actions,features = [],[]
        placements = self.current_mapping.placements
        for op in self.workload.operators:
            shape = self.workload.tensor_map[op.output].shape
            length = shape[2] if len(shape) == 4 else shape[1]
            old_group = tuple(t for t in self.current_mapping.tiles if t.operator == op.id)
            current = len(old_group)
            # Adjacent counts keep every partition count reachable, with
            # power-of-two jumps rather than enumerating every height at once.
            counts = part_counts(length,current)
            for count in counts:
                group = operator_tiles(self.workload,op.id,count)
                for target in self.hardware.compute_ids:
                    for distribute in ((False,True) if count > 1 and self.compute_count > 1 else (False,)):
                        offset = self.hardware.compute_ids.index(target)
                        places = tuple(self.hardware.compute_ids[(offset+i)%self.compute_count]
                            if distribute else target for i in range(len(group)))
                        if group == old_group and all(placements[t.id] == n for t,n in zip(group,places)): continue
                        actions.append(RetileAction(op.id,count,target,distribute))
                        features.append((tuple(t.id for t in old_group),target,
                            math.log1p(count)/math.log1p(length),float(distribute)))
        return actions,features

    def _independent_actions(self):
        actions,features = [],[]
        for tile in self.current_mapping.tiles:
            for target in self.hardware.compute_ids:
                if target == self.current_mapping.placements[tile.id]: continue
                actions.append(IndependentAction('place_tile',target=target,tile=tile.id))
                features.append(((tile.id,),target,0.0,0.5))
        for name,current in self.current_mapping.storage:
            op = self.workload.producers.get(name)
            if name in self.workload.parameter_map: op = name.split(':',1)[1]
            group = tuple(t.id for t in self.current_mapping.tiles if t.operator == op) or tuple(t.id for t in self.current_mapping.tiles)
            for target in self.hardware.memory_ids:
                if target == current: continue
                actions.append(IndependentAction('store_object',target=target,storage=name))
                features.append((group,target,0.0,1.0))
        for op in self.workload.operators:
            group = tuple(t for t in self.current_mapping.tiles if t.operator == op.id)
            shape = self.workload.tensor_map[op.output].shape
            length = shape[2] if len(shape) == 4 else shape[1]
            for count in part_counts(length,len(group)):
                if count == len(group): continue
                actions.append(IndependentAction('partition_operator',operator=op.id,parts=count))
                features.append((tuple(t.id for t in group),self.current_mapping.placements[group[0].id],
                    math.log1p(count)/math.log1p(length),0.0))
        return actions,features

    def _refresh(self):
        started = time.perf_counter()
        ma = (legal_actions(self.workload,self.hardware,self.current_mapping,self.settings['max_scope'])
            if self.settings['action_mode'] == 'coupled' else ())
        ca = [CountAction(n) for n in self.counts if n != self.compute_count]
        ra,rf = self._retile_actions() if self.settings['action_mode'] == 'coupled' else self._independent_actions()
        self.actions = tuple(ma)+tuple(ca)+tuple(ra)
        self.structural_validation_time_s += time.perf_counter()-started
        self.termination = ('budget_exhausted' if self.oracle.evaluations >= self.oracle.budget else
            'request_limit' if self.oracle.requests >= self.settings['max_requests'] else
            'no_legal_action' if not self.actions else None)
        self.done = self.termination is not None
        graph = observe(self.workload,self.hardware,self.current_mapping,self.current_record,ma,
            current=self.current_objective,best=self.best_objective,remaining=self.oracle.budget-self.oracle.evaluations,
            budget=self.oracle.budget,remaining_requests=(self.settings['max_requests']-self.oracle.requests)/self.settings['max_requests'],
            cached_mapping_hashes=self.oracle.cache,best_mapping_hash=digest(self.best_mapping) if self.best_mapping else '',
            last_reward=self.last_reward)
        from dataclasses import replace
        graph = replace(graph,global_features=(*graph.global_features[:6],
            self.compute_count/max(self.counts),*graph.global_features[7:]),
            task_features=self.scorer.features(self.current_record,self.feasible(self.best_record)))
        cf = tuple((a.compute_count/max(self.counts),*mesh_shape(a.compute_count),
            a.compute_count/max(self.counts)) for a in ca)
        self.observation = DesignObservation(graph,self.actions,tuple(True for _ in self.actions),tuple(cf),tuple(rf),
            self.compute_count,digest(self.best_hardware) if self.best_hardware else '')

    def step(self,action):
        if self.done: raise RuntimeError('episode has terminated')
        if action not in self.actions: raise ValueError('action excluded by structural mask')
        h,count = self.hardware,self.compute_count
        if isinstance(action,CountAction):
            count = action.compute_count
            h = self.hardware_candidates[count]
            mapping = resized_mapping(self.current_mapping,h)
        elif isinstance(action,RetileAction): mapping = retiled_mapping(self.workload,h,self.current_mapping,action)
        elif isinstance(action,IndependentAction): mapping = independent_mapping(self.workload,h,self.current_mapping,action)
        else: mapping = apply(self.workload,h,self.current_mapping,action)
        old_best = self.best_objective
        old_phi = self.phi()
        record = self.oracle.evaluate(h,mapping)
        feasible = self.feasible(record)
        if record.feasible:
            self.hardware,self.compute_count,self.current_mapping,self.current_record = h,count,mapping,record
            self.current_objective = self.scorer.objective(record)
            if self.scorer.better(record,self.best_record):
                self.best_objective,self.best_hardware,self.best_mapping,self.best_record = self.current_objective,h,mapping,record
        self._refresh()
        r = reward(1.0,old_best,self.best_objective,terminal=self.done,variant=self.settings['reward_variant'],
            old_phi=old_phi,new_phi=self.phi())
        self.last_reward = r
        from dataclasses import replace
        self.observation = replace(self.observation,graph=replace(self.observation.graph,
            global_features=(*self.observation.graph.global_features[:-1],r)))
        self.trace.append({'action':action,'feasible':feasible,'physical_feasible':record.feasible,
            'ppa_assessment':self.scorer.assess(record),'candidate_hardware_hash':record.hardware_hash,
            'candidate_mapping_hash':record.mapping_hash,'candidate_compute_count':count,'compute_count':self.compute_count,
            'current_objective':self.current_objective,'best_objective':self.best_objective,'reward':r,
            'evaluations':self.oracle.evaluations,'requests':self.oracle.requests,'cache_hits':self.oracle.cache_hits,'termination':self.termination})
        return self.observation,r,self.done

    def bottleneck(self):
        if self.current_record is None: return 0.0
        diagnostics = self.current_record.diagnostics['criticality']
        critical = [e for e in self.current_record.schedule_trace if e['job']['kind'] != 'compute'
            and diagnostics[e['job']['id']]['critical']]
        return max((e['finish_s']-e['start_s'] for e in critical),default=0)/self.current_record.latency_s

    def phi(self):
        return potential(1.0,self.best_objective,self.bottleneck(),self.settings['lambda_b'],self.settings['lambda_c'],self.done)


def nondominated(rows):
    keys = ('average_power_w','latency_s','area_mm2')
    valid = [r for r in rows if r['feasible']]
    return [r for r in valid if not any(all(q[k] <= r[k] for k in keys) and any(q[k] < r[k] for k in keys) for q in valid)]


def search_mesh(workload,folder,*,compute_counts=DEFAULT_COUNTS,episodes=3,budget=16,
                max_scope=2,seed=7,max_area_mm2=None,width=32,state_variant='full',
                action_mode='coupled',reward_variant='dense',lambda_c=0.0,requirements=None,early_stopping=None,recording='compact'):
    from .training import RACETrainer
    from .cli import export
    if type(episodes) is not int or episodes < 1: raise ValueError('positive episodes required')
    if recording not in ('compact','full'): raise ValueError('recording must be compact or full')
    plateau = ObjectivePlateau(PlateauSettings(**early_stopping)) if early_stopping is not None else None
    stop_reason = 'EPISODE_LIMIT'
    folder.mkdir(parents=True,exist_ok=True)
    if any(folder.iterdir()): raise ValueError('choose a fresh output directory')
    settings = dict(compute_counts=tuple(compute_counts),budget=budget,max_scope=max_scope,max_area_mm2=max_area_mm2,requirements=requirements,
        action_mode=action_mode,reward_variant=reward_variant,lambda_c=lambda_c)
    DesignEnvironment(workload,**settings)
    trainer = RACETrainer(seed=seed,width=width,precision='float64',state_variant=state_variant)
    recorder = None
    if recording == 'compact':
        from .recording import CompactRecorder
        recorder = CompactRecorder(folder/'history.sqlite',digest({'workload':workload,'settings':settings,'source':code_hashes()}))
    save_json(folder/'config.json',{'env':settings,'episodes':episodes,'early_stopping':early_stopping,'training':trainer.config,'source_hashes':code_hashes(),
        'recording':recording,'precision':'float64 workload/features/parameters/returns/policy-gradient arithmetic',
        'policy':'ONE shared encoder, actor and optimizer; no critic; all action families may occur at every step',
        'exploration':'5% uniform mixture in family and conditional distributions; actual mixture log probabilities',
        'profile':'compute-only mesh; external-memory services are not physical chiplets; fixed per-compute specs; uncalibrated capacity/area density',
        'pruning':'shape/coverage/placement constraints only; all requested counts may be tried in the environment','status':'UNIFIED_RL_DEVELOPMENT_PILOT'})
    started = time.perf_counter()
    rows,winner = [],None
    for i in range(episodes):
        env = DesignEnvironment(workload,**settings)
        env.progress_path = folder/"progress.json"
        stats = trainer.run_episode(env)
        stats.update(compute_count=len(env.best_hardware.compute_ids) if env.best_hardware else None,best_weighted_ppa_cost=env.scorer.assess(env.best_record)['weighted_cost'],
            requirements_satisfied=env.feasible(env.best_record),
            resize_steps=sum(isinstance(r['action'],CountAction) for r in env.trace),
            multi_tile_coupled_steps=sum(len(getattr(r['action'],'scope',()))>1 for r in env.trace),
            retile_steps=sum(isinstance(r['action'],RetileAction) for r in env.trace))
        if recorder is not None:
            recorder.record('train',seed,i+1,env,stats)
            if i == 0: save_json(folder/'compiler.json',env.compiler)
        else:
            export(folder/f'episode_{i+1:04d}',env,stats)
            if env.best_hardware is not None:
                save_json(folder/f'episode_{i+1:04d}'/('best_hardware.json' if env.feasible(env.best_record) else 'closest_hardware.json'),env.best_hardware)
            save_json(folder/f'episode_{i+1:04d}'/'compiler.json',env.compiler)
        for key,record in env.oracle.cache.items():
            h,mapping = env.oracle.designs[key]
            rows.append({'episode':i+1,'design_hash':key,'compute_count':len(h.compute_ids),'total_chiplets':len(h.nodes),
                'feasible':env.feasible(record),'ppa_assessment':env.scorer.assess(record),'energy_j':record.energy_j,'latency_s':record.latency_s,
                'average_power_w':record.average_power_w,'area_mm2':record.area_mm2,'mapping_hash':record.mapping_hash,
                'hardware_hash':record.hardware_hash,'violations':record.violations})
        if winner is None or env.scorer.better(env.best_record,winner.best_record): winner = env
        print(f"Unified REINFORCE episode {i+1}: selected compute={stats['compute_count']}, PPA cost={stats['best_weighted_ppa_cost']}, compliant={env.feasible(env.best_record)}, evaluations={env.oracle.evaluations}, resize={stats['resize_steps']}, retile={stats['retile_steps']}",flush=True)
        should_stop = plateau.update(winner.scorer.assess(winner.best_record),i+1) if plateau else False
        stats['plateau_stale_episodes'] = plateau.stale if plateau else None
        save_json(folder/'training_curve.json',trainer.history)
        if should_stop:
            stop_reason='OBJECTIVE_PLATEAU'
            break
    identity = digest({'workload':workload,'settings':settings,'early_stopping':early_stopping,'source':code_hashes()})
    trainer.save(folder/'checkpoint.pt',experiment_identity=identity)
    prefix = 'best' if winner.feasible(winner.best_record) else 'closest'
    if winner.best_record is not None:
        save_json(folder/f'{prefix}_hardware.json',winner.best_hardware)
        save_json(folder/f'{prefix}_mapping.json',winner.best_mapping)
        save_json(folder/f'{prefix}_evaluation.json',winner.best_record)
    report = winner.scorer.report(winner.best_record)
    report['status'] = 'FEASIBLE_DESIGN_FOUND' if prefix == 'best' else 'NO_FEASIBLE_DESIGN_FOUND'
    save_json(folder/'vendor_report.json',report)
    save_json(folder/'candidate_metrics.json',rows)
    save_json(folder/'pareto_frontier.json',nondominated(rows))
    summary = {'status':'UNIFIED_RL_DEVELOPMENT_PILOT','single_policy':True,'algorithm':'reinforce','critic_used':False,
        'requirements_status':report['status'],'requirements':winner.requirements,
        'episodes_requested':episodes,'episodes_completed':trainer.episodes,'search_stop_reason':stop_reason,
        'early_stopping':early_stopping,'convergence_proven':False,
        'selected_compute_count':len(winner.best_hardware.compute_ids) if winner.best_hardware else None,
        'selected_total_chiplets':len(winner.best_hardware.nodes) if winner.best_hardware else None,
        'best_compute_count':len(winner.best_hardware.compute_ids) if prefix == 'best' else None,
        'best_weighted_ppa_cost':report['assessment']['weighted_cost'] if prefix == 'best' else None,
        'selected_assessment':report['assessment'],
        'total_oracle_evaluations':trainer.training_evaluations,'oracle_budget_ceiling':episodes*budget,
        'initial_evaluations_included':False,'evaluations_before_first_action':0,
        'policy_updates':sum(s['parameter_updates'] for s in trainer.history),
        'counts_evaluated':sorted({r['compute_count'] for r in rows}),'counts_requested':compute_counts,
        'resize_steps':sum(s['resize_steps'] for s in trainer.history),'retile_steps':sum(s['retile_steps'] for s in trainer.history),
        'multi_tile_coupled_steps':sum(s['multi_tile_coupled_steps'] for s in trainer.history),
        'wall_time_s':time.perf_counter()-started,'source_hashes':code_hashes(),'recording':recording,
        'limitations':'finite count/output-partition candidates; short same-workload learning; no global-optimum or silicon PPA guarantee',
        'reward_contract':{'global_reference':'unmeasured archive sentinel J=1; measured J=M/(1+M); fixed vendor-limit scales or declared SI unit scales','variant':reward_variant,'lambda_c':lambda_c,
            'gamma':1.0,'accepts_worse_feasible_states':True}}
    save_json(folder/'summary.json',summary)
    if recorder is not None: recorder.close()
    return summary
