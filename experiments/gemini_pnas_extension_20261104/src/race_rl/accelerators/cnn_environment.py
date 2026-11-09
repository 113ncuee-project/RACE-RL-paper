"""One complete coupled CNN action -> environment -> reward and next state."""
from dataclasses import dataclass,replace,asdict
import math
from ..serialization import digest,code_hashes
from ..requirements import PPARequirements
from ..workload import OPERATOR_KINDS
from ..rewards import reward
from ..observation import GraphObservation
from ..tiles import whole,Tile,operations,macs
from ..oracle import BudgetExhausted
from .gemmini_environment import GemminiDesignEnvironment,GemminiEvaluation
from .gemmini_model import GemmShape,PlatformScenario
from .memory_hierarchy import MemoryHierarchy
from .cnn_model import GemminiCNNModel
from .cnn_design import CNNDesign,CNNAction,LayerPlan,transaction,PROFILES,coupled_scope
from .cnn_ppa import cnn_hardware_record


@dataclass(frozen=True)
class CNNObservation:
    graph: GraphObservation
    current_design: CNNDesign
    workload: object
    max_scope: int
    model_feedback_valid: bool
    ppa_valid: bool
    last_reward_valid: bool
    action_support_size: int
    anchor_indices: tuple
    backend: str='race_gemmini_cnn_hierarchy_v1'


class CNNOracle:
    def __init__(self,model,profile,budget):
        self.model,self.profile,self.budget=model,profile,budget
        self.requests=self.evaluations=self.cache_hits=0
        self.cache={}

    def evaluate(self,design):
        key=digest(design)
        if key not in self.cache and self.evaluations>=self.budget:raise BudgetExhausted('CNN budget exhausted')
        self.requests+=1
        if key in self.cache:self.cache_hits+=1;return self.cache[key]
        self.evaluations+=1
        try:feedback=self.model.evaluate(design)
        except ValueError as error:
            result=GemminiEvaluation(key,False,self.profile is not None,'PHYSICAL_INVALID',(str(error),),None,None)
        else:
            ppa=self.profile.assess(feedback) if self.profile else None
            result=GemminiEvaluation(key,True,ppa is not None,'EVALUATED' if ppa else 'PPA_UNAVAILABLE',(),feedback,ppa,
                **({name:ppa[name] for name in ('energy_j','latency_s','average_power_w','area_mm2')} if ppa else {}))
        self.cache[key]=result
        return result


class GemminiCNNEnvironment:
    def __init__(self,workload,*,requirements=None,memory=None,platform=None,cost_profile=None,
                 allow_test_fixture=False,budget=64,max_scope=2):
        workload.validate()
        if type(budget) is not int or budget<1 or type(max_scope) is not int or not 1<=max_scope<=4:
            raise ValueError('positive interaction budget and coupled scope 1–4 required')
        if any(t.bytes_per_element!=1 for t in workload.tensors):
            raise ValueError('explicitly calibrated INT8 workload required; no silent FP32-to-INT8 reinterpretation')
        if not any(key=='quantization_sha256' for key,_ in workload.source_metadata):
            raise ValueError('quantization protocol identity required')
        self.workload=workload;self.requirements=requirements or PPARequirements()
        self.memory=memory or MemoryHierarchy();self.platform=platform or PlatformScenario()
        self.cost_profile=cost_profile.validate(allow_test_fixture=allow_test_fixture,memory=self.memory) if cost_profile else None
        if cost_profile and self.platform.clock_hz not in (None,cost_profile.base.clock_hz):
            raise ValueError('CNN platform/cost clock mismatch')
        self.budget,self.max_scope=budget,max_scope
        self.anchor_indices=tuple(i for i,op in enumerate(workload.operators) if op.kind not in ('Flatten','Identity'))
        if not self.anchor_indices:raise ValueError('CNN search requires a material operator, not only aliases')
        self.action_support_size=sum((len(PROFILES)*4*2*n)**len(coupled_scope(workload,i,max_scope))
                    for n in range(1,17) for i in self.anchor_indices)
        self.model=GemminiCNNModel(workload,memory=self.memory,platform=self.platform)
        self.hw_graphs={}

    @property
    def experiment_identity(self):
        return digest({'workload':self.workload,'requirements':self.requirements,'memory':self.memory.record(),
                       'platform':self.platform,'cost_profile':self.cost_profile,'budget':self.budget,
                       'max_scope':self.max_scope,'hardware':cnn_hardware_record(self.memory),'code':code_hashes()})

    def reset(self):
        self.oracle=CNNOracle(self.model,self.cost_profile,self.budget)
        self.current_design=CNNDesign(2,tuple(LayerPlan() for _ in self.workload.operators))
        self.compute_count=2
        self.current_record=self.best_record=self.current_feedback=self.best_design=None
        self.current_objective=self.best_objective=1.0
        self.last_reward=0.0;self.last_reward_valid=False
        self.scorer=self.requirements.bind();self.trace=[];self.done=False;self.termination=None
        self._refresh();return self.observation

    def feasible(self,record):return bool(record and record.scorable and self.scorer.assess(record)['feasible'])

    def _graph(self):
        count=self.current_design.compute_count
        if count not in self.hw_graphs:
            basis=GemminiDesignEnvironment(GemmShape(16,16,16),compute_counts=(count,),initial_count=count,memory=self.memory)
            graph=basis.reset().graph
            names=list(graph.hardware_ids);features=list(graph.hardware_features)
            edges=list(graph.hardware_edge_index);edge_features=list(graph.hardware_edge_features)
            old_index={name:i for i,name in enumerate(names)}
            for node in range(count):
                array=features[old_index[f'c{node}.array']]
                vector_index=len(names);names.append(f'c{node}.vector')
                features.append((1,0,0,math.log1p(16),0,0,array[6],array[7],0))
                for operand in ('A','B','psum'):
                    rf=old_index[f'c{node}.rf{operand}'];bw=features[rf][5]
                    edges.extend(((rf,vector_index),(vector_index,rf)))
                    edge_features.extend(((bw,0,0,0),(bw,0,0,0)))
            self.hw_graphs[count]=replace(graph,hardware_ids=tuple(names),hardware_features=tuple(features),
                hardware_edge_index=tuple(edges),hardware_edge_features=tuple(edge_features))
        base=self.hw_graphs[count];w=self.workload
        names=tuple(op.id for op in w.operators);oi={name:i for i,name in enumerate(names)};hi={name:i for i,name in enumerate(base.hardware_ids)}
        measured=self.current_feedback is not None;scored=self.current_record is not None
        rows={row['operator']:row for row in (self.current_feedback or {}).get('layers',())}
        features=[];placement=[]
        for op,plan in zip(w.operators,self.current_design.plans):
            tile=Tile(op.id,op.id,whole(w.tensor_map[op.output].shape))
            shape=w.tensor_map[op.output].shape
            input_shape=w.tensor_map[op.inputs[0]].shape
            axes=(*shape,1,1) if len(shape)==2 else shape
            input_axes=(*input_shape,1,1) if len(input_shape)==2 else input_shape
            elapsed=rows.get(op.id,{}).get('cycles',0)/(self.current_feedback['modeled_cycles'] if measured else 1)
            features.append((*[float(op.kind==kind) for kind in OPERATOR_KINDS],math.log1p(operations(w,tile)),
                math.log1p(sum(w.tensor_map[name].bytes for name in op.inputs)),math.log1p(w.tensor_map[op.output].bytes),
                elapsed,0.0,0.0,math.log1p(plan.h),math.log1p(plan.w),math.log1p(plan.c),math.log1p(plan.k),
                float(plan.keep_output),float(plan.keep_weights),plan.target/16,float(plan.spread),
                *(math.log1p(v) for v in (*axes,*op.kernel,*op.stride,*op.padding,*op.dilation,op.groups,*input_axes))))
            actual=rows.get(op.id,{}).get('placements',{})
            if op.kind in ('Flatten','Identity'):
                root=self.model.roots[op.output]
                producer=w.producers.get(root)
                if producer:
                    parent=w.operator_map[producer]
                    actual=rows.get(producer,{}).get('placements',{})
                    parent_target=self.current_design.plans[self.model.op_index[producer]].target
                    owners=actual if actual else {f'c{parent_target}':1}
                    engine='array' if parent.kind in ('Conv','Linear') else 'vector'
                    for owner in owners:placement.append((oi[op.id],hi[owner+'.'+engine]))
                else:placement.append((oi[op.id],hi['dram0']))
                continue
            owners=actual if actual else {f'c{plan.target}':1}
            for owner in owners:
                placement.append((oi[op.id],hi[owner+'.vector']))
                if op.kind in ('Conv','Linear'):placement.append((oi[op.id],hi[owner+'.array']))
        deps=tuple(dict.fromkeys((oi[w.producers[name]],oi[op.id]) for op in w.operators for name in op.inputs if name in w.producers))
        link_bytes={(f'c{x["from"]}.dma',f'c{x["to"]}.dma'):x['bytes'] for x in (self.current_feedback or {}).get('directed_links',())}
        ef=tuple((old[0],math.log1p(link_bytes.get((base.hardware_ids[a],base.hardware_ids[b]),0)),0.0,0.0)
                 for (a,b),old in zip(base.hardware_edge_index,base.hardware_edge_features))
        edge_index={(base.hardware_ids[a],base.hardware_ids[b]):i for i,(a,b) in enumerate(base.hardware_edge_index)}
        relations=[]
        for row in (self.current_feedback or {}).get('dependency_link_bytes',()):
            dep=(oi[row['producer']],oi[row['consumer']])
            link=(f'c{row["from"]}.dma',f'c{row["to"]}.dma')
            if dep in deps and link in edge_index:
                relations.append((deps.index(dep),edge_index[link],f'{row["producer"]}>{row["consumer"]}',row['direction']))
        current=self.current_record
        global_base=(math.log(self.current_objective),math.log(self.best_objective),
            (self.budget-self.oracle.evaluations)/self.budget,self.oracle.evaluations/self.budget,
            current.average_power_w if scored else 0,current.latency_s if scored else 0,count/16,
            (self.budget-self.oracle.requests)/self.budget,float(scored),self.last_reward)
        shape=w.tensor_map[w.operators[0].inputs[0]].shape
        access=(self.current_feedback or {}).get('traffic',{}).get('storage_access_bytes',{})
        peak=max((self.current_feedback or {}).get('storage',{}).get('gb_activation_peak_bytes_by_node',[0]))
        extra=(math.log1p(len(w.operators)),math.log1p(shape[-2]),math.log1p(shape[-1]),
               math.log1p(sum(row.get('logical_macs',0) for row in rows.values())),
               *(math.log1p(access.get(level,0)) for level in ('gb','lb','rf','accumulator')),
               peak/65536,math.log1p((self.current_feedback or {}).get('operations',{}).get('vector_ops',0)),float(measured),float(self.last_reward_valid))
        masks={'tile':tuple((True,)*14+(measured,False,False)+(True,)*25 for _ in features),
               'hardware':tuple((True,)*8+(False,) for _ in base.hardware_features),
               'link':tuple((True,measured,False,False) for _ in ef),
               'global':(scored,scored,True,True,scored,scored,True,True,True,self.last_reward_valid)+(True,)*3+(measured,)*7+(True,)*2,
               'task':(True,)*9+(scored,)*6+(True,)}
        return GraphObservation(8,names,base.hardware_ids,tuple(features),base.hardware_features,deps,
            tuple((1.0,) for _ in deps),base.hardware_edge_index,ef,tuple(placement),tuple(sorted(set(relations))),(*global_base,*extra),(),(),masks,
            tuple(self.oracle.cache),self.best_record.design_hash if self.best_record else '',base.memory_port_edge_index,
            self.scorer.features(current,self.feasible(self.best_record)))

    def _refresh(self):
        if not self.termination:
            self.termination='request_limit' if self.oracle.requests>=self.budget else 'budget_exhausted' if self.oracle.evaluations>=self.budget else None
        self.done=self.termination is not None
        support=self.action_support_size
        self.observation=CNNObservation(self._graph(),self.current_design,self.workload,self.max_scope,
            self.current_feedback is not None,self.current_record is not None,self.last_reward_valid,support,self.anchor_indices)

    def step(self,action):
        if self.done:raise RuntimeError('CNN episode terminated')
        if not isinstance(action,CNNAction) or action!=transaction(self.workload,self.current_design,action.choices,self.max_scope):
            raise ValueError('complete CNN transaction inconsistent with observed state')
        before=self.best_objective
        state_before={'design_hash':digest(self.current_design),'compute_count':self.compute_count,
            'objective':self.current_objective,'best_objective':self.best_objective,'last_reward':self.last_reward,
            'last_reward_valid':self.last_reward_valid,'ppa_valid':self.current_record is not None}
        cache_before=self.oracle.cache_hits
        result=self.oracle.evaluate(action.design)
        if result.physical_feasible:
            self.current_design=action.design;self.compute_count=action.design.compute_count;self.current_feedback=result.model_feedback
            self.current_record=result if result.scorable else None
            if result.scorable:
                self.current_objective=self.scorer.objective(result)
                if self.scorer.better(result,self.best_record):
                    self.best_record=result;self.best_design=action.design;self.best_objective=self.current_objective
        if not result.scorable:
            r=None;self.last_reward_valid=False;self.termination='ppa_unavailable'
        else:r=reward(1.0,before,self.best_objective,variant='dense');self.last_reward_valid=True
        self.last_reward=0.0 if r is None else r
        self.trace.append({'choices':action.choices,'scope':[self.workload.operators[i].id for i in action.scope],
            'edits':{self.workload.operators[i].id:asdict(action.design.plans[i]) for i in action.scope},
            'compute_count':action.design.compute_count,'design_hash':result.design_hash,'reward':r,
            'physical_feasible':result.physical_feasible,'status':result.evaluation_status,'violations':result.violations,
            'current_objective':self.current_objective,'best_objective':self.best_objective,
            'model_cycles':result.model_feedback['modeled_cycles'] if result.model_feedback else None,
            'ppa':result.ppa_detail and {key:result.ppa_detail[key] for key in ('average_power_w','latency_s','area_mm2')},
            'requests':self.oracle.requests,'evaluations':self.oracle.evaluations,'cache_hits':self.oracle.cache_hits})
        self.trace[-1].update(state_before=state_before,
            state_after={'design_hash':digest(self.current_design),'compute_count':self.compute_count,
                'objective':self.current_objective,'best_objective':self.best_objective,'last_reward':self.last_reward,
                'last_reward_valid':self.last_reward_valid,'ppa_valid':self.current_record is not None},
            cached=self.oracle.cache_hits>cache_before)
        self._refresh();return self.observation,r,self.done
