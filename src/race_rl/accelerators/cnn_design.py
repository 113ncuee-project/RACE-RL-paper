"""Complete CNN transactions; H/W/C partition and K blocking are distinct."""
from dataclasses import dataclass, replace
from collections import deque
from functools import lru_cache
from ..tiles import Tile,Region

# Physical hardware stays fixed. Profiles only select software blocking.
PROFILES=((8,8,16,16),(8,16,32,32),(16,8,32,32),(16,16,64,64))

@dataclass(frozen=True)
class LayerPlan:
    h: int=8
    w: int=8
    c: int=16
    k: int=16
    target: int=0
    spread: bool=True
    keep_output: bool=False
    keep_weights: bool=False

    def __post_init__(self):
        if (self.h,self.w,self.c,self.k) not in PROFILES:
            raise ValueError('unknown CNN blocking profile')
        if type(self.target) is not int or not 0<=self.target<16:
            raise ValueError('unknown chiplet target')
        if any(type(getattr(self,key)) is not bool for key in ('spread','keep_output','keep_weights')):
            raise ValueError('CNN residency/layout flags must be boolean')

@dataclass(frozen=True)
class CNNDesign:
    compute_count: int
    plans: tuple[LayerPlan,...]

    def __post_init__(self):
        object.__setattr__(self,'plans',tuple(self.plans))
        if type(self.compute_count) is not int or not 1<=self.compute_count<=16:
            raise ValueError('RL must select 1–16 chiplets')
        if not self.plans or any(not isinstance(p,LayerPlan) or p.target>=self.compute_count for p in self.plans):
            raise ValueError('complete CNN plans with existing targets required')

@dataclass(frozen=True)
class CNNAction:
    design: CNNDesign
    choices: tuple[int,...] # count-1, anchor, then per-scope profile/residency/layout/target
    scope: tuple[int,...]
    kind: str='cnn_coupled_edit'


@lru_cache(maxsize=4096)
def coupled_scope(workload,anchor,size):
    adjacency={i:set() for i in range(len(workload.operators))}
    indices={op.id:i for i,op in enumerate(workload.operators)}
    for i,op in enumerate(workload.operators):
        for name in op.inputs:
            if name in workload.producers:
                j=indices[workload.producers[name]]
                adjacency[i].add(j);adjacency[j].add(i)
    selected=[];queue=deque([anchor]);seen={anchor}
    while queue and len(selected)<size:
        i=queue.popleft()
        if workload.operators[i].kind not in ('Flatten','Identity'):selected.append(i)
        for j in sorted(adjacency[i]):
            if j not in seen:queue.append(j);seen.add(j)
    return tuple(selected)


def transaction(workload,current,choices,max_scope):
    if len(choices)<6 or any(type(v) is not int for v in choices):
        raise ValueError('complete integer actor decisions required')
    count,anchor=choices[:2]
    count+=1
    if not (1<=count<=16 and 0<=anchor<len(workload.operators)):
        raise ValueError('out-of-range CNN action decision')
    plans=[replace(p,target=p.target%count) for p in current.plans]
    scope=coupled_scope(workload,anchor,max_scope)
    if len(choices)!=2+4*len(scope):raise ValueError('one decision set per coupled operator required')
    for i,index in enumerate(scope):
        profile,residency,layout,target=choices[2+4*i:6+4*i]
        if not (0<=profile<len(PROFILES) and 0<=residency<4 and layout in (0,1) and 0<=target<count):
            raise ValueError('out-of-range CNN action decision')
        plans[index]=LayerPlan(*PROFILES[profile],target,bool(layout),bool(residency&1),bool(residency&2))
    return CNNAction(CNNDesign(count,tuple(plans)),tuple(choices),scope)


def output_tiles(workload,op,plan,count):
    shape=workload.tensor_map[op.output].shape
    index=0
    if len(shape)==4:
        n,c,h,w=shape
        for b in range(n):
            for channel in range(0,c,plan.c):
                for y in range(0,h,plan.h):
                    for x in range(0,w,plan.w):
                        bounds=((b,b+1),(channel,min(c,channel+plan.c)),(y,min(h,y+plan.h)),(x,min(w,x+plan.w)))
                        node=(plan.target+index)%count if plan.spread else plan.target
                        yield Tile(f'{op.id}:{index}',op.id,Region(bounds)),node
                        index+=1
    else:
        n,c=shape
        for b in range(n):
            for channel in range(0,c,plan.c):
                node=(plan.target+index)%count if plan.spread else plan.target
                yield Tile(f'{op.id}:{index}',op.id,Region(((b,b+1),(channel,min(c,channel+plan.c))))),node
                index+=1
