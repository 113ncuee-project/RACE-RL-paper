#!/usr/bin/env python3
"""Coupled-edit ablation: scope 1 / dependent scope 2 / random non-adjacent scope 2 / scope 4.

Exploratory experiment. Every arm uses the same REINFORCE trainer, CNN
environment, requirements, cost profile and 64-proposal budget (16 per
episode), exactly as the common-DSE `reinforce` method. Only the coupled
scope differs. Arm C replaces the dependency-graph partner with a fixed,
seeded, non-adjacent material operator by patching `coupled_scope` inside
this process only; `src/` is not modified.

Decomposition test (arms with >=2 operators): for every proposal that
improved the episode best, the count-only, anchor-only and partner-only
half designs are evaluated off-budget with a separate oracle. They never
feed the policy and are reported separately.
"""
import argparse
from dataclasses import replace,asdict
from functools import lru_cache
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

ARMS={'A_scope1':dict(scope=1,partner='dependency'),
      'B_scope2_dependent':dict(scope=2,partner='dependency'),
      'C_scope2_random_nonadjacent':dict(scope=2,partner='random_nonadjacent'),
      'D_scope4_dependent':dict(scope=4,partner='dependency')}
MODELS=('resnet18','vgg16','mobilenet_v2')
PREFERENCES=('balanced','power','performance','area')
SEEDS=tuple(range(601,621))


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def material_neighbors(workload,anchor):
    """Material operators adjacent to `anchor`, looking through Flatten/Identity aliases."""
    ops=workload.operators;index={op.id:i for i,op in enumerate(ops)}
    adj={i:set() for i in range(len(ops))}
    for i,op in enumerate(ops):
        for name in op.inputs:
            if name in workload.producers:
                j=index[workload.producers[name]];adj[i].add(j);adj[j].add(i)
    seen={anchor};frontier=list(adj[anchor]);result=set()
    while frontier:
        j=frontier.pop()
        if j in seen:continue
        seen.add(j)
        if ops[j].kind in ('Flatten','Identity'):frontier.extend(adj[j])
        else:result.add(j)
    return result


def install_random_partner(seed):
    """Patch coupled_scope in every module that bound it, for this process only."""
    import race_rl.accelerators.cnn_design as design
    import race_rl.accelerators.cnn_policy as policy
    import race_rl.accelerators.cnn_environment as environment
    original=design.coupled_scope.__wrapped__ if hasattr(design.coupled_scope,'__wrapped__') else design.coupled_scope

    @lru_cache(maxsize=None)
    def random_scope(workload,anchor,size):
        if size!=2:return original(workload,anchor,size)
        material=[i for i,op in enumerate(workload.operators) if op.kind not in ('Flatten','Identity')]
        excluded=material_neighbors(workload,anchor)|{anchor}
        pool=[i for i in material if i not in excluded]
        if not pool:raise ValueError(f'no non-adjacent partner for anchor {anchor}')
        rng=random.Random(f'{seed}:{anchor}')  # fixed per (trial seed, anchor), like the BFS partner
        return (anchor,rng.choice(pool))
    previous={m:m.coupled_scope for m in (design,policy,environment)}
    for module in previous:module.coupled_scope=random_scope
    return lambda:[setattr(m,'coupled_scope',f) for m,f in previous.items()]


def run_trial(workload,model,preference,seed,arm,out,*,memory,platform,profile,budget,episode_size):
    spec=ARMS[arm]
    restore=install_random_partner(seed) if spec['partner']=='random_nonadjacent' else None
    try:return _run_trial(workload,model,preference,seed,arm,out,spec=spec,memory=memory,platform=platform,
                          profile=profile,budget=budget,episode_size=episode_size)
    finally:
        if restore:restore()


def _run_trial(workload,model,preference,seed,arm,out,*,spec,memory,platform,profile,budget,episode_size):
    from race_rl.accelerators.cnn_environment import GemminiCNNEnvironment,CNNOracle
    from race_rl.accelerators.cnn_design import CNNDesign
    from race_rl.requirements import PPARequirements
    from race_rl.recording import assessment_better
    from race_rl.serialization import digest,plain,save_json
    from race_rl.training import RACETrainer
    requirements=PPARequirements(**read(ROOT/f'experiments/runs/common_dse_study_v2/requirements/{model}/{preference}.json')).require_vendor_inputs()
    side_oracle=CNNOracle(GemminiCNNEnvironment(workload,requirements=requirements,memory=memory,platform=platform,
        cost_profile=profile,budget=1,max_scope=spec['scope']).model,profile,10**9)
    curve=[];decomposition=[];started=time.perf_counter()

    class AblationEnv(GemminiCNNEnvironment):
        def step(self,action):
            before_design=self.current_design;before_best=self.best_objective
            result=super().step(action)
            record=self.oracle.cache[digest(action.design)]
            a=self.scorer.assess(record) if record.scorable else self.scorer.assess(None)
            curve.append({'physical_feasible':record.physical_feasible,'feasible':bool(record.scorable and a['feasible']),
                          'merit':a['search_merit'],'weighted_cost':a['weighted_cost'] if a['feasible'] else None,
                          'cached':self.trace[-1]['cached'],'scope_size':len(action.scope),
                          'scope':[self.workload.operators[i].id for i in action.scope],'compute_count':action.design.compute_count})
            if len(action.scope)>=2 and self.best_objective<before_best:
                decomposition.append(decompose(self,before_design,before_best,action))
            return result

    def objective(env,design):
        record=side_oracle.evaluate(design)
        return env.scorer.objective(record) if record.scorable else 1.0

    def decompose(env,before,best_before,action):
        count=action.design.compute_count
        base=[replace(p,target=p.target%count) for p in before.plans]
        def with_ops(ops):
            plans=list(base)
            for i in ops:plans[i]=action.design.plans[i]
            return CNNDesign(count,tuple(plans))
        anchor,*partners=action.scope
        j_base=objective(env,with_ops(()));j_anchor=objective(env,with_ops((anchor,)))
        j_partner=objective(env,with_ops(tuple(partners)));j_joint=env.oracle.cache[digest(action.design)]
        j_joint=env.scorer.objective(j_joint)
        gain=lambda j:math.log(j_base/j)
        return {'j_base':j_base,'j_anchor_only':j_anchor,'j_partner_only':j_partner,'j_joint':j_joint,
                'best_before':best_before,'interaction':gain(j_joint)-gain(j_anchor)-gain(j_partner),
                'joint_only':j_anchor>=best_before and j_partner>=best_before and j_base>=best_before}

    def environment(n):
        return AblationEnv(workload,requirements=requirements,memory=memory,platform=platform,
            cost_profile=profile,budget=n,max_scope=spec['scope'])
    trainer=RACETrainer(seed=seed,precision='float64',policy_backend='gemmini_cnn')
    requests=evaluations=cache_hits=0;winner=winner_design=None;episodes=[]
    while requests<budget:
        env=environment(min(episode_size,budget-requests))
        stats=trainer.run_episode(env)
        episodes.append({k:stats.get(k) for k in ('steps','oracle_evaluations','oracle_requests','cache_hits','best_objective','return','termination','entropy')})
        requests+=env.oracle.requests;evaluations+=env.oracle.evaluations;cache_hits+=env.oracle.cache_hits
        candidate=env.scorer.assess(env.best_record)
        if assessment_better(candidate,winner):winner,winner_design=candidate,env.best_design
        if not env.oracle.requests:raise RuntimeError('no progress')
    result={'model':model,'preference':preference,'seed':seed,'arm':arm,'arm_spec':spec,'budget':budget,
            'episode_size':episode_size,'requests':requests,'actual_evaluations':evaluations,'cache_hits':cache_hits,
            'best_assessment':winner,'best_compute_count':winner_design.compute_count if winner_design else None,
            'best_design':plain(winner_design) if winner_design else None,'curve':curve,'episodes':episodes,
            'decomposition':decomposition,'side_evaluations':side_oracle.evaluations,
            'wall_time_s':time.perf_counter()-started,'trainer_config':trainer.config}
    save_json(out,result)
    return result


def cmd_run(args):
    import gc
    import torch
    from race_rl.accelerators.cnn_search import prepare_model
    from race_rl.accelerators.cnn_ppa import load_cnn_cost_profile
    from race_rl.accelerators.gemmini_model import PlatformScenario
    from race_rl.accelerators.memory_hierarchy import MemoryHierarchy
    from race_rl.serialization import code_hashes,save_json
    torch.set_num_threads(1)
    out=args.out;(out/'models'/args.model).mkdir(parents=True,exist_ok=True)
    memory=MemoryHierarchy.from_record(read(ROOT/'configs/memory_hierarchy.json'))
    platform=PlatformScenario.from_record(read(ROOT/'configs/gemmini_platform.json'))
    profile=load_cnn_cost_profile(read(ROOT/'configs/ppa_profiles/cnn_45nm_reference.json'),memory=memory)
    protocol={'status':'EXPLORATORY_COUPLED_EDIT_ABLATION','models':MODELS,'preferences':args.preferences,'seeds':args.seeds,
              'arms':{a:ARMS[a] for a in args.arms},'budget':args.budget,'episode_size':args.episode_size,'input_size':64,
              'model_seed':7,'requirements':'experiments/runs/common_dse_study_v2/requirements (reference-v2, frozen)',
              'cost_profile':'configs/ppa_profiles/cnn_45nm_reference.json','source_hashes':code_hashes(),
              'tool_sha256':__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()}
    pfile=out/'models'/args.model/'protocol.json'
    if pfile.exists() and read(pfile)['source_hashes']!=protocol['source_hashes']:
        raise SystemExit('source changed since this run started; refusing to mix versions')
    save_json(pfile,protocol)
    workload,_=prepare_model(args.model,64,7,out/'models'/args.model)
    tasks=[(p,s,a) for s in args.seeds for p in args.preferences for a in args.arms]
    for k,(pref,seed,arm) in enumerate(tasks,1):
        path=out/'trials'/args.model/pref/str(seed)/f'{arm}.json'
        if path.exists():continue
        path.parent.mkdir(parents=True,exist_ok=True)
        r=run_trial(workload,args.model,pref,seed,arm,path,memory=memory,platform=platform,profile=profile,
                    budget=args.budget,episode_size=args.episode_size)
        a=r['best_assessment'] or {}
        print(f"[{args.model} {k}/{len(tasks)}] {pref} seed={seed} {arm}: feasible={a.get('feasible')} "
              f"cost={a.get('weighted_cost')} n={r['best_compute_count']} {r['wall_time_s']:.1f}s",flush=True)
        gc.collect()
    save_json(out/'models'/args.model/'done.json',{'completed':len(tasks)})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True)
    r=sub.add_parser('run');r.add_argument('--model',choices=MODELS,required=True)
    r.add_argument('--arms',nargs='+',choices=tuple(ARMS),default=list(ARMS))
    r.add_argument('--preferences',nargs='+',choices=PREFERENCES,default=list(PREFERENCES))
    r.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    r.add_argument('--out',type=Path,required=True);r.add_argument('--budget',type=int,default=64)
    r.add_argument('--episode-size',type=int,default=16)
    args=p.parse_args()
    sys.path.insert(0,str(ROOT/'src'))
    cmd_run(args)


if __name__=='__main__':main()
