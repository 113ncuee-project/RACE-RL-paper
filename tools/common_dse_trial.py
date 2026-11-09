#!/usr/bin/env python3
"""Run three DSE methods on an identical CNN/PPA task (exploratory only)."""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import time

from race_rl.accelerators.cnn_environment import GemminiCNNEnvironment
from race_rl.accelerators.cnn_ppa import load_cnn_cost_profile
from race_rl.accelerators.cnn_search import prepare_model, read
from race_rl.accelerators.gemmini_model import PlatformScenario
from race_rl.accelerators.memory_hierarchy import MemoryHierarchy
from race_rl.recording import assessment_better
from race_rl.requirements import PPARequirements
from race_rl.serialization import code_hashes, digest, save_json
from race_rl.training import RACETrainer
from common_dse_evaluator import CommonDSEEvaluator
import gemini_common_dse
import chiplet_gym_dse
import common_dse_baselines
import common_dse_genetic
import chiplet_gym_common
import common_dse_bayes
import common_dse_timeloop_restricted
from paper_comparison import confuciux_bridge, gamma_bridge, csdse_bridge, moham_bridge

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=('resnet18','mobilenet_v2','vgg16'),required=True)
    parser.add_argument('--input-size',type=int,default=64)
    parser.add_argument('--seed',type=int,default=301)
    parser.add_argument('--model-seed',type=int,default=7)
    parser.add_argument('--budget',type=int,default=64)
    parser.add_argument('--rl-episode-size',type=int,default=16)
    parser.add_argument('--requirements',type=Path,required=True)
    parser.add_argument('--profile',type=Path,default=ROOT/'configs/ppa_profiles/cnn_45nm_reference.json')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--methods', nargs='+', choices=('random','local_random','sa','genetic','bayesopt_gp_ei','timeloop_hybrid_restricted','gemini_common_restricted','chiplet_gym_common_full_design','chiplet_gym_common_local_edits','confuciux_actor_ga_restricted','gamma_mapping_restricted','digamma_joint_restricted','csdse_cooperative_restricted','moham_nsga_restricted','reinforce'), default=['random','local_random','sa','gemini_common_restricted','chiplet_gym_common_full_design','chiplet_gym_common_local_edits','reinforce'])
    args=parser.parse_args()
    if (args.budget<4 or args.rl_episode_size<1 or not 32<=args.input_size<=512 or
            any(not 0<=seed<2**31 for seed in (args.seed,args.model_seed)) or len(args.methods)!=len(set(args.methods))):
        parser.error('invalid input size, seed, budget, or episode size')
    if args.out.exists() and any(args.out.iterdir()): parser.error('choose a fresh output directory')
    args.out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    memory=MemoryHierarchy.from_record(read(ROOT/'configs/memory_hierarchy.json'))
    platform=PlatformScenario.from_record(read(ROOT/'configs/gemmini_platform.json'))
    profile_record=read(args.profile)
    profile=load_cnn_cost_profile(profile_record,memory=memory)
    requirements=PPARequirements(**read(args.requirements)).require_vendor_inputs()
    tool_paths=[Path(__file__),ROOT/'tools/common_dse_evaluator.py',
                ROOT/'tools/gemini_common_dse.py',ROOT/'tools/chiplet_gym_dse.py',
                ROOT/'tools/common_dse_baselines.py',ROOT/'tools/chiplet_gym_common.py',ROOT/'tools/common_dse_genetic.py']
    protocol={'status':'EXPLORATORY_NOT_PAPER_REPRODUCTION', 'model':args.model,
        'input_size':args.input_size,'model_seed':args.model_seed,'search_seed':args.seed,
        'budget_per_method':args.budget,'rl_episode_size':args.rl_episode_size,
        'requirements':asdict(requirements),'cost_profile':profile_record,'memory':memory.record(),
        'platform':asdict(platform),'source_hashes':code_hashes(),
        'tool_hashes':{path.name:file_hash(path) for path in tool_paths},
        'methods':args.methods,
        'limitations':['Gemini and Chiplet-Gym are restricted adaptations, not original full methods',
            'unvalidated analytical PPA','single seed','fixed-profile CNNDesign limits','no hyperparameter tuning']}
    protocol['identity']=digest(protocol)
    save_json(args.out/'protocol.json',protocol)
    workload,numerical=prepare_model(args.model,args.input_size,args.model_seed,args.out)
    def environment(budget):
        return GemminiCNNEnvironment(workload,requirements=requirements,memory=memory,
                                     platform=platform,cost_profile=profile,budget=budget,max_scope=2)
    shared_identity=CommonDSEEvaluator(environment(args.budget),args.budget).snapshot()['environment_identity']
    results=[]
    for method in protocol['methods']:
        begin=time.perf_counter()
        if method == 'reinforce':
            trainer=RACETrainer(seed=args.seed,precision='float64',policy_backend='gemmini_cnn')
            requests=evaluations=cache_hits=0
            winner=winner_design=winner_feedback=None
            episodes=[]
            while requests<args.budget:
                env=environment(min(args.rl_episode_size,args.budget-requests))
                if CommonDSEEvaluator(env,1).snapshot()['environment_identity']!=shared_identity:
                    raise RuntimeError('RACE task identity differs from common task')
                stats=trainer.run_episode(env)
                episodes.append({'stats':stats,'trace':env.trace})
                requests+=env.oracle.requests
                evaluations+=env.oracle.evaluations
                cache_hits+=env.oracle.cache_hits
                candidate=env.scorer.assess(env.best_record)
                if assessment_better(candidate,winner):
                    winner,winner_design,winner_feedback=candidate,env.best_design,env.best_record.model_feedback
                if not env.oracle.requests: raise RuntimeError('RACE made no progress')
            result={'method':method,'seed':args.seed,'result':{'requests':requests,
                'actual_evaluations':evaluations,'cache_hits':cache_hits,
                'best_assessment':winner or requirements.bind().assess(None),
                'best_design':winner_design,'best_feedback':winner_feedback},'episodes':episodes,
                'trainer_config':trainer.config}
        elif method=='chiplet_gym_common_local_edits':
            def local_env(budget):
                candidate=environment(budget)
                if CommonDSEEvaluator(candidate,1).snapshot()['environment_identity']!=shared_identity:
                    raise RuntimeError('Chiplet-Gym local task identity differs from common task')
                return candidate
            local=chiplet_gym_common.run_chiplet_gym_common(local_env,budget=args.budget,seed=args.seed)
            result={'method':method,'seed':args.seed,'result':{'requests':local['oracle_requests'],
                'actual_evaluations':local['oracle_evaluations'],'cache_hits':local['cache_hits'],
                'best_assessment':local['assessment'],'best_design':local['best_design'],
                'best_feedback':local['best_feedback'],'trace':local['trace']},
                'protocol':local['protocol'],
                'limitations':['Experimental local-action adaptation; not native Chiplet-Gym',
                    'episode resets and per-episode caches differ from full-design variant']}
        else:
            evaluator=CommonDSEEvaluator(environment(args.budget),args.budget)
            if evaluator.snapshot()['environment_identity'] != shared_identity:
                raise RuntimeError('method task identity differs from common task')
            if method in ('random','local_random','sa'):
                result=common_dse_baselines.run(evaluator,method=method,seed=args.seed)
            elif method=='genetic':
                result=common_dse_genetic.run(evaluator,seed=args.seed)
            elif method=='bayesopt_gp_ei':
                result=common_dse_bayes.run(evaluator,seed=args.seed)
            elif method=='timeloop_hybrid_restricted':
                result=common_dse_timeloop_restricted.run(evaluator,seed=args.seed)
            elif method=='gemini_common_restricted':
                result=gemini_common_dse.run(evaluator,seed=args.seed)
            elif method=='confuciux_actor_ga_restricted':
                result=confuciux_bridge.run(evaluator,seed=args.seed)
            elif method in ('gamma_mapping_restricted','digamma_joint_restricted'):
                result=gamma_bridge.run(evaluator,seed=args.seed,method=method)
            elif method=='csdse_cooperative_restricted':
                result=csdse_bridge.run(evaluator,seed=args.seed)
            elif method=='moham_nsga_restricted':
                result=moham_bridge.run(evaluator,seed=args.seed)
            else:
                result=chiplet_gym_dse.run(evaluator,seed=args.seed)
        if result['result']['requests'] != args.budget:
            raise RuntimeError('method did not use requested evaluation budget')
        result['common_task_identity']=shared_identity
        result['total_wall_time_s']=time.perf_counter()-begin
        save_json(args.out/f'{method}.json',result)
        results.append(result)
        print(f"{method}: {result['result']['requests']} requests; {result['total_wall_time_s']:.1f}s",flush=True)
    save_json(args.out/'summary.json',{'status':protocol['status'],'protocol_identity':protocol['identity'],
        'common_task_identity':shared_identity,'results':results,'numerical_validation':numerical,
        'wall_time_s':time.perf_counter()-started})


if __name__=='__main__': main()
