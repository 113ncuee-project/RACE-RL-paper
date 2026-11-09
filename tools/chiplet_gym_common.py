"""Chiplet-Gym PPO + SA adapted to the common RACE CNN evaluator.

Uses Stable-Baselines3 PPO, as in Chiplet-Gym, and the authors' SA acceptance
rule. CNN actions, reward and the finite evaluation budget necessarily differ
from the original. Original pretrained policies cannot be used in this space.
"""
from copy import deepcopy
import hashlib
import importlib.metadata
import os
from pathlib import Path
import sys
import time

import numpy as np

from race_rl.accelerators.cnn_design import coupled_scope, transaction
from race_rl.recording import assessment_better
from race_rl.training import seed_all

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT/'.cache/chiplet-gym-deps'


def decisions(observation, action):
    count = int(action[0])+1
    anchor = observation.anchor_indices[int(action[1])]
    scope = coupled_scope(observation.workload,anchor,observation.max_scope)
    result = [count-1,anchor]
    for i in range(len(scope)):
        profile, residency, layout, target = (int(v) for v in action[2+4*i:6+4*i])
        result.extend((profile,residency,layout,target % count))
    return transaction(observation.workload,observation.current_design,tuple(result),observation.max_scope)


def state(observation):
    graph = observation.graph
    return np.asarray((*graph.global_features[:4], *graph.task_features[:3],
                       *graph.task_features[9:12]), dtype=np.float32)


def run_chiplet_gym_common(make_env, *, budget, seed):
    if budget < 4: raise ValueError('Chiplet-Gym common baseline needs at least four requests')
    if not (DEPS/'stable_baselines3').exists():
        raise ValueError('Chiplet-Gym dependencies missing; see docs/research/chiplet_gym_common.md')
    sys.path.insert(0,str(DEPS))
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/chiplet-gym-mpl'))
    import gymnasium as gym
    from stable_baselines3 import PPO
    seed_all(seed)
    start = time.perf_counter()
    ppo_budget = 2*(budget//4)
    sa_budget = budget-ppo_budget
    rollout_steps = max(2,ppo_budget//2)
    while ppo_budget % rollout_steps: rollout_steps -= 1
    first_env = make_env(2)
    action_sizes = np.array([16,len(first_env.anchor_indices),4,4,2,16,4,4,2,16],dtype=np.int64)
    winner = winner_design = winner_feedback = winner_ppa = None
    requests = evaluations = cache_hits = 0
    trace = []

    def archive(env, phase):
        nonlocal winner,winner_design,winner_feedback,winner_ppa
        candidate = env.scorer.assess(env.best_record)
        if assessment_better(candidate,winner):
            winner, winner_design = candidate, env.best_design
            winner_feedback, winner_ppa = env.best_record.model_feedback, env.best_record.ppa_detail
        item = deepcopy(env.trace[-1])
        item['phase'] = phase
        trace.append(item)

    class CommonEnv(gym.Env):
        def __init__(self):
            self.action_space = gym.spaces.MultiDiscrete(action_sizes)
            self.observation_space = gym.spaces.Box(low=-np.inf,high=np.inf,shape=(10,),dtype=np.float32)
            self.env = None
            self.observation = None

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.env = make_env(2)
            self.observation = self.env.reset()
            return state(self.observation), {}

        def step(self, action):
            nonlocal requests,evaluations,cache_hits
            before_eval,before_cache = self.env.oracle.evaluations,self.env.oracle.cache_hits
            self.observation, reward, done = self.env.step(decisions(self.observation,action))
            requests += 1
            evaluations += self.env.oracle.evaluations-before_eval
            cache_hits += self.env.oracle.cache_hits-before_cache
            archive(self.env,'ppo')
            return state(self.observation),float(reward),False,done,{}

    wrapper = CommonEnv()
    # Original SB3 PPO defaults from paper; rollout/batch scaled to finite budget.
    model = PPO('MlpPolicy',wrapper,learning_rate=3e-4,n_steps=rollout_steps,
        batch_size=min(64,rollout_steps),n_epochs=10,gamma=0.99,gae_lambda=0.95,
        clip_range=0.2,ent_coef=0.1,vf_coef=0.5,seed=seed,device='cpu',
        policy_kwargs={'net_arch':dict(pi=[64,64],vf=[64,64])},verbose=0)
    model.learn(total_timesteps=ppo_budget)
    if requests != ppo_budget: raise RuntimeError('PPO consumed an unexpected number of requests')
    # Original source uses a step size of 10 and accepts worse moves with
    # probability 400/(iteration+1). Adapt the neighbour vector to CNN actions.
    env = make_env(sa_budget)
    observation = env.reset()
    rng = np.random.default_rng(seed)
    current = rng.integers(0,action_sizes)
    accepted_score = None
    for step in range(sa_budget):
        candidate = current if step == 0 else np.clip(current+rng.integers(-10,11,size=10),0,action_sizes-1)
        before_eval,before_cache = env.oracle.evaluations,env.oracle.cache_hits
        old = (env.current_design,env.compute_count,env.current_record,env.current_feedback,env.current_objective)
        observation,_,_ = env.step(decisions(observation,candidate))
        requests += 1
        evaluations += env.oracle.evaluations-before_eval
        cache_hits += env.oracle.cache_hits-before_cache
        archive(env,'sa')
        valid = env.trace[-1]['physical_feasible']
        score = env.current_objective if valid else None
        accepted = valid and (accepted_score is None or score < accepted_score or rng.random() < 400.0/(step+1))
        trace[-1]['accepted'] = accepted
        if accepted:
            current = candidate
            accepted_score = score
        elif valid:
            (env.current_design,env.compute_count,env.current_record,env.current_feedback,env.current_objective) = old
            env._refresh()
            observation = env.observation
            trace[-1]['state_after'].update(design_hash=trace[-1]['state_before']['design_hash'],
                compute_count=env.compute_count, objective=env.current_objective,
                ppa_valid=env.current_record is not None)
        trace[-1]['accepted_design_hash'] = env.current_record.design_hash if env.current_record else None
    if requests != budget: raise RuntimeError('combined PPO + SA exceeded or missed request budget')
    return {'method':'chiplet_gym_common_adapted','seed':seed,'oracle_requests':requests,
        'oracle_evaluations':evaluations,'cache_hits':cache_hits,'wall_time_s':time.perf_counter()-start,
        'assessment':winner or first_env.requirements.bind().assess(None),'best_design':winner_design,
        'best_feedback':winner_feedback,'best_ppa_detail':winner_ppa,'trace':trace,
        'protocol':{'ppo_requests':ppo_budget,'sa_requests':sa_budget,'sb3_version':importlib.metadata.version('stable-baselines3'),
            'ppo_episode_length':2,'n_steps':rollout_steps,'batch_size':min(64,rollout_steps),
            'ppo_parameter_updates':model._n_updates,'n_epochs':10,
            'gamma':0.99,'gae_lambda':0.95,'entropy_coefficient':0.1,'value_coefficient':0.5,
            'sa_temperature_numerator':400,'sa_step_size':10,
            'action_mapping':'MultiDiscrete CNN decisions with modulo target; second layer unused if scope has one',
            'original_pretrained_used':False,'deviations':'One newly trained PPO and one SA; equal request split; common RACE reward and evaluator',
            'source_reference':'https://github.com/KFM135/chiplet-optimizer/tree/6e327348310f927c997efd86b8d3ec2b8986e1f7',
            'numpy_version':np.__version__,'gymnasium_version':gym.__version__,
            'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
