"""Chiplet-Gym PPO + SA, adapted to full designs and the common evaluator.

Uses the paper's PPO implementation and upstream SA acceptance rule. Workload,
actions, reward, finite budget and pretrained policies differ from the paper.
"""
import os
from pathlib import Path
import sys
import time

import numpy as np
from race_rl.accelerators.cnn_design import CNNDesign, LayerPlan, PROFILES
from race_rl.training import seed_all

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT/'.cache/chiplet-gym-deps'


def domain(workload):
    indices = [i for i,op in enumerate(workload.operators) if op.kind not in ('Flatten','Identity')]
    sizes = [16]
    for _ in indices: sizes.extend((4,4,2,16))
    return indices,np.asarray(sizes,dtype=np.int64)


def decode(workload, indices, action):
    count = int(action[0])+1
    plans = [LayerPlan() for _ in workload.operators]
    for i, op_index in enumerate(indices):
        profile,residency,layout,target = map(int,action[1+4*i:5+4*i])
        plans[op_index] = LayerPlan(*PROFILES[profile], target % count, bool(layout),
                                    bool(residency & 1), bool(residency & 2))
    return CNNDesign(count,tuple(plans))


def run(evaluator, *, seed):
    if evaluator.budget < 4:
        raise ValueError('at least four requests are required')
    if not (DEPS/'stable_baselines3').exists():
        raise ValueError('Chiplet-Gym dependencies missing; see docs/research/chiplet_gym_common.md')
    sys.path.insert(0,str(DEPS))
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/chiplet-gym-mpl'))
    import gymnasium as gym
    from stable_baselines3 import PPO
    seed_all(seed)
    started = time.perf_counter()
    workload = evaluator.environment.workload
    indices,sizes = domain(workload)
    ppo_budget = 2*(evaluator.budget//4)
    sa_budget = evaluator.budget-ppo_budget
    # Ensure an update can influence later choices when the budget permits it.
    rollout_steps = max(2,ppo_budget//2)
    while ppo_budget % rollout_steps: rollout_steps -= 1
    weights = evaluator.environment.requirements.normalized_weights
    limits = evaluator.environment.requirements.limits

    def observation(result=None, design=None):
        assessment = result.get('assessment') if result else None
        metrics = assessment['metrics'] if assessment and assessment['physical_feasible'] else {}
        ratios = [min(10,float(metrics[k])/limit) if metrics.get(k) else 0 for k,limit in zip(
            ('average_power_w','latency_s','area_mm2'),limits)]
        return np.asarray([*ratios,*weights, float(bool(assessment and assessment['feasible'])),
            (design.compute_count/16) if design else 0,
            (evaluator.budget-evaluator.requests)/evaluator.budget,
            float(assessment is not None and assessment['physical_feasible'])],dtype=np.float32)

    class SharedEnv(gym.Env):
        def __init__(self):
            self.action_space = gym.spaces.MultiDiscrete(sizes)
            self.observation_space = gym.spaces.Box(-np.inf,np.inf,shape=(10,),dtype=np.float32)
            self.steps = 0
        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.steps = 0
            return observation(),{}
        def step(self, action):
            design = decode(workload,indices,action)
            result = evaluator.evaluate(design)
            result['phase'] = 'ppo'
            assessment = result.get('assessment')
            merit = assessment.get('search_merit') if assessment else None
            reward = 0.0 if merit is None else 1.0/(1.0+merit)
            self.steps += 1
            return observation(result,design),reward,False,self.steps >= 2,{}

    model = PPO('MlpPolicy',SharedEnv(),learning_rate=3e-4,n_steps=rollout_steps,
        batch_size=min(64,rollout_steps),n_epochs=10,gamma=.99,gae_lambda=.95,
        clip_range=.2,ent_coef=.1,vf_coef=.5,seed=seed,device='cpu',
        policy_kwargs={'net_arch':dict(pi=[64,64],vf=[64,64])},verbose=0)
    model.learn(total_timesteps=ppo_budget)
    if evaluator.requests != ppo_budget: raise RuntimeError('PPO request budget mismatch')
    rng = np.random.default_rng(seed)
    current = rng.integers(0,sizes)
    current_score = None
    for step in range(sa_budget):
        candidate = current if step == 0 else np.clip(current+rng.integers(-10,11,size=len(sizes)),0,sizes-1)
        result = evaluator.evaluate(decode(workload,indices,candidate))
        result['phase'] = 'sa'
        assessment = result.get('assessment')
        score = assessment.get('search_merit') if assessment else None
        accepted = score is not None and (current_score is None or score < current_score or rng.random() < 400.0/(step+1))
        result['accepted'] = bool(accepted)
        if accepted: current,current_score = candidate,score
    return {'method':'chiplet_gym_common_full_design', 'seed':seed, 'wall_time_s':time.perf_counter()-started,
        'result':evaluator.snapshot(), 'protocol':{'ppo_requests':ppo_budget,'sa_requests':sa_budget,
            'ppo_episode_length':2,'n_steps':rollout_steps,'batch_size':min(64,rollout_steps),
            'ppo_parameter_updates':model._n_updates,'n_epochs':10,
            'sa_neighbour_step':10,'sa_acceptance_numerator':400,'original_pretrained_used':False,
            'action_encoding':'one count plus profile/residency/layout/target for every material operator; target modulo count',
            'limitations':['common CNN design and PPA replace original architecture/packaging model',
                'one newly trained PPO and one SA instead of original ensemble',
                'rollout and budget scaled; reward and action space adapted']}}
