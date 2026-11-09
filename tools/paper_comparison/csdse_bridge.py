"""Restricted CSDSE-style cooperative search using author policy and buffer.

The original complete CSDSE uses additional agents, compression and MAESTRO.
"""
import importlib.util
from dataclasses import replace
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.distributions import Categorical

from race_rl.accelerators.cnn_design import CNNDesign, LayerPlan, PROFILES

ROOT = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decode_actions(actions, n):
    if len(actions) != 1 + 4 * n or not 0 <= actions[0] < 16:
        raise ValueError('invalid action sequence')
    count = actions[0] + 1
    plans = []
    for i in range(n):
        profile, residency, spread, target = actions[1 + i * 4: 5 + i * 4]
        if not (0 <= profile < 4 and 0 <= residency < 4 and spread in (0, 1)
                and 0 <= target < count):
            return None
        plans.append(LayerPlan(*PROFILES[profile], target, bool(spread),
                               bool(residency & 1), bool(residency & 2)))
    return CNNDesign(count, tuple(plans))


def action_distribution(policy, state, step, prior_actions):
    """Mask nonexistent chiplet targets before sampling and during replay."""
    probs = policy(torch.as_tensor(state, dtype=torch.float32), step)
    if step == 5:
        count = int(prior_actions[0]) + 1
        mask = torch.arange(probs.shape[-1], device=probs.device) < count
        probs = probs * mask
        probs = probs / probs.sum()
    return Categorical(probs)


def run(evaluator, *, seed):
    source = ROOT / 'third_party/csdse/util'
    mlp = _load('race_author_csdse_mlp', source / 'mlp.py')
    source_buffer = _load('race_author_csdse_buffer', source / 'sample_buffer.py')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    n = len(evaluator.environment.workload.operators)
    # The shared RACE search edits a small layer scope. A single-layer edit
    # here avoids independently sampling hundreds of unrelated placements.
    scales = [16, n, 4, 4, 2, 16]
    policies = [mlp.mlp_policyfunction(8, scales) for _ in range(2)]
    optimizers = [torch.optim.Adam(p.parameters(), lr=.001) for p in policies]
    buffer = source_buffer.buffer(distance_type='Hamming', distance_threshold=0,
                                  adaptive=False, agent_cnt=2)
    started = time.perf_counter()
    parent = CNNDesign(2, tuple(LayerPlan() for _ in range(n)))
    initial = evaluator.evaluate(parent)
    initial.update(phase='charged_common_initial', agent=None)
    first_assessment = initial.get('assessment') or {}
    best_merit = first_assessment.get('search_merit')
    while evaluator.requests < evaluator.budget:
        agent = (evaluator.requests - 1) % 2
        policy = policies[agent]
        states, actions = [], []
        for step, scale in enumerate(scales):
            obs = np.array([step / len(scales), n / 100, agent,
                            evaluator.requests / evaluator.budget,
                            (actions[0] + 1) / 16 if actions else 0,
                            (sum(actions) / (len(actions) * 16)) if actions else 0,
                            0, 0], dtype=np.float32)
            with torch.no_grad():
                dist = action_distribution(policy, obs, step, actions)
                action = int(dist.sample())
            states.append(obs.tolist())
            actions.append(action)
        count, anchor, profile, residency, spread, target = actions
        count += 1
        plans = list(parent.plans)
        if any(p.target >= count for p in plans):
            candidate = None
        else:
            plans[anchor] = LayerPlan(*PROFILES[profile], target, bool(spread),
                                      bool(residency & 1), bool(residency & 2))
            candidate = CNNDesign(count, tuple(plans))
        item = evaluator.evaluate(candidate)
        assessment = item.get('assessment') or {}
        merit = assessment.get('search_merit')
        if candidate is not None and merit is not None and (best_merit is None or merit < best_merit):
            parent, best_merit = candidate, merit
        reward = 1.0 / (1.0 + max(0., float(merit))) if merit is not None and np.isfinite(merit) else 0.
        # Author buffer expects normalized obs and higher reward is better.
        sample = {'obs': [a / max(1, scale - 1) for a, scale in zip(actions, scales)],
                  'action_list': actions, 'states': states, 'reward': reward}
        buffer.update(sample, agent)
        mode = 'max' if agent == 0 else 'random'
        replay = buffer.acquire(mode)
        loss = torch.tensor(0.)
        for step, (state, action) in enumerate(zip(replay['states'], replay['action_list'])):
            dist = action_distribution(policy, state, step, replay['action_list'][:step])
            discount = .999 ** (len(scales) - step - 1)
            loss = loss - dist.log_prob(torch.tensor(action)) * (replay['reward'] * discount) - .1 * dist.entropy()
        loss = loss / len(scales)
        optimizers[agent].zero_grad()
        loss.backward()
        optimizers[agent].step()
        item.update(phase='author_policy_shared_author_buffer', agent=agent,
                    native_buffer_mode=mode, native_buffer_size=buffer.size(),
                    action_sequence=actions, reward=reward)
    return {'method': 'csdse_cooperative_restricted', 'seed': seed,
            'wall_time_s': time.perf_counter() - started, 'result': evaluator.snapshot(),
            'limitations': ['author MLP policy and sample buffer, two agents and adapted training loop',
                'not full CSDSE: no adaptive agent organization, space compression, or MAESTRO',
                'charged default seed and one-layer edits from current best, not original global search',
                'observations, actions, reward and search space adapted to the shared CNN evaluator',
                'target choices are masked to the sampled chiplet count before sampling and during replay']}
