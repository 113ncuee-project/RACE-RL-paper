"""Restricted ConfuciuX actor and GA crossover on the common CNN evaluator.

Not the native ConfuciuX environment: see docs/research/five_paper_bridge_20261104.md.
"""
from dataclasses import replace
import importlib.util
from pathlib import Path
import random
import time

import numpy as np
import torch

from race_rl.accelerators.cnn_design import CNNDesign, LayerPlan, PROFILES

ROOT = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _encode(design):
    rows = [[design.compute_count, 0]]
    for p in design.plans:
        residency = int(p.keep_output) + 2 * int(p.keep_weights)
        rows.append([PROFILES.index((p.h, p.w, p.c, p.k)),
                     p.target * 8 + residency * 2 + int(p.spread)])
    return np.asarray(rows, dtype=np.int64)


def _decode(rows):
    count = int(rows[0, 0])
    plans = []
    for profile, flags in rows[1:]:
        flags = int(flags)
        target, bits = divmod(flags, 8)
        if not 0 <= target < count:
            return None
        residency, spread = divmod(bits, 2)
        plans.append(LayerPlan(*PROFILES[int(profile)], target, bool(spread),
                               bool(residency & 1), bool(residency & 2)))
    return CNNDesign(count, tuple(plans))


def run(evaluator, *, seed):
    """Use author Agent and crossover, with a charged 50/50 phase split."""
    if evaluator.budget < 4:
        raise ValueError('ConfuciuX bridge needs at least four proposals')
    source = ROOT / 'third_party/confuciux/src/ConfX'
    rl = _load('race_author_confx_rl', source / 'rl_confx.py')
    ga = _load('race_author_confx_ga', source / 'ga_confx.py')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = rl.Agent(7, 2, 2, 12, seed)
    agent.set_fitness('latency')  # selects the normal native learn branch
    rng = random.Random(seed)
    workload = evaluator.environment.workload
    n = len(workload.operators)
    start = time.perf_counter()
    pool = []
    actor_budget = evaluator.budget // 2

    def evaluate(candidate, phase):
        item = evaluator.evaluate(candidate)
        item['phase'] = phase
        assessment = item.get('assessment') or {}
        merit = assessment.get('search_merit')
        return float(merit) if merit is not None and np.isfinite(merit) else 1e6

    for proposal in range(actor_budget):
        agent.reset()
        logs = []

        def sample(index, stage):
            # Seven adapted features; original MAESTRO layer features are not equivalent.
            state = np.array([index / max(1, n), stage / 3, proposal / actor_budget,
                              n / 100, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
            action, log_prob = agent.act(state, {})
            logs.append((state, action, log_prob))
            return [int(x) for x in action.detach().cpu().tolist()]

        a, b = sample(0, 0)
        count = (12 * a + b) % 16 + 1
        plans = []
        for i in range(n):
            a, b = sample(i, 1)
            profile, residency = a % 4, b % 4
            c, d = sample(i, 2)
            spread, target = bool(c % 2), (12 * c + d) % count
            plans.append(LayerPlan(*PROFILES[profile], target, spread,
                                   bool(residency & 1), bool(residency & 2)))
        candidate = CNNDesign(count, tuple(plans))
        merit = evaluate(candidate, 'author_actor')
        pool.append((merit, candidate))
        for i, (state, action, log_prob) in enumerate(logs):
            last = i == len(logs) - 1
            agent.step(state, action, log_prob, -merit if last else 0.0,
                       state, last, last, None, {})

    while evaluator.requests < evaluator.budget:
        selected = sorted(pool, key=lambda x: x[0])[:min(4, len(pool))]
        # Source crossover exchanges the two-column genes across layer halves.
        parents = np.stack([_encode(d) for _, d in selected])
        offspring = ga.crossover(parents, 1, parents.shape[1])[0].astype(np.int64)
        idx = rng.randrange(1, len(offspring))
        if rng.randrange(2):
            offspring[idx, 0] = rng.randrange(len(PROFILES))
        else:
            offspring[idx, 1] = rng.randrange(int(offspring[0, 0]) * 8)
        candidate = _decode(offspring)
        merit = evaluate(candidate, 'author_crossover_adapted_mutation')
        if candidate is not None:
            pool.append((merit, candidate))
    return {'method': 'confuciux_actor_ga_restricted', 'seed': seed,
            'wall_time_s': time.perf_counter() - start,
            'result': evaluator.snapshot(),
            'limitations': ['author Agent and GA crossover; adapted observations, reward and mutation',
                            'not the original full ConfuciuX environment or native GA operators',
                            'two phases split budget 50/50; whole design scored once per proposal']}
