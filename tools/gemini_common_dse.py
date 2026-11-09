"""Gemini-style local SA against action-independent common CNN DSE evaluation.

This is a restricted-space adaptation, not the native Gemini mapping engine.
"""
from dataclasses import replace
import math
import random
import time

from race_rl.accelerators.cnn_design import CNNDesign, LayerPlan, PROFILES


def propose(workload, current, rng):
    material = [i for i,op in enumerate(workload.operators) if op.kind not in ('Flatten','Identity')]
    if not material: raise ValueError('workload has no material operators')
    index = rng.choice(material)
    operation = rng.randrange(6)
    count = current.compute_count
    plans = list(current.plans)
    plan = plans[index]
    if operation == 0:
        plans[index] = replace(plan, **dict(zip(('h','w','c','k'),rng.choice(PROFILES))))
    elif operation == 1:
        plans[index] = replace(plan, target=rng.randrange(count))
    elif operation == 2 and len(material) > 1:
        other = rng.choice([i for i in material if i != index])
        plans[index] = replace(plan,target=plans[other].target)
        plans[other] = replace(plans[other],target=plan.target)
    elif operation == 3:
        plans[index] = replace(plan, spread=not plan.spread)
    elif operation == 4:
        plans[index] = replace(plan, keep_output=bool(rng.randrange(2)), keep_weights=bool(rng.randrange(2)))
    elif operation == 5:
        count = rng.randrange(1,17)
        plans = [replace(p,target=p.target % count) for p in plans]
    return CNNDesign(count,tuple(plans)),operation


def run(evaluator, *, seed):
    rng = random.Random(seed)
    started = time.perf_counter()
    workload = evaluator.environment.workload
    current = CNNDesign(2,tuple(LayerPlan() for _ in workload.operators))
    # Gemini begins from a stripe-based initial scheme. The closest supported
    # RACE scheme uses round-robin placement with the default blocking profile.
    initial = evaluator.evaluate(current)
    initial['phase'] = 'initial'
    merit = initial['assessment']['search_merit'] if initial.get('assessment') else None
    current_score = merit
    temperature = merit/10 if merit is not None and merit > 0 else None
    iteration = 0
    while evaluator.requests < evaluator.budget:
        design,operation = propose(workload,current,rng)
        result = evaluator.evaluate(design)
        assessment = result.get('assessment')
        merit = assessment.get('search_merit') if assessment else None
        score = merit
        accepted = score is not None and (current_score is None or score <= current_score or
                    (temperature is not None and rng.random() < math.exp((current_score-score)/(0.5*temperature))))
        if accepted:
            current,current_score = design,score
            if temperature is None: temperature=max(score/10,1e-12)
        result.update(search_operator=operation, accepted=accepted, temperature=temperature)
        if temperature is not None and iteration % 10 == 0: temperature *= .99
        iteration += 1
    return {'method':'gemini_common_restricted', 'seed':seed, 'wall_time_s':time.perf_counter()-started,
        'result':evaluator.snapshot(),
        'method_limits':['Gemini-inspired SA operations, not native Gemini engine',
            'no native DP layer grouping, per-core permutations, multi-DRAM or intra-core exhaustive search',
            'uses RACE fixed-profile CNNDesign representation and common PPA objective',
            'initial evaluated design and source-inspired temperature/cooling are charged to budget']}
