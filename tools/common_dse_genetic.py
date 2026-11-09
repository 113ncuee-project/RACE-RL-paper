"""A reproducible generic genetic search over the shared CNN design space.

This is a baseline inspired by common GA practice, not GAMMA reproduction.
Each initial, invalid, duplicate, or offspring proposal consumes one request.
"""
from dataclasses import replace
import random
import time

from gemini_common_dse import propose
from race_rl.accelerators.cnn_design import CNNDesign, LayerPlan


def run(evaluator, *, seed, population_size=8, tournament_size=3):
    if population_size < 2 or tournament_size < 1:
        raise ValueError('invalid GA population or tournament size')
    rng = random.Random(seed)
    started = time.perf_counter()
    workload = evaluator.environment.workload
    initial = CNNDesign(2, tuple(LayerPlan() for _ in workload.operators))
    population = []

    def parent():
        candidates = rng.choices(population, k=tournament_size)
        return min(candidates, key=lambda entry: entry[0])[1]

    while evaluator.requests < evaluator.budget:
        if evaluator.requests == 0:
            candidate, phase = initial, 'initial'
        elif len(population) < 2:
            candidate, _ = propose(workload, population[0][1] if population else initial, rng)
            phase = 'initial_mutation'
        else:
            first, second = parent(), parent()
            count = rng.choice((first.compute_count, second.compute_count))
            genes = [rng.choice((a, b)) for a, b in zip(first.plans, second.plans)]
            plans = tuple(replace(gene, target=gene.target % count) for gene in genes)
            candidate, _ = propose(workload, CNNDesign(count, plans), rng)
            phase = 'tournament_crossover_mutation'
        item = evaluator.evaluate(candidate)
        item['phase'] = phase
        assessment = item.get('assessment')
        merit = assessment.get('search_merit') if assessment else None
        if merit is not None:
            population.append((merit, candidate))
            population.sort(key=lambda entry: entry[0])
            del population[population_size:]
    return {'method': 'genetic', 'seed': seed, 'wall_time_s': time.perf_counter()-started,
            'result': evaluator.snapshot(),
            'method_limits': ['generic population GA; not GAMMA or DiGamma implementation',
                'shared CNNDesign space and local mutation; population=8, tournament=3',
                'all proposals including initial, invalid and cached count toward budget']}
