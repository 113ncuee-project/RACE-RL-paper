"""GAMMA/DiGamma source operators on a disclosed restricted CNN representation.

This is not the native MAESTRO mapping problem or a reproduction. Each native
per-layer genome is projected to the closest available CNN blocking profile.
"""
import copy
import importlib.util
import math
from pathlib import Path
import random
import time

import numpy as np

from race_rl.accelerators.cnn_design import CNNDesign, LayerPlan, PROFILES

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'third_party/gamma/src/GAMMA/gamma.py'


def _author_module():
    spec = importlib.util.spec_from_file_location('race_author_gamma', SOURCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def project_profile(genome):
    """Documented nearest log-distance projection of Y/X/C/K tile sizes."""
    values = {str(d): int(v) for d, v in genome[1:7]}
    if set(values) != {'K', 'C', 'Y', 'X', 'R', 'S'}:
        raise ValueError('native GAMMA genome must have six unique dimensions')
    target = (values['Y'], values['X'], values['C'], values['K'])
    if any(v < 1 for v in target):
        raise ValueError('invalid native tile size')
    return min(range(len(PROFILES)), key=lambda i: (
        sum(math.log(a / b) ** 2 for a, b in zip(target, PROFILES[i])), i))


def to_design(chromosomes, count):
    return CNNDesign(count, tuple(LayerPlan(*PROFILES[project_profile(g)],
                                           target=0, spread=True,
                                           keep_output=False, keep_weights=False)
                                  for g in chromosomes))


def run(evaluator, *, seed, method, fixed_count=2, population_size=8):
    if method not in ('gamma_mapping_restricted', 'digamma_joint_restricted'):
        raise ValueError('invalid method')
    if not 1 <= fixed_count <= 16 or population_size < 2:
        raise ValueError('invalid common-space configuration')
    random.seed(seed)
    np.random.seed(seed)
    author = _author_module()
    # The author code expects one convolution's K/C/Y/X/R/S/T bounds.  This
    # common envelope supports all four shared profiles, not native shapes.
    engine = author.GAMMA([64, 64, 16, 16, 3, 3, 1], num_pe=16, pe_limit=16)
    n_layers = len(evaluator.environment.workload.operators)
    start = time.perf_counter()
    population = []

    def fresh():
        chromosomes = [engine.create_genome() for _ in range(n_layers)]
        count = fixed_count if method == 'gamma_mapping_restricted' else random.randint(1, 16)
        return chromosomes, count

    while evaluator.requests < evaluator.budget:
        if len(population) < population_size:
            genes, count = fresh()
            phase = 'author_genome_initialization'
        else:
            parents = sorted(population, key=lambda x: x[0])[:max(2, population_size // 2)]
            first, second = random.sample(parents, 2)
            genes = []
            for i in range(n_layers):
                offspring = [None, None]
                engine.crossover_tile([first[1][i], second[1][i]], offspring, alpha=.57)
                engine.mutate_tile(offspring, num_mu_loc=3, alpha=.53, range_alpha=.53)
                genes.append(copy.deepcopy(random.choice(offspring)))
            if method == 'gamma_mapping_restricted':
                count = fixed_count
            else:
                # Source single-level PE mutation applied to the shared count.
                holder = [copy.deepcopy(first[1][0])]
                holder[0][0][1] = first[2]
                engine.mutate_pe(holder, alpha=.5)
                count = int(holder[0][0][1])
            phase = 'author_crossover_tile_mutation'
        design = to_design(genes, count)
        item = evaluator.evaluate(design)
        native_genomes = [[[str(d), int(v)] for d, v in genome] for genome in genes]
        item.update(phase=phase, native_compute_count=count, native_genomes=native_genomes,
                    native_genome_sha256=__import__('hashlib').sha256(
                        repr(genes).encode()).hexdigest(),
                    projection='nearest_log_Y_X_C_K_to_four_profiles')
        assessment = item.get('assessment') or {}
        merit = assessment.get('search_merit')
        if merit is not None and np.isfinite(merit):
            population.append((float(merit), genes, count))
            population.sort(key=lambda x: x[0])
            del population[population_size:]
    return {'method': method, 'seed': seed, 'wall_time_s': time.perf_counter() - start,
            'result': evaluator.snapshot(),
            'limitations': ['author GAMMA create_genome, crossover_tile and mutate_tile; DiGamma also uses mutate_pe',
                'network-level composition, selection and four-profile projection are adaptations',
                'loop order, cluster, kernel tiling, residency and placement are not faithfully represented',
                'GAMMA fixed at two compute chiplets; DiGamma count varies from one to sixteen',
                'not the complete native GAMMA or DiGamma algorithm or a fair full-space comparison']}
