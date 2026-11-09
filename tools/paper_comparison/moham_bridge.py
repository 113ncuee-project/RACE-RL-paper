"""Restricted MOHaM-style search using the author's C++ NSGA ranking.

Native MOHaM uses different chiplet structures and genetic operators. This
adapter does not claim to reproduce the full MOHaM algorithm.
"""
from dataclasses import replace
import hashlib
from pathlib import Path
import random
import subprocess
import tempfile
import time

from gemini_common_dse import propose
from race_rl.accelerators.cnn_design import CNNDesign, LayerPlan

ROOT = Path(__file__).resolve().parents[2]


def build_ranker():
    wrapper = ROOT / 'tools/paper_comparison/moham_nsga.cc'
    author_header = ROOT / 'third_party/moham/src/nsga.h'
    key = hashlib.sha256(wrapper.read_bytes() + author_header.read_bytes()).hexdigest()[:16]
    binary = Path(tempfile.gettempdir()) / ('race_moham_nsga_' + key)
    if not binary.is_file():
        subprocess.run(['c++', '-std=c++14', '-O2', str(wrapper), '-o', str(binary)], check=True)
    return binary


def rank(binary, objectives):
    data = str(len(objectives)) + '\n' + ''.join(' '.join(map(str, row)) + '\n' for row in objectives)
    out = subprocess.run([str(binary)], input=data, text=True, capture_output=True, check=True)
    rows = [(int(i), int(r), float(c)) for i, r, c in (line.split() for line in out.stdout.splitlines())]
    if len(rows) != len(objectives) or {row[0] for row in rows} != set(range(len(objectives))):
        raise RuntimeError('author ranker returned invalid population')
    return rows


def run(evaluator, *, seed, population_size=8):
    started = time.perf_counter()
    binary = build_ranker()
    rng = random.Random(seed)
    workload = evaluator.environment.workload
    initial = CNNDesign(2, tuple(LayerPlan() for _ in workload.operators))
    population = []
    def mutate(design):
        proposed, operation = propose(workload, design, rng)
        # The shared operator normally remaps placements after count changes;
        # reject that specific action instead of treating the repair as native.
        if operation == 5 and any(p.target >= proposed.compute_count for p in design.plans):
            return None
        return proposed
    while evaluator.requests < evaluator.budget:
        if not population:
            candidate = initial
            phase = 'initial'
        elif len(population) < 2:
            candidate = mutate(population[0][0])
            phase = 'adapted_mutation'
        else:
            # Evaluate NSGA only on physically valid items, preferring feasible
            # candidates before selecting a Pareto front.
            pool = [p for p in population if p[2]] or population
            scores = rank(binary, [p[1] for p in pool])
            indices = sorted(scores, key=lambda x: (x[1], -x[2], x[0]))
            parent_pool = [pool[v[0]][0] for v in indices[:max(2, len(indices)//2)]]
            a, b = rng.choices(parent_pool, k=2)
            count = rng.choice((a.compute_count, b.compute_count))
            cut = rng.randrange(len(a.plans) + 1)
            genes = a.plans[:cut] + b.plans[cut:]
            # An invalid crossover is still charged to the evaluator rather
            # than silently repairing its targets.
            if any(p.target >= count for p in genes):
                candidate = None
            else:
                candidate = mutate(CNNDesign(count, genes))
            phase = 'author_nsga_rank_adapted_crossover_mutation'
        item = evaluator.evaluate(candidate)
        item['phase'] = phase
        assessment = item.get('assessment') or {}
        metrics = assessment.get('metrics') or {}
        keys = ('average_power_w', 'latency_s', 'area_mm2')
        if candidate is not None and assessment.get('physical_feasible') and all(metrics.get(k) is not None for k in keys):
            population.append((candidate, [float(metrics[k]) for k in keys], bool(assessment.get('feasible'))))
            if len(population) > population_size:
                # Keep feasibility-prioritized non-dominated designs.
                valid = [p for p in population if p[2]]
                invalid = [p for p in population if not p[2]]
                ordered = []
                for group in (valid, invalid):
                    if group:
                        ranked = rank(binary, [p[1] for p in group])
                        ordered.extend(group[r[0]] for r in sorted(ranked, key=lambda x: (x[1], -x[2], x[0])))
                population = ordered[:population_size]
    return {'method': 'moham_nsga_restricted', 'seed': seed,
            'wall_time_s': time.perf_counter() - started, 'result': evaluator.snapshot(),
            'limitations': ['author MOHaM NSGA rank and crowding implementation, adapted genetic operators',
                'power replaces MOHaM energy objective; shared final score uses the same PPA scorer',
                'count mutations that would require target remapping are rejected and charged',
                'no native heterogeneous subaccelerators, multi-workload or native MOHaM mutation',
                'not the complete MOHaM algorithm or a direct paper reproduction']}
