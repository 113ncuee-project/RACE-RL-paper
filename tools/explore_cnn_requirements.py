#!/usr/bin/env python3
"""Explore model-specific limits from independently sampled designs.

The resulting limits are candidates, not established medium-difficulty tasks.
Use separate seeds for tuning and final comparisons and publish this record.
"""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import statistics
import time

from race_rl.accelerators.cnn_baselines import run_cnn_baseline
from race_rl.accelerators.cnn_environment import GemminiCNNEnvironment
from race_rl.accelerators.cnn_ppa import load_cnn_cost_profile
from race_rl.accelerators.cnn_search import prepare_model, read
from race_rl.accelerators.gemmini_model import PlatformScenario
from race_rl.accelerators.memory_hierarchy import MemoryHierarchy
from race_rl.requirements import METRICS, PPARequirements
from race_rl.serialization import code_hashes, save_json

ROOT = Path(__file__).resolve().parents[1]


def choose_limits(trace):
    rows = [row for row in trace if row['physical_feasible'] and row['ppa']]
    if not rows:
        raise ValueError('no physically valid PPA sample; no limit can be derived')
    medians = {key: statistics.median(row['ppa'][key] for row in rows) for key in METRICS}
    # Predeclared median-rank witness, with deterministic tie-breaking.
    rows.sort(key=lambda row: (sum(row['ppa'][key] / medians[key] for key in METRICS), row['design_hash']))
    witness = rows[(len(rows)-1)//2]
    ppa = witness['ppa']
    limits = PPARequirements(ppa['average_power_w']*1.1, ppa['latency_s']*1.2,
                             ppa['area_mm2']*1.1)
    return limits, witness, len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('resnet18', 'mobilenet_v2', 'vgg16'), required=True)
    parser.add_argument('--input-size', type=int, default=64)
    parser.add_argument('--budget', type=int, default=16)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--model-seed', type=int, default=7)
    parser.add_argument('--profile', type=Path, default=ROOT/'configs/ppa_profiles/cnn_45nm_reference.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.budget < 1 or not 32 <= args.input_size <= 512 or any(not 0 <= s < 2**31 for s in (args.seed, args.model_seed)):
        parser.error('invalid budget, input size or seed')
    if args.out.exists() and any(args.out.iterdir()): parser.error('output directory must be empty')
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    memory = MemoryHierarchy.from_record(read(ROOT/'configs/memory_hierarchy.json'))
    profile = load_cnn_cost_profile(read(args.profile), memory=memory)
    platform = PlatformScenario.from_record(read(ROOT/'configs/gemmini_platform.json'))
    workload, numerical = prepare_model(args.model, args.input_size, args.model_seed, args.out)
    env = GemminiCNNEnvironment(workload, memory=memory, platform=platform, cost_profile=profile,
                                requirements=PPARequirements(), budget=args.budget)
    result = run_cnn_baseline(env, 'random', seed=args.seed)
    save_json(args.out/'exploration.json', {'model':args.model, 'input_size':args.input_size,
        'seed':args.seed, 'model_seed':args.model_seed, 'budget':args.budget, 'profile':read(args.profile), 'result':result,
        'numerical_validation':numerical, 'source_hashes':code_hashes(),
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'total_wall_time_s':time.perf_counter()-started})
    try:
        limits, witness, count = choose_limits(result['trace'])
    except ValueError as error:
        save_json(args.out/'status.json', {'status':'NO_VALID_WITNESS', 'reason':str(error)})
        raise
    save_json(args.out/'requirements.json', asdict(limits))
    save_json(args.out/'status.json', {'status':'CANDIDATE_LIMITS_NOT_DIFFICULTY_VALIDATED',
        'rule':'median rank by sum of per-metric median-normalized PPA; witness margins 1.1/1.2/1.1',
        'witness_hash':witness['design_hash'], 'witness_ppa':witness['ppa'],
        'valid_samples':count, 'requests':result['oracle_requests']})
    print(f"{args.model}: {count}/{args.budget} physically valid samples; candidate limits saved", flush=True)


if __name__ == '__main__':
    main()
