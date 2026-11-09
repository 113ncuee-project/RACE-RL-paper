#!/usr/bin/env python3
"""Explore preference weights with frozen, per-model empirical PPA scales.

Calibration uses previously audited traces only. New trials run the unchanged
RACE trainer, hard limits and PPA model; this is not a baseline comparison.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from race_rl.requirements import PPARequirements
from race_rl.serialization import code_hashes, digest, save_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiments/runs/preference_tradeoff_validation_20261031/reference_v2'
TRIAL = ROOT / 'tools/common_dse_trial.py'
PLOTTER = ROOT / 'tools/plot_preference_calibrated.py'
MODELS = ('resnet18', 'mobilenet_v2', 'vgg16')
PREFS = ('balanced', 'power', 'performance', 'area')
KEYS = ('average_power_w', 'latency_s', 'area_mm2')
LIMIT_KEYS = ('max_power_w', 'max_latency_s', 'max_area_mm2')
AXIS = {'power': 0, 'performance': 1, 'area': 2}
TOOL_PATHS = [TRIAL, ROOT / 'tools/common_dse_evaluator.py', ROOT / 'tools/gemini_common_dse.py',
              ROOT / 'tools/chiplet_gym_dse.py', ROOT / 'tools/common_dse_baselines.py',
              ROOT / 'tools/chiplet_gym_common.py', ROOT / 'tools/common_dse_genetic.py', PLOTTER]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def versions():
    result = {}
    for name in ('numpy', 'torch', 'torchvision', 'matplotlib'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def calibrate(strength):
    suite = read(SOURCE / 'suite_protocol.json')
    audit = read(SOURCE / 'analysis.json')
    if (suite['methods'] != ['reinforce'] or suite['budget'] != 64 or
            audit['trial_count'] != 60 or audit['result_count'] != 60 or
            set(suite['models']) != set(MODELS) or set(suite['preferences']) != set(PREFS)):
        raise ValueError('calibration source is incomplete or has different scope')
    if sha(SOURCE / 'suite_protocol.json') != audit['batch_protocol_sha256']:
        raise ValueError('calibration protocol hash mismatch')
    provenance = {}
    calibration = {}
    for model in MODELS:
        limits = np.array([suite['limits'][model][k] for k in LIMIT_KEYS], dtype=float)
        candidates = {}
        for pref in PREFS:
            for seed in suite['seeds']:
                rel = Path('runs') / model / pref / str(seed) / 'reinforce.json'
                path = SOURCE / rel
                file_sha = sha(path)
                if file_sha != audit['method_results_sha256'][str(rel)]:
                    raise ValueError(f'calibration trace hash mismatch: {rel}')
                provenance[str(rel)] = file_sha
                data = read(path)
                trace = [t for episode in data['episodes'] for t in episode['trace']]
                if len(trace) != suite['budget']:
                    raise ValueError(f'calibration trace request count mismatch: {rel}')
                for t in trace:
                    if not t.get('physical_feasible') or not t.get('ppa') or not t.get('design_hash'):
                        continue
                    try:
                        values = np.array([t['ppa'][k] for k in KEYS], dtype=float)
                    except (KeyError, TypeError, ValueError):
                        continue
                    if not np.all(np.isfinite(values) & (values > 0) & (values <= limits)):
                        continue
                    key = t['design_hash']
                    if key in candidates and not np.allclose(candidates[key], values, rtol=1e-12, atol=1e-12):
                        raise ValueError('same candidate hash has inconsistent PPA')
                    candidates[key] = values
        if len(candidates) < 20:
            raise ValueError(f'not enough feasible calibration designs for {model}')
        data = np.array(list(candidates.values()))
        lower, upper = np.quantile(data, [0.05, 0.95], axis=0)
        ranges = upper - lower
        if not np.all(np.isfinite(ranges) & (ranges > 0)):
            raise ValueError(f'constant or invalid calibration range for {model}')
        preferences = {}
        for pref in PREFS:
            target = np.full(3, (1-strength)/2) if pref != 'balanced' else np.full(3, 1/3)
            if pref != 'balanced':
                target[AXIS[pref]] = strength
            # PPAScorer already divides by limits. Convert w_i/range_i to
            # equivalent normalized limit weights without changing hard limits.
            raw = target * limits / ranges
            effective = raw / raw.sum()
            preferences[pref] = {'semantic_range_weights': target.tolist(),
                                 'effective_limit_weights': effective.tolist()}
        calibration[model] = {'limits': dict(zip(LIMIT_KEYS, limits.tolist())),
                              'unique_feasible_candidates': len(candidates),
                              'p05': dict(zip(KEYS, lower.tolist())),
                              'p95': dict(zip(KEYS, upper.tolist())),
                              'ranges': dict(zip(KEYS, ranges.tolist())),
                              'preferences': preferences}
    return suite, calibration, provenance


def valid_trial(folder, manifest, model, pref, seed):
    try:
        protocol = read(folder / 'protocol.json')
        data = read(folder / 'reinforce.json')
        summary = read(folder / 'summary.json')
        expected_req = manifest['requirements'][model][pref]
        actual_req = protocol['requirements']
        req_matches = (set(actual_req) == set(expected_req) and
                       all(actual_req[k] == expected_req[k] for k in expected_req if k != 'weights') and
                       len(actual_req['weights']) == len(expected_req['weights']) and
                       all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-14)
                           for a, b in zip(actual_req['weights'], expected_req['weights'])))
        basic = (protocol['model'] == model and protocol['search_seed'] == seed and
                protocol['input_size'] == manifest['input_size'] and
                protocol['model_seed'] == manifest['model_seed'] and
                protocol['budget_per_method'] == manifest['budget'] and
                protocol['rl_episode_size'] == manifest['episode_size'] and
                req_matches and
                protocol['cost_profile'] == manifest['profile'] and
                protocol['source_hashes'] == manifest['source_hashes'] and
                protocol['methods'] == ['reinforce'] and
                data['result']['requests'] == manifest['budget'] and
                len([t for ep in data['episodes'] for t in ep['trace']]) == manifest['budget'] and
                data['common_task_identity'] == summary['common_task_identity'] and
                summary['protocol_identity'] == protocol['identity'])
        if not basic or any(manifest['tool_hashes'].get(k) != v for k, v in protocol['tool_hashes'].items()):
            return False
        req = PPARequirements(**expected_req)
        weights = req.normalized_weights
        limits = req.limits
        scores = []
        for ep in data['episodes']:
            for t in ep['trace']:
                if not t.get('physical_feasible') or not t.get('ppa'):
                    continue
                v = [t['ppa'][k] for k in KEYS]
                if any(x is None or not math.isfinite(x) or x <= 0 for x in v):
                    return False
                cost = math.fsum(w*x/l for w, x, l in zip(weights, v, limits))
                excess = [max(0., x/l-1) for x, l in zip(v, limits)]
                scores.append(cost/(1+cost) if not any(excess) else 2+math.fsum(excess)+cost/(1+cost))
        assessment = data['result']['best_assessment']
        if not scores:
            return not assessment['physical_feasible']
        v = [assessment['metrics'][k] for k in KEYS]
        cost = math.fsum(w*x/l for w, x, l in zip(weights, v, limits))
        return (assessment['physical_feasible'] and
                assessment['feasible'] == all(x <= l for x, l in zip(v, limits)) and
                math.isclose(assessment['weighted_cost'], cost, rel_tol=1e-10, abs_tol=1e-12) and
                math.isclose(assessment['search_merit'], min(scores), rel_tol=1e-10, abs_tol=1e-12))
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        return False


def save_progress(out, manifest):
    rows = []
    for model in manifest['models']:
        for seed in manifest['seeds']:
            for pref in PREFS:
                folder = out / 'runs' / model / str(seed) / pref
                if not valid_trial(folder, manifest, model, pref, seed):
                    continue
                data = read(folder / 'reinforce.json')
                result = data['result']
                assessment = result['best_assessment']
                rows.append({'model': model, 'seed': seed, 'preference': pref,
                             'feasible': assessment['feasible'], 'metrics': assessment['metrics'],
                             'weighted_cost': assessment['weighted_cost'],
                             'requests': result['requests'], 'actual_evaluations': result['actual_evaluations'],
                             'cache_hits': result['cache_hits'], 'time_s': data['total_wall_time_s'],
                             'result_sha256': sha(folder / 'reinforce.json')})
    save_json(out / 'results.json', {'manifest_identity': manifest['identity'], 'rows': rows})
    subprocess.run([sys.executable, str(PLOTTER), '--study', str(out)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[351, 352, 353])
    parser.add_argument('--models', choices=MODELS, nargs='+', default=list(MODELS))
    parser.add_argument('--budget', type=int, default=64)
    parser.add_argument('--episode-size', type=int, default=16)
    parser.add_argument('--strength', type=float, default=.99)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--import-from', type=Path,
                        help='revalidate and copy compatible completed trials from a previous output')
    args = parser.parse_args()
    if (args.budget < 4 or args.episode_size < 1 or not .6 <= args.strength < 1 or
            len(set(args.seeds)) != len(args.seeds) or len(set(args.models)) != len(args.models) or
            any(s < 0 or s >= 2**31 for s in args.seeds)):
        parser.error('invalid budget, weight or duplicate/invalid seeds')
    source, calibration, provenance = calibrate(args.strength)
    if set(args.seeds) & set(source['seeds']):
        parser.error('new trial seeds must differ from calibration seeds')
    requirements = {}
    for model in args.models:
        requirements[model] = {}
        for pref in PREFS:
            weights = calibration[model]['preferences'][pref]['effective_limit_weights']
            req = dict(source['limits'][model], preference='custom', weights=weights)
            obj = PPARequirements(**req).require_vendor_inputs()
            req['weights'] = list(obj.weights)
            requirements[model][pref] = req
    manifest = {'kind': 'EXPLORATORY_RANGE_NORMALIZED_PREFERENCE_STUDY',
                'models': args.models, 'seeds': args.seeds, 'budget': args.budget,
                'episode_size': args.episode_size, 'strength': args.strength,
                'input_size': source['input_size'], 'model_seed': source['model_seed'],
                'profile': source['profile'], 'requirements': requirements, 'calibration': calibration,
                'calibration_source': str(SOURCE.relative_to(ROOT)),
                'calibration_protocol_sha256': sha(SOURCE / 'suite_protocol.json'),
                'calibration_audit_sha256': sha(SOURCE / 'analysis.json'),
                'calibration_trace_sha256': provenance,
                'source_hashes': code_hashes(), 'tool_hashes': {p.name: sha(p) for p in TOOL_PATHS},
                'runner_sha256': sha(Path(__file__)), 'python': platform.python_version(),
                'packages': versions(), 'calibration_seeds': source['seeds'],
                'limitations': ['exploratory calibration and strength chosen after prior results',
                                'PPA is an uncalibrated analytical estimate',
                                'same seed does not imply identical proposals',
                                'range normalization changes the meaning of balanced compared with older runs']}
    old_out = args.import_from.resolve() if args.import_from else None
    if old_out:
        old_manifest = read(old_out / 'manifest.json')
        if digest({k: v for k, v in old_manifest.items() if k != 'identity'}) != old_manifest['identity']:
            raise ValueError('import source manifest identity is invalid')
        check_keys = ('models', 'seeds', 'budget', 'episode_size', 'strength', 'input_size', 'model_seed',
                      'profile', 'requirements', 'calibration', 'calibration_protocol_sha256',
                      'calibration_audit_sha256', 'calibration_trace_sha256', 'source_hashes', 'tool_hashes',
                      'python', 'packages')
        if any(old_manifest.get(k) != manifest.get(k) for k in check_keys):
            raise ValueError('import source is not the same experiment/configuration')
        manifest['import_source'] = str(old_out)
        manifest['import_manifest_sha256'] = sha(old_out / 'manifest.json')
    manifest['identity'] = digest(manifest)
    out = args.out.resolve()
    if args.resume:
        if read(out / 'manifest.json') != manifest:
            raise ValueError('resume refused: sources, environment, or experiment configuration changed')
    else:
        if out.exists() and any(out.iterdir()):
            parser.error('output directory is not empty; use --resume or a new directory')
        out.mkdir(parents=True, exist_ok=True)
        save_json(out / 'manifest.json', manifest)
        if old_out:
            copied = []
            for model in args.models:
                for seed in args.seeds:
                    for pref in PREFS:
                        previous = old_out / 'runs' / model / str(seed) / pref
                        if valid_trial(previous, manifest, model, pref, seed):
                            destination = out / 'runs' / model / str(seed) / pref
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copytree(previous, destination)
                            copied.append([model, seed, pref, sha(previous / 'reinforce.json')])
            save_json(out / 'import_audit.json', {'import_source': str(old_out), 'validated_copies': copied,
                                                  'reason': 'tolerate sub-ulp weight renormalization in audit'})
            print(f'已重新稽核並匯入 {len(copied)} 組既有完整結果', flush=True)
    save_json(out / 'profile.json', manifest['profile'])
    tasks = [(m, s, p) for m in args.models for s in args.seeds for p in PREFS]
    save_json(out / 'status.json', {'status': 'running', 'finished_trials': sum(
        valid_trial(out / 'runs' / m / str(s) / p, manifest, m, p, s) for m, s, p in tasks),
        'total_trials': len(tasks)})
    try:
        for i, (model, seed, pref) in enumerate(tasks, 1):
            if (code_hashes() != manifest['source_hashes'] or
                    {p.name: sha(p) for p in TOOL_PATHS} != manifest['tool_hashes'] or
                    sha(Path(__file__)) != manifest['runner_sha256']):
                raise RuntimeError('source changed; study stopped')
            folder = out / 'runs' / model / str(seed) / pref
            if valid_trial(folder, manifest, model, pref, seed):
                print(f'[{i}/{len(tasks)}] {model} {seed} {pref}: 已完成', flush=True)
                continue
            if folder.exists() and any(folder.iterdir()):
                folder.rename(folder.with_name(folder.name + '.interrupted.' + str(time.time_ns())))
            req_path = out / 'requirements' / model / f'{pref}.json'
            save_json(req_path, manifest['requirements'][model][pref])
            print(f'[{i}/{len(tasks)}] {model} seed={seed} {pref}', flush=True)
            cmd = [sys.executable, str(TRIAL), '--model', model, '--input-size', str(source['input_size']),
                   '--seed', str(seed), '--model-seed', str(source['model_seed']), '--budget', str(args.budget),
                   '--rl-episode-size', str(args.episode_size), '--requirements', str(req_path),
                   '--profile', str(out / 'profile.json'), '--out', str(folder), '--methods', 'reinforce']
            subprocess.run(cmd, cwd=ROOT, check=True)
            if not valid_trial(folder, manifest, model, pref, seed):
                raise RuntimeError('completed trial failed protocol/trace audit')
            save_progress(out, manifest)
            save_json(out / 'status.json', {'status': 'running', 'finished_trials': len(read(out / 'results.json')['rows']),
                                            'total_trials': len(tasks), 'last_finished': [model, seed, pref]})
        save_json(out / 'status.json', {'status': 'completed', 'finished_trials': len(tasks), 'total_trials': len(tasks)})
    except BaseException:
        save_progress(out, manifest)
        count = len(read(out / 'results.json')['rows'])
        save_json(out / 'status.json', {'status': 'interrupted_or_failed', 'finished_trials': count, 'total_trials': len(tasks)})
        raise
    print(f'完成: {out / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
