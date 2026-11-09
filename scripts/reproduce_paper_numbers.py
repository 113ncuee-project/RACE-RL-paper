#!/usr/bin/env python3
"""Recompute the manuscript's reported numbers from the extracted raw results."""
import argparse
import glob
import json
from pathlib import Path
import statistics as st
import sys

PKG = 'experiments/paper_sources/RACE-RL_v20261106-0100'
ABLATION = 'experiments/runs/coupled_edit_ablation_20261105'
BASELINES = 'experiments/runs/fresh_all_models_dse_20261104'
ARMS = {'scope1': 'A_scope1', 'scope2': 'B_scope2_dependent', 'scope2_random': 'C_scope2_random_nonadjacent', 'scope4': 'D_scope4_dependent'}
PREFS = ('balanced', 'power', 'performance', 'area')
SELECTED = (('preference_calibrated_v1_20261104', 'resnet18', 351), ('preference_additional_four_20261104', 'resnet34', 351),
            ('preference_additional_four_20261104', 'resnet50', 353), ('preference_calibrated_v1_20261104', 'vgg16', 351),
            ('preference_additional_four_20261104', 'vgg19', 352), ('preference_calibrated_v1_20261104', 'mobilenet_v2', 353),
            ('preference_additional_four_20261104', 'alexnet', 352), ('preference_resnext_20261104', 'resnext50_32x2d_gemini', 353),
            ('preference_inception_20261104', 'inception_resnet_v1_gemini', 351), ('preference_pnas_20261104', 'pnasnet5large_gemini', 352),
            ('preference_transformer_20261104', 'transformer_gemini_shape', 353))
errors = []


def check(label, actual, expected, digits=1):
    # Allow 0.6 of the last reported digit: some manuscript values use Table A.I's rounded PPA.
    ok = abs(actual - expected) <= 0.6 * 10 ** -digits
    if not ok: errors.append(label)
    print(f"  {'OK ' if ok else 'BAD'} {label}: {actual:.{digits + 1}f} (paper {expected})")


def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def fig5(root):
    print('Fig. 5 (Coupled-Edit scopes)')
    rows = {}
    for f in glob.glob(str(root / ABLATION / 'trials/*/*/*/*.json')):
        r = load(f); rows[r['model'], r['arm']] = rows.get((r['model'], r['arm']), []) + [r['best_assessment']['weighted_cost']]
    med = {k: round(st.median(v), 3) for k, v in rows.items()}
    expected = {'resnet18': (103.3, 96.7), 'vgg16': (102.3, 102.3), 'mobilenet_v2': (102.2, 95.9)}
    for model, (s1, s4) in expected.items():
        base = med[model, ARMS['scope2']]
        check(f'{model} scope1 %', 100 * med[model, ARMS['scope1']] / base, s1)
        check(f'{model} scope4 %', 100 * med[model, ARMS['scope4']] / base, s4)
        print(f"      {model} scope2_random %: {100 * med[model, ARMS['scope2_random']] / base:.1f} (control arm, in the data)")


def fig6(root):
    print('Fig. 6 / Table A.I (preference trade-offs)')
    base = root / PKG / 'D_十一模型偏好'; data = {}
    for study, model, seed in SELECTED:
        data[model] = {}
        for pref in PREFS:
            result = load(base / study / 'runs' / model / str(seed) / pref / 'summary.json')['results'][0]['result']['best_assessment']['metrics']
            data[model][pref] = (result['average_power_w'], result['latency_s'], result['area_mm2'])
    expected = {'power': (90.2, 113.6, 84.1), 'performance': (156.8, 93.8, 297.7), 'area': (93.0, 103.5, 84.1)}
    for pref, values in expected.items():
        for i, (name, value) in enumerate(zip(('power', 'latency', 'area'), values)):
            mean = 100 / len(data) * sum(d[pref][i] / d['balanced'][i] for d in data.values())
            check(f'{pref}-first relative {name} %', mean, value)


def vgg(root):
    print('Fig. 7 and frozen-policy comparison (VGG16)')
    base = root / PKG / 'B_VGG16策略訓練'
    a = load(base / 'vgg16/report.json')['assessment']
    check('best balanced cost', a['weighted_cost'], 0.4668, 4)
    check('best power W', a['metrics']['average_power_w'], 0.3673, 4)
    check('best latency s', a['metrics']['latency_s'], 0.4939, 4)
    change = load(base / 'analysis.json')['paired_mean_relative_change']
    for key, value in (('average_power_w', 1.2), ('latency_s', 8.3), ('area_mm2', 9.1)):
        check(f'{key} mean reduction %', -100 * change[key], value)


def fig8(root):
    print('Fig. 8 (search baselines; seeds 411-415)')
    expected = {'resnet18': (0.454, 0.495), 'alexnet': (0.548, 0.635), 'mobilenet_v2': (0.468, 0.466)}
    for model, (race, bo) in expected.items():
        by = {}
        for seed in range(411, 416):
            for item in load(root / BASELINES / 'runs' / model / 'balanced' / str(seed) / 'summary.json')['results']:
                by.setdefault(item['method'], []).append(item['result']['best_assessment'])
        def med(method): return st.median(x['weighted_cost'] for x in by[method] if x['feasible'])
        check(f'{model} RACE-RL', med('reinforce'), race, 3); check(f'{model} BO', med('bayesopt_gp_ei'), bo, 3)
        for method in ('local_random', 'sa', 'gemini_common_restricted'):
            print(f"      {model} {method}: {med(method):.3f} ({sum(x['feasible'] for x in by[method])}/5 feasible)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().root
    for step in (fig5, fig6, vgg, fig8):
        step(root)
    print('\nAll checked values match.' if not errors else f'\nMismatches: {errors}')
    sys.exit(1 if errors else 0)


if __name__ == '__main__':
    main()
