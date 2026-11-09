#!/usr/bin/env python3
"""Audit a finished common DSE batch and compute inspectable paired summaries."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    suite=json.loads((args.batch/'suite_protocol.json').read_text())
    rows=json.loads((args.batch/'results.json').read_text())['rows']
    expected={(m,p,s) for m in suite['models'] for p in suite['preferences'] for s in suite['seeds']}
    grouped=defaultdict(dict)
    for row in rows:
        key=(row['model'],row['preference'],row['seed'])
        if key not in expected or row['method'] in grouped[key]: raise ValueError('unexpected or duplicated result')
        grouped[key][row['method']]=row
    if set(grouped)!=expected: raise ValueError('batch has missing trial summaries')
    first_model,first_pref,first_seed=sorted(expected)[0]
    first_protocol=json.loads((args.batch/'runs'/first_model/first_pref/str(first_seed)/'protocol.json').read_text())
    method_names=tuple(first_protocol['methods'])
    if 'reinforce' not in method_names or len(set(method_names))!=len(method_names):
        raise ValueError('missing RACE or duplicate methods')
    validity=defaultdict(Counter); reasons=defaultdict(Counter); trials=[]; method_stats=defaultdict(list)
    hashes={}
    for model,pref,seed in sorted(expected):
        folder=args.batch/'runs'/model/pref/str(seed)
        protocol=json.loads((folder/'protocol.json').read_text())
        summary=json.loads((folder/'summary.json').read_text())
        if (protocol['source_hashes']!=suite['source_hashes'] or protocol['tool_hashes']!=suite['tool_hashes'] or
            protocol['identity']!=summary['protocol_identity'] or tuple(protocol['methods'])!=method_names or
            set(grouped[(model,pref,seed)])!=set(method_names)):
            raise ValueError(f'trial identity or method mismatch: {model} {pref} {seed}')
        for method in method_names:
            path=folder/f'{method}.json'; data=json.loads(path.read_text()); row=grouped[(model,pref,seed)][method]
            if data['common_task_identity']!=summary['common_task_identity'] or data['result']['requests']!=suite['budget']:
                raise ValueError(f'identity/budget mismatch: {path}')
            if data['result']['best_assessment']!=next(x['result']['best_assessment'] for x in summary['results'] if x['method']==method):
                raise ValueError(f'assessment mismatch: {path}')
            assessment=data['result']['best_assessment']
            if (row['feasible']!=assessment['feasible'] or row['weighted_cost']!=assessment['weighted_cost'] or
                row['metrics']!=assessment['metrics'] or row['requests']!=data['result']['requests']):
                raise ValueError(f'aggregated row differs from trial evidence: {path}')
            trace=([t for episode in data['episodes'] for t in episode['trace']]
                   if method=='reinforce' else data['result']['trace'])
            if len(trace)!=suite['budget']: raise ValueError(f'trace length mismatch: {path}')
            validity[(model,method)]['valid']+=sum(bool(t['physical_feasible']) for t in trace)
            validity[(model,method)]['total']+=len(trace)
            for item in trace:
                if not item['physical_feasible']: reasons[(model,method)].update(item.get('violations',[]))
            method_stats[(model,method)].append(row)
            hashes[str(path.relative_to(args.batch))]=sha(path)
        trials.append({'model':model,'preference':pref,'seed':seed,'identity':summary['common_task_identity']})
    output={'batch_protocol_sha256':sha(args.batch/'suite_protocol.json'),'results_sha256':sha(args.batch/'results.json'),
            'trial_count':len(trials),'result_count':len(rows),'trials':trials,'method_results_sha256':hashes,
            'method_summary':[], 'pairwise':[]}
    for (model,method), data in sorted(method_stats.items()):
        feasible=[x for x in data if x['feasible']]
        output['method_summary'].append({'model':model,'method':method,'trials':len(data),
            'feasible_trials':len(feasible),'mean_feasible_cost':statistics.mean(x['weighted_cost'] for x in feasible) if feasible else None,
            'physical_valid_proposals':validity[(model,method)]['valid'],
            'total_proposals':validity[(model,method)]['total'],
            'invalid_reasons':dict(reasons[(model,method)]),
            'mean_seconds':statistics.mean(x['wall_time_s'] for x in data)})
    for model in suite['models']:
        for method in (m for m in method_names if m!='reinforce'):
            win=loss=tie=both=neither=0
            for key,methods in grouped.items():
                if key[0]!=model: continue
                a,b=methods['reinforce'],methods[method]
                if a['feasible'] and not b['feasible']: win+=1
                elif b['feasible'] and not a['feasible']: loss+=1
                elif a['feasible'] and b['feasible']:
                    both+=1
                    if a['weighted_cost']<b['weighted_cost']-1e-12:win+=1
                    elif a['weighted_cost']>b['weighted_cost']+1e-12:loss+=1
                    else:tie+=1
                else: neither+=1
            output['pairwise'].append({'model':model,'reference':method,'wins':win,'losses':loss,
                'ties':tie,'both_feasible':both,'neither_feasible':neither,
                'trials':sum(key[0]==model for key in grouped)})
    if args.out.exists(): raise ValueError('choose a fresh analysis output')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(output,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(f"Audited {len(trials)} trials, {len(rows)} method results: {args.out}")


if __name__=='__main__':main()
