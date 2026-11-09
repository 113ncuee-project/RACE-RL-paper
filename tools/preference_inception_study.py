#!/usr/bin/env python3
"""Run the Inception-ResNet-v1 variant using an isolated extended workload IR.

The three earlier CNNs are reused. The additional models get model-specific
power/latency limits and empirical ranges from a separately accounted random
calibration, then use the same 64-proposal, paired-seed RACE workflow.
"""
import argparse
import importlib.metadata
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import preference_calibrated_study as base
from explore_cnn_requirements import choose_limits
from race_rl.requirements import PPARequirements
from race_rl.serialization import code_hashes, digest, save_json

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'experiments/runs/preference_calibrated_v1_20261104'
ENTRY=ROOT/'tools/gemini_inception_entry.py'
PLOT=ROOT/'tools/plot_preference_inception.py'
EXPLORER=ROOT/'tools/explore_cnn_requirements.py'
BLOCKS=ROOT/'third_party/facenet_pytorch/blocks.py'
ADDITIONAL=('inception_resnet_v1_gemini',)
ALLOWED_SOURCE_DELTAS={'frontend.py','workload.py','tiles.py','accelerators/cnn_quantization.py'}


def calibration_result(folder, cfg, model, seed):
    try:
        d=base.read(folder/'exploration.json');status=base.read(folder/'status.json')
        result=d['result'];trace=result['trace']
        if (d['model']!=model or d['seed']!=seed or d['model_seed']!=cfg['model_seed'] or
            d['input_size']!=cfg['input_size'] or d['budget']!=cfg['calibration_budget'] or
            d['source_hashes']!=cfg['source_hashes'] or d['profile']!=cfg['profile'] or
            d['runner_sha256']!=cfg['explorer_sha256'] or
            result['oracle_requests']!=cfg['calibration_budget'] or
            len(trace)!=cfg['calibration_budget'] or
            status['status'] not in ('CANDIDATE_LIMITS_NOT_DIFFICULTY_VALIDATED','NO_VALID_WITNESS')):
            return None
        return d
    except (OSError,ValueError,TypeError,KeyError):
        return None


def run_calibration(out,cfg,model,seed):
    folder=out/'calibration_trials'/model/str(seed)
    d=calibration_result(folder,cfg,model,seed)
    if d is not None:
        print(f'校準 {model} seed={seed}：已完成',flush=True)
        return d
    if folder.exists() and any(folder.iterdir()):
        folder.rename(folder.with_name(folder.name+'.interrupted.'+str(time.time_ns())))
    cmd=[sys.executable,str(ENTRY),'--entry','explore','--model',model,'--input-size',str(cfg['input_size']),
         '--budget',str(cfg['calibration_budget']),'--seed',str(seed),'--model-seed',str(cfg['model_seed']),
         '--profile',str(out/'profile.json'),'--out',str(folder)]
    print(f'校準 {model} seed={seed}；{cfg["calibration_budget"]} 次提案',flush=True)
    proc=subprocess.run(cmd,cwd=ROOT,check=False)
    d=calibration_result(folder,cfg,model,seed)
    if d is None or (proc.returncode and base.read(folder/'status.json')['status']!='NO_VALID_WITNESS'):
        raise RuntimeError(f'calibration failed or could not be audited: {folder}')
    return d


def make_calibration(cfg, model, trials):
    # Keep the common 49.5 mm² cap used in all three previous CNNs. Select
    # the first calibration seed with a physically valid witness under it.
    witness=None;witness_seed=None;limits=None
    for seed,d in trials:
        eligible=[r for r in d['result']['trace'] if r['physical_feasible'] and r['ppa'] and
                  r['ppa']['area_mm2']<=cfg['area_limit']]
        if eligible:
            selected,witness,_=choose_limits(eligible)
            limits={'max_power_w':selected.max_power_w,'max_latency_s':selected.max_latency_s,
                    'max_area_mm2':cfg['area_limit'],'preference':'balanced','weights':None}
            witness_seed=seed;break
    if not limits:
        return None
    upper=np.array([limits[k] for k in base.LIMIT_KEYS]);candidates={};trace_shas={}
    for seed,d in trials:
        trace_shas[str(seed)]=base.sha(Path(d['_source_path']))
        for row in d['result']['trace']:
            if not row.get('physical_feasible') or not row.get('ppa') or not row.get('design_hash'):continue
            try:v=np.array([row['ppa'][k] for k in base.KEYS],dtype=float)
            except (KeyError,ValueError,TypeError):continue
            if not np.all(np.isfinite(v)&(v>0)&(v<=upper)):continue
            key=row['design_hash']
            if key in candidates and not np.allclose(candidates[key],v,rtol=1e-12,atol=1e-12):
                raise ValueError('same calibration design has inconsistent PPA')
            candidates[key]=v
    if len(candidates)<cfg['minimum_calibration_candidates']:
        return None
    matrix=np.array(list(candidates.values()));lo,hi=np.quantile(matrix,[.05,.95],axis=0);ranges=hi-lo
    if not np.all(np.isfinite(ranges)&(ranges>0)):
        return None
    preferences={}
    for pref in base.PREFS:
        semantic=np.full(3,(1-cfg['strength'])/2) if pref!='balanced' else np.full(3,1/3)
        if pref!='balanced':semantic[base.AXIS[pref]]=cfg['strength']
        effective=semantic*upper/ranges;effective/=effective.sum()
        preferences[pref]={'semantic_range_weights':semantic.tolist(),'effective_limit_weights':effective.tolist()}
    return {'limits':{k:limits[k] for k in base.LIMIT_KEYS},
            'unique_feasible_candidates':len(candidates),'p05':dict(zip(base.KEYS,lo.tolist())),
            'p95':dict(zip(base.KEYS,hi.tolist())),'ranges':dict(zip(base.KEYS,ranges.tolist())),
            'preferences':preferences,'limits_witness_seed':witness_seed,
            'limits_witness_design_hash':witness['design_hash'],'limits_witness_ppa':witness['ppa'],
            'calibration_trace_sha256':trace_shas,
            'calibration_requests':sum(x['result']['oracle_requests'] for _,x in trials),
            'limits_rule':'first seed with valid <=49.5mm² candidate; median-rank witness; power*1.1 latency*1.2; area 49.5mm²'}


def write_results(out,manifest):
    rows=[]
    for model in manifest['models']:
        for seed in manifest['seeds']:
            for pref in base.PREFS:
                folder=out/'runs'/model/str(seed)/pref
                if not base.valid_trial(folder,manifest,model,pref,seed):continue
                d=base.read(folder/'reinforce.json');r=d['result'];a=r['best_assessment']
                rows.append({'model':model,'seed':seed,'preference':pref,'feasible':a['feasible'],
                             'metrics':a['metrics'],'weighted_cost':a['weighted_cost'],
                             'requests':r['requests'],'actual_evaluations':r['actual_evaluations'],
                             'cache_hits':r['cache_hits'],'time_s':d['total_wall_time_s'],
                             'result_sha256':base.sha(folder/'reinforce.json')})
    save_json(out/'results.json',{'manifest_identity':manifest['identity'],'rows':rows})
    subprocess.run([sys.executable,str(PLOT),'--study',str(out)],cwd=ROOT,check=True)
    path=out/'report.md';text=path.read_text(encoding='utf-8')
    text=text.replace('本研究先用舊 seed 341–345 的已稽核、達標候選，按模型計算各 PPA 的 P05–P95 變動範圍。',
                      '新增四模型使用獨立隨機校準種子實際評估、達標且去重的候選，按模型計算各 PPA 的 P05–P95 變動範圍；校準預算另列於 manifest。',1)
    path.write_text(text,encoding='utf-8')
    return len(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--models',choices=ADDITIONAL,nargs='+',default=list(ADDITIONAL))
    p.add_argument('--seeds',type=int,nargs='+',default=[351,352,353])
    p.add_argument('--budget',type=int,default=64)
    p.add_argument('--episode-size',type=int,default=16)
    p.add_argument('--strength',type=float,default=.99)
    p.add_argument('--input-size',type=int,default=75)
    p.add_argument('--calibration-budget',type=int,default=16)
    p.add_argument('--min-calibration-seeds',type=int,default=3)
    p.add_argument('--max-calibration-seeds',type=int,default=10)
    p.add_argument('--minimum-calibration-candidates',type=int,default=8)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if (len(set(a.models))!=len(a.models) or len(set(a.seeds))!=len(a.seeds) or
        not a.models or not a.seeds or a.budget<4 or a.episode_size<1 or a.calibration_budget<1 or
        not 1<=a.min_calibration_seeds<=a.max_calibration_seeds<=20 or
        a.minimum_calibration_candidates<2 or not .6<=a.strength<1 or not 75<=a.input_size<=512 or
        any(s<0 or s>=2**31 or 101<=s<101+a.max_calibration_seeds for s in a.seeds)):
        p.error('invalid or overlapping search/calibration configuration')
    old=base.read(SOURCE/'manifest.json')
    if base.read(SOURCE/'status.json')['status']!='completed':raise ValueError('three-model reference is incomplete')
    current=code_hashes()
    changed={k for k in set(current)|set(old['source_hashes']) if current.get(k)!=old['source_hashes'].get(k)}
    if (not changed <= ALLOWED_SOURCE_DELTAS or
            {x.name:base.sha(x) for x in base.TOOL_PATHS}!=old['tool_hashes']):
        raise ValueError('source or common tools changed outside the isolated operator extension')
    if base.read(ROOT/'configs/ppa_profiles/cnn_45nm_reference.json')!=old['profile']:
        raise ValueError('current cost profile differs from the three-model study')
    limits=[old['calibration'][m]['limits']['max_area_mm2'] for m in old['models']]
    if not all(math.isclose(limits[0],x,rel_tol=1e-12) for x in limits):raise ValueError('reference area limits differ')
    cfg={'kind':'ADDITIONAL_CNN_PREFERENCE_STUDY','models':a.models,'seeds':a.seeds,'budget':a.budget,
         'episode_size':a.episode_size,'strength':a.strength,'input_size':a.input_size,
         'model_seed':old['model_seed'],'profile':old['profile'],'area_limit':limits[0],
         'calibration_budget':a.calibration_budget,'min_calibration_seeds':a.min_calibration_seeds,
         'max_calibration_seeds':a.max_calibration_seeds,
         'minimum_calibration_candidates':a.minimum_calibration_candidates,
         'reference_manifest_sha256':base.sha(SOURCE/'manifest.json'),
         'source_hashes':current,'source_delta_from_reference':sorted(changed),
         'facenet_blocks_sha256':base.sha(BLOCKS),'tool_hashes':{x.name:base.sha(x) for x in base.TOOL_PATHS},
         'entry_sha256':base.sha(ENTRY),'explorer_sha256':base.sha(EXPLORER),
         'all_model_plot_sha256':base.sha(PLOT),'runner_sha256':base.sha(Path(__file__)),
         'python':base.platform.python_version(),'packages':base.versions(),
         'calibration_policy':f'seeds 101 onward, at least {a.min_calibration_seeds}, max {a.max_calibration_seeds}; '
                              f'>={a.minimum_calibration_candidates} unique feasible candidates and positive ranges',
         'limitations':['isolated IR adds explicit Concat and Scale; these add policy observation features',
                        '75x75 is minimum viable input, differs from earlier CNN 64x64 and Gemini 299x299',
                        'Gemini-style v1 stage counts with explicit BN/ReLU/scale; not identical Gemini graph',
                        'exploratory PPA limits and scale from small random calibration',
                        'prior three models used larger RACE-trace calibration',
                        'same seed does not imply identical proposals',
                        'PPA model is analytical and uncalibrated']}
    cfg['identity']=digest(cfg);out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if a.resume:
        if base.read(out/'study_config.json')!=cfg:raise ValueError('resume refused: source or settings changed')
    else:
        if any(out.iterdir()):p.error('output directory not empty; use --resume')
        save_json(out/'study_config.json',cfg)
    save_json(out/'profile.json',cfg['profile'])
    def verify():
        if (code_hashes()!=cfg['source_hashes'] or {x.name:base.sha(x) for x in base.TOOL_PATHS}!=cfg['tool_hashes'] or
            base.sha(ENTRY)!=cfg['entry_sha256'] or base.sha(EXPLORER)!=cfg['explorer_sha256'] or
            base.sha(PLOT)!=cfg['all_model_plot_sha256'] or base.sha(BLOCKS)!=cfg['facenet_blocks_sha256'] or
            base.sha(Path(__file__))!=cfg['runner_sha256']):
            raise RuntimeError('source changed during study')
    try:
        calibration={}
        for model in a.models:
            trials=[];result=None
            for seed in range(101,101+a.max_calibration_seeds):
                verify();save_json(out/'status.json',{'status':'running','stage':'calibration','model':model,'seed':seed})
                d=run_calibration(out,cfg,model,seed)
                d['_source_path']=str(out/'calibration_trials'/model/str(seed)/'exploration.json')
                trials.append((seed,d))
                if len(trials)>=a.min_calibration_seeds:
                    result=make_calibration(cfg,model,trials)
                    if result is not None:break
            if result is None:
                raise RuntimeError(f'{model}: insufficient feasible diverse calibration candidates after {len(trials)} seeds')
            calibration[model]=result
            print(f'{model} 校準完成：{result["unique_feasible_candidates"]} 個可行設計，'
                  f'{result["calibration_requests"]} 次校準提案',flush=True)
        requirements={}
        for model in a.models:
            requirements[model]={}
            for pref in base.PREFS:
                req=dict(calibration[model]['limits'],preference='custom',
                         weights=calibration[model]['preferences'][pref]['effective_limit_weights'])
                req['weights']=list(PPARequirements(**req).require_vendor_inputs().weights)
                requirements[model][pref]=req
        manifest={**cfg,'calibration':calibration,'requirements':requirements,
                  'calibration_source':'independent random proposals (separately accounted)',
                  'calibration_seeds':'101 onwards until predefined diversity threshold'}
        manifest.pop('identity',None);manifest['identity']=digest(manifest)
        if (out/'manifest.json').exists() and base.read(out/'manifest.json')!=manifest:
            raise ValueError('derived calibration/manifest changed; refusing to resume')
        save_json(out/'manifest.json',manifest)
        tasks=[(m,s,pref) for m in a.models for s in a.seeds for pref in base.PREFS]
        for i,(model,seed,pref) in enumerate(tasks,1):
            verify();folder=out/'runs'/model/str(seed)/pref
            if base.valid_trial(folder,manifest,model,pref,seed):
                print(f'[{i}/{len(tasks)}] {model} seed={seed} {pref}：已完成',flush=True);continue
            if folder.exists() and any(folder.iterdir()):
                folder.rename(folder.with_name(folder.name+'.interrupted.'+str(time.time_ns())))
            req=out/'requirements'/model/f'{pref}.json';save_json(req,requirements[model][pref])
            cmd=[sys.executable,str(ENTRY),'--entry','trial','--model',model,'--input-size',str(cfg['input_size']),
                 '--budget',str(cfg['budget']),'--rl-episode-size',str(cfg['episode_size']),
                 '--seed',str(seed),'--model-seed',str(cfg['model_seed']),'--requirements',str(req),
                 '--profile',str(out/'profile.json'),'--out',str(folder),'--methods','reinforce']
            save_json(out/'status.json',{'status':'running','stage':'search','current':i,
                                         'total_trials':len(tasks),'model':model,'seed':seed,'preference':pref})
            print(f'[{i}/{len(tasks)}] {model} seed={seed} {pref}',flush=True)
            subprocess.run(cmd,cwd=ROOT,check=True)
            if not base.valid_trial(folder,manifest,model,pref,seed):raise RuntimeError('trial failed audit')
            count=write_results(out,manifest)
            save_json(out/'status.json',{'status':'running','stage':'search','finished_trials':count,
                                         'total_trials':len(tasks),'last_finished':[model,seed,pref]})
        count=write_results(out,manifest)
        if count!=len(tasks):raise RuntimeError('final trial count mismatch')
        save_json(out/'status.json',{'status':'completed','stage':'search','finished_trials':count,'total_trials':len(tasks)})
        print('全部完成:',out/'report.md',flush=True)
    except BaseException as exc:
        if (out/'manifest.json').exists():write_results(out,base.read(out/'manifest.json'))
        save_json(out/'status.json',{'status':'interrupted_or_failed','reason':str(exc)})
        raise


if __name__=='__main__':main()
