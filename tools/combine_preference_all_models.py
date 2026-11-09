#!/usr/bin/env python3
"""Combine audited prior and additional CNN preference runs without re-running."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import preference_calibrated_study as base
from race_rl.frontend import SUPPORTED_MODELS
from race_rl.serialization import digest,save_json

ROOT=Path(__file__).resolve().parents[1]


def link(source,target):
    if not source.exists():return
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_symlink():
        if target.resolve()!=source.resolve():raise ValueError(f'conflicting symlink: {target}')
        return
    if target.exists():raise ValueError(f'refusing to overwrite existing path: {target}')
    target.symlink_to(os.path.relpath(source,target.parent),target_is_directory=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--additional',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();old=args.reference.resolve();new=args.additional.resolve();out=args.out.resolve()
    a,b=base.read(old/'manifest.json'),base.read(new/'manifest.json')
    for key in ('profile','model_seed','input_size','budget','episode_size','strength','seeds','source_hashes','tool_hashes'):
        if a[key]!=b[key]:raise ValueError(f'incompatible old/new studies: {key}')
    if set(a['models']) & set(b['models']):raise ValueError('old and new model lists overlap')
    models=[m for m in SUPPORTED_MODELS if m in set(a['models']+b['models'])]
    if len(models)!=len(a['models'])+len(b['models']):raise ValueError('unsupported or duplicate model')
    existing=base.read(old/'results.json')['rows'];extra=base.read(new/'results.json')['rows'] if (new/'results.json').exists() else []
    out.mkdir(parents=True,exist_ok=True)
    manifest=dict(a)
    for k in ('identity','import_source','import_manifest_sha256'):manifest.pop(k,None)
    manifest.update(kind='COMBINED_EXPLORATORY_SEVEN_CNN_PREFERENCE_STUDY',models=models,
                    calibration={**a['calibration'],**b['calibration']},
                    requirements={**a['requirements'],**b['requirements']},
                    source_studies={'reference':str(old),'additional':str(new)},
                    source_manifests={'reference':base.sha(old/'manifest.json'),'additional':base.sha(new/'manifest.json')},
                    limitations=list(a['limitations'])+['additional four CNNs use smaller independent random calibration pools'])
    manifest['identity']=digest(manifest);save_json(out/'manifest.json',manifest)
    for model in models:
        source=old if model in a['models'] else new
        link(source/'runs'/model,out/'runs'/model)
    rows=existing+extra
    save_json(out/'results.json',{'manifest_identity':manifest['identity'],'rows':rows})
    subprocess.run([sys.executable,str(ROOT/'tools/plot_preference_all_models.py'),'--study',str(out)],cwd=ROOT,check=True)
    report=out/'report.md';text=report.read_text(encoding='utf-8')
    text=text.replace('本研究先用舊 seed 341–345 的已稽核、達標候選，按模型計算各 PPA 的 P05–P95 變動範圍。',
        '原三模型使用舊 seed 341–345 的 RACE 候選校準；新增四模型使用獨立隨機搜尋校準。'
        '各模型都使用其可行候選的 P05–P95 變動範圍；兩批校準樣本數及來源不同。',1)
    report.write_text(text,encoding='utf-8')
    shared=out/'shared_archive';shared.mkdir(exist_ok=True)
    shared_manifest=dict(manifest);shared_manifest.pop('identity',None)
    shared_manifest.update(kind='COMBINED_SHARED_ARCHIVE_PREFERENCE_STUDY',
                           budget_per_original_trial=a['budget'],budget=4*a['budget'])
    shared_manifest['identity']=digest(shared_manifest);save_json(shared/'manifest.json',shared_manifest)
    shared_rows=[]
    for source in (old,new):
        f=source/'shared_archive'/'results.json'
        if f.exists():shared_rows+=base.read(f)['rows']
    for model in models:
        source=old if model in a['models'] else new
        link(source/'shared_archive'/'runs'/model,shared/'runs'/model)
    save_json(shared/'results.json',{'manifest_identity':shared_manifest['identity'],'rows':shared_rows})
    subprocess.run([sys.executable,str(ROOT/'tools/plot_preference_all_models.py'),'--study',str(shared)],cwd=ROOT,check=True)
    sp=shared/'report.md';st=sp.read_text(encoding='utf-8')
    st=st.replace('本研究先用舊 seed 341–345 的已稽核、達標候選，按模型計算各 PPA 的 P05–P95 變動範圍。',
        '原三模型使用舊 RACE 候選校準；新增四模型使用獨立隨機候選校準，樣本數不同。',1)
    sp.write_text(f'# 四偏好共用候選研究\n\n每個模型與 seed 合併四次各 {a["budget"]} 次提案，'
                  f'共 {4*a["budget"]} 次提案；各偏好從共享候選中離線重選。\n\n'+st,encoding='utf-8')
    subprocess.run([sys.executable,str(ROOT/'tools/render_preference_all_models.py'),'--study',str(out)],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'tools/render_preference_all_models.py'),'--study',str(shared)],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'tools/summarize_preference_all_models.py'),'--study',str(out)],cwd=ROOT,check=True)
    completed=[m for m in models if sum(r['model']==m for r in rows)==len(a['seeds'])*len(base.PREFS)]
    status={'status':'completed' if len(completed)==len(models) else 'running',
            'finished_trials':len(rows),'total_trials':len(models)*len(a['seeds'])*len(base.PREFS),
            'completed_models':completed,'pending_models':[m for m in models if m not in completed]}
    save_json(out/'status.json',status)
    print(f'合併結果 {len(rows)}/{status["total_trials"]} 組：{out/"direction_summary.md"}',flush=True)


if __name__=='__main__':main()
