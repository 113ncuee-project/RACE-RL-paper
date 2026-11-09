#!/usr/bin/env python3
"""Summarize every completed paired seed for individual and shared searches."""
import argparse
import math
import statistics
from pathlib import Path

import preference_calibrated_study as base
from race_rl.serialization import digest, save_json

PREFS=base.PREFS
KEYS=base.KEYS
NAMES={'resnet18':'ResNet18','mobilenet_v2':'MobileNetV2','vgg16':'VGG16'}


def summarize(study):
    manifest=base.read(study/'manifest.json')
    data=base.read(study/'results.json')
    idx={(r['model'],r['seed'],r['preference']):r for r in data['rows']}
    out=[]
    for model in manifest['models']:
        bypref={p:[] for p in PREFS[1:]};four_different=complete=0
        for seed in manifest['seeds']:
            group={p:idx.get((model,seed,p)) for p in PREFS}
            if not all(group.values()) or not all(x['feasible'] for x in group.values()):
                continue
            complete+=1
            designs=[]
            for p in PREFS:
                f=study/'runs'/model/str(seed)/p/'reinforce.json'
                designs.append(digest(base.read(f)['result']['best_design']))
            four_different+=len(set(designs))==4
            for p in PREFS[1:]:
                k=KEYS[base.AXIS[p]];current=group[p]['metrics'][k];b=group['balanced']['metrics'][k]
                bypref[p].append(100*(current/b-1))
        out.append({'model':model,'paired_seeds':complete,'total_seeds':len(manifest['seeds']),
                    'four_different_designs':four_different,
                    'preferences':{p:{'n':len(values),'lower':sum(x < -1e-9 for x in values),
                                       'same':sum(abs(x)<=1e-9 for x in values),
                                       'higher':sum(x > 1e-9 for x in values),
                                       'median_change_pct':statistics.median(values) if values else None,
                                       'all_changes_pct':values} for p,values in bypref.items()}})
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);a=p.parse_args()
    study=a.study.resolve();shared=study/'shared_archive';manifest=base.read(study/'manifest.json')
    individual=summarize(study);archive=summarize(shared)
    outcome={'individual':individual,'shared_archive':archive,
             'individual_results_sha256':base.sha(study/'results.json'),
             'shared_results_sha256':base.sha(shared/'results.json')}
    save_json(study/'direction_summary.json',outcome)
    lines=['# 四偏好方向實驗摘要','',
           f"每個模型與 seed 有四次各 {manifest['budget']} 提案的 RACE 搜尋。"
           f"共用候選結果可使用四次搜尋的結果，合計 {4*manifest['budget']} 提案。"
           '原版各自搜尋與共用候選的數字分開列出。','',
           '數字為目標比同 seed 平衡結果低／相同／高；中位數為所有共同達標 seed 的目標相對變化，負值表示降低。'
           '只用四偏好都達標的 seed，未達標需回看完整紀錄。','']
    for label,items in [('各自搜尋',individual),('四偏好共用候選',archive)]:
        lines += [f'## {label}','',
                  '| 模型 | 四偏好達標 seed | 四個不同設計 | 功耗較低／相同／較高；中位數 | 效能延遲較低／相同／較高；中位數 | 面積較低／相同／較高；中位數 |',
                  '|---|---:|---:|---|---|---|']
        for x in items:
            cells=[]
            for p in PREFS[1:]:
                q=x['preferences'][p];med='—' if q['median_change_pct'] is None else f"{q['median_change_pct']:+.2f}%"
                cells.append(f"{q['lower']}/{q['same']}/{q['higher']}；{med}")
            lines.append(f"| {NAMES[x['model']]} | {x['paired_seeds']}/{x['total_seeds']} | "
                         f"{x['four_different_designs']}/{x['paired_seeds']} | {' | '.join(cells)} |")
        lines.append('')
    lines += ['各圖為依明示規則挑出的展示案例，並非平均結果；全部 seed 及選取規則見 '
              '`shared_archive/showcase/cases.md`。共同候選是既有 PPA 的離線重選，不能稱為新的 RL 訓練。',
              '原始硬上限、硬體與 PPA 成本模型不變；平衡定義採舊可行候選的 P05–P95 變動範圍，'
              '和舊版 1/3 權重直接除以限制不同。此為探索性結果、樣本少，PPA 未經晶片校準。','']
    (study/'direction_summary.md').write_text('\n'.join(lines),encoding='utf-8')
    print('完成:',study/'direction_summary.md')


if __name__=='__main__':main()
