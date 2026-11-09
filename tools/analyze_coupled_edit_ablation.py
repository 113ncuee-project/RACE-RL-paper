#!/usr/bin/env python3
"""Summarize the coupled-edit ablation (descriptive; paired by model/preference/seed)."""
import argparse
import glob
import json
import math
from pathlib import Path
import random
import statistics

ARMS=('A_scope1','B_scope2_dependent','C_scope2_random_nonadjacent','D_scope4_dependent')
SHORT={'A_scope1':'A 關閉(scope1)','B_scope2_dependent':'B 相依 scope2','C_scope2_random_nonadjacent':'C 隨機不相鄰 scope2','D_scope4_dependent':'D 相依 scope4'}
PAIRS=(('B_scope2_dependent','A_scope1'),('B_scope2_dependent','C_scope2_random_nonadjacent'),
       ('C_scope2_random_nonadjacent','A_scope1'),('D_scope4_dependent','B_scope2_dependent'),('D_scope4_dependent','A_scope1'))


def load(root):
    rows={}
    for f in glob.glob(str(root/'trials/*/*/*/*.json')):
        r=json.load(open(f));rows[(r['model'],r['preference'],r['seed'],r['arm'])]=r
    return rows


def merit(r):
    a=r['best_assessment'];return math.inf if not a or a.get('search_merit') is None else a['search_merit']


def anytime(r,budget):
    """Mean over proposals of best-so-far merit (lower is better); unmeasured counts as 3."""
    best=3.0;acc=0.0
    for c in r['curve'][:budget]:
        if c['merit'] is not None:best=min(best,c['merit'])
        acc+=best
    return acc/budget


def first_feasible(r):
    for i,c in enumerate(r['curve'],1):
        if c['feasible']:return i
    return None


def bootstrap_median(values,n=4000,seed=0):
    if not values:return (None,None)
    rng=random.Random(seed);meds=sorted(statistics.median(rng.choices(values,k=len(values))) for _ in range(n))
    return meds[int(.025*n)],meds[int(.975*n)-1]


def sign_test(w,l):
    n=w+l
    if n==0:return None
    k=min(w,l)
    p=sum(math.comb(n,i) for i in range(k+1))/2**n*2
    return min(1.0,p)


def compare(rows,model,x,y):
    w=l=t=0;ratios=[];only_x=only_y=0
    for (m,p,s,a),rx in rows.items():
        if (model!='all' and m!=model) or a!=x or (m,p,s,y) not in rows:continue
        ry=rows[(m,p,s,y)];ax,ay=rx['best_assessment'] or {},ry['best_assessment'] or {}
        fx,fy=bool(ax.get('feasible')),bool(ay.get('feasible'))
        if fx and fy:ratios.append(math.log(ax['weighted_cost']/ay['weighted_cost']))
        elif fx:only_x+=1
        elif fy:only_y+=1
        mx,my=merit(rx),merit(ry)
        if abs(mx-my)<=1e-12*max(1,abs(my)):t+=1
        elif mx<my:w+=1
        else:l+=1
    lo,hi=bootstrap_median(ratios)
    med=statistics.median(ratios) if ratios else None
    pct=lambda v:None if v is None else 100*(math.exp(v)-1)
    return {'wins':w,'losses':l,'ties':t,'both_feasible':len(ratios),'only_left_feasible':only_x,'only_right_feasible':only_y,
            'median_cost_change_pct':pct(med),'ci95_pct':(pct(lo),pct(hi)),'sign_test_p':sign_test(w,l)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    rows=load(args.root);models=sorted({k[0] for k in rows},key=['resnet18','vgg16','mobilenet_v2'].index)
    out={'per_arm':{},'pairs':{},'decomposition':{}}
    lines=['# Coupled edit 消融實驗結果','',f'共 {len(rows)} 個 trial；每 trial 64 提案（每回合 16，4 次 REINFORCE 更新），reference-v2 限制，seeds 601–620，四偏好。','',
           '加權成本只在達標時有意義；勝負用 search merit（達標優先，再比成本），平手為完全相同 merit。成本變化為「左 / 右 − 1」，負值代表左邊較好。PPA 為未校準分析估計。','']
    lines+=['## 各組總覽','','| 模型 | 組別 | trial | 達標率 | 成本中位數 | anytime merit 均值 | 首次達標（提案序，中位數） | 物理無效提案比例 | 最佳設計顆數中位數 | 每 trial 秒 |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in models:
        for a in ARMS:
            rs=[r for k,r in rows.items() if k[0]==m and k[3]==a]
            if not rs:continue
            feas=[r for r in rs if (r['best_assessment'] or {}).get('feasible')]
            costs=[r['best_assessment']['weighted_cost'] for r in feas]
            ff=[first_feasible(r) for r in rs if first_feasible(r)]
            inv=sum(not c['physical_feasible'] for r in rs for c in r['curve'])/sum(len(r['curve']) for r in rs)
            counts=[r['best_compute_count'] for r in rs if r['best_compute_count']]
            row={'trials':len(rs),'feasible_rate':len(feas)/len(rs),'median_cost':statistics.median(costs) if costs else None,
                 'anytime_mean':statistics.mean(anytime(r,r['budget']) for r in rs),'first_feasible_median':statistics.median(ff) if ff else None,
                 'invalid_rate':inv,'median_count':statistics.median(counts) if counts else None,'mean_wall_s':statistics.mean(r['wall_time_s'] for r in rs)}
            out['per_arm'][f'{m}/{a}']=row
            fmt=lambda v,d=4:'—' if v is None else f'{v:.{d}f}'
            lines.append(f"| {m} | {SHORT[a]} | {row['trials']} | {100*row['feasible_rate']:.0f}% | {fmt(row['median_cost'])} | {fmt(row['anytime_mean'])} | {fmt(row['first_feasible_median'],1)} | {100*inv:.1f}% | {fmt(row['median_count'],1)} | {row['mean_wall_s']:.0f} |")
    lines+=['','## 配對比較','','| 模型 | 左 vs 右 | 左勝 | 右勝 | 平 | 雙方達標 | 只有左達標 | 只有右達標 | 成本變化中位數 | 95% bootstrap CI | 符號檢定 p |','|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|']
    for m in models+['all']:
        for x,y in PAIRS:
            c=compare(rows,m,x,y)
            if c['wins']+c['losses']+c['ties']==0:continue
            out['pairs'][f'{m}/{x}/{y}']=c
            f=lambda v:'—' if v is None else f'{v:+.2f}%'
            ci=c['ci95_pct'];p=c['sign_test_p']
            lines.append(f"| {m} | {SHORT[x]} vs {SHORT[y]} | {c['wins']} | {c['losses']} | {c['ties']} | {c['both_feasible']} | {c['only_left_feasible']} | {c['only_right_feasible']} | {f(c['median_cost_change_pct'])} | [{f(ci[0])}, {f(ci[1])}] | {'—' if p is None else f'{p:.3g}'} |")
    lines+=['','## 拆解測試（只看改善最佳值的多層提案，排除第一個量測）','',
            '對每個讓回合最佳值變好的多層提案，額外（不計預算、不回饋策略）評估：只改顆數、只改 anchor、只改 partner。「只有一起改才改善」= 三個半提案都沒有優於改善前的最佳值。交互作用 = 聯合增益 − anchor 增益 − partner 增益（log objective）。','',
            '| 模型 | 組別 | 改善提案數 | 只有一起改才改善 | 交互作用 > 0 比例 | 交互作用中位數 |','|---|---|---:|---:|---:|---:|']
    for m in models:
        for a in ARMS[1:]:
            ds=[d for k,r in rows.items() if k[0]==m and k[3]==a for d in r['decomposition'] if d['best_before']<1.0]
            if not ds:continue
            jo=sum(d['joint_only'] for d in ds);pos=sum(d['interaction']>1e-9 for d in ds)
            med=statistics.median(d['interaction'] for d in ds)
            out['decomposition'][f'{m}/{a}']={'improving_proposals':len(ds),'joint_only':jo,'positive_interaction':pos,'median_interaction':med}
            lines.append(f"| {m} | {SHORT[a]} | {len(ds)} | {jo} ({100*jo/len(ds):.1f}%) | {100*pos/len(ds):.1f}% | {med:.3g} |")
    (args.root/'analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':main()
