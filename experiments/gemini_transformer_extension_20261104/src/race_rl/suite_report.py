"""Reports use measured sessions only; candidate search gains are not policy gains."""
import gzip
import html
import json
from pathlib import Path
import sqlite3
import statistics
from contextlib import closing
from .serialization import save_json


def read_sessions(folder):
    path = folder/'history.sqlite'
    if not path.exists(): return []
    with closing(sqlite3.connect(f'{path.resolve().as_uri()}?mode=ro', uri=True)) as db:
        return [{'phase': p, 'seed': s, 'episode_number': e, **json.loads(v)}
                for p, s, e, v in db.execute('SELECT phase,seed,episode,stats FROM episodes ORDER BY rowid')]


def policy_comparisons(sessions):
    before = {s['seed']: s for s in sessions if s['phase'] == 'baseline'}
    rows = []
    for s in sessions:
        if s['phase'] != 'final' or s['seed'] not in before: continue
        a, b = before[s['seed']], s
        ja, jb = a['best_objective'], b['best_objective']
        rows.append({'evaluation_seed': s['seed'], 'before': a['assessment'], 'after': b['assessment'],
                     'before_objective': ja, 'after_objective': jb,
                     'objective_relative_improvement': (ja-jb)/ja if ja else None,
                     'before_compute': a['selected_compute_count'], 'after_compute': b['selected_compute_count'],
                     'before_steps': a['steps'], 'after_steps': b['steps'],
                     'frozen_parameters_unchanged': not a['policy_changed'] and not b['policy_changed']})
    return rows


def policy_summary(paired):
    if not paired:return None
    before=[p['before_objective'] for p in paired]
    after=[p['after_objective'] for p in paired]
    return {'pairs':len(paired),'before_mean_J':statistics.mean(before),'after_mean_J':statistics.mean(after),
            'after_std_J':statistics.stdev(after) if len(after)>1 else None,
            'paired_difference':__import__('race_rl.statistics',fromlist=['paired_summary']).paired_summary(before,after)
                if len(before)>1 else None,
            'scope':'Conditional on ONE trained policy/seed; evaluation seeds are not independent training replicates.'}


def search_comparisons(sessions):
    rows=[]
    for phase in ('train','random','sa'):
        values=[s for s in sessions if s['phase']==phase]
        if not values:continue
        best=min(values,key=lambda s:s['best_objective'])
        rows.append({'method':'reinforce' if phase=='train' else phase,
                     'completed_episodes':len(values),'requests':sum(s['steps'] for s in values),
                     'new_evaluations':sum(s['oracle_evaluations'] for s in values),
                     'cache_hits':sum(s['cache_hits'] for s in values),
                     'best_J':best['best_objective'],'best_assessment':best['assessment'],
                     'compliant_episodes':sum(s['assessment']['feasible'] for s in values),
                     'wall_s':sum(s['wall_s'] for s in values),
                     'parameter_updates':sum(s['parameter_updates'] for s in values)})
    return rows


def plot_training(folder, train, sessions=()):
    if not train: return 'NO_COMPLETED_TRAINING_EPISODES'
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return 'MATPLOTLIB_NOT_INSTALLED; numerical curves remain available'
    best, values = 1.0, []
    for s in train:
        best = min(best, s['best_objective'])
        values.append(best)
    xs = [s['episode_number'] for s in train]
    fig, axes = plt.subplots(3,1, figsize=(8,9), constrained_layout=True)
    axes[0].plot(xs, [s['best_objective'] for s in train], label='Episode best objective', alpha=.65,marker='o',markersize=3)
    axes[0].step(xs, values, where='post', label='Best found so far',marker='o',markersize=3)
    axes[0].set(ylabel='Feasibility-first objective J (lower is better)')
    axes[0].legend()
    tests=[s for s in sessions if s['phase'] in ('baseline','periodic','final')]
    groups={}
    for s in tests:groups.setdefault(s.get('training_episode_at_measurement',0 if s['phase']=='baseline' else len(train)),[]).append(s['best_objective'])
    ts=sorted(groups)
    axes[1].errorbar(ts,[statistics.mean(groups[t]) for t in ts],
        yerr=[statistics.stdev(groups[t]) if len(groups[t])>1 else 0 for t in ts],fmt='o-',capsize=3)
    axes[1].set(ylabel='Frozen policy mean best J (+/- sample SD)')
    axes[2].plot(xs, [s['wall_s'] for s in train], label='Episode wall time',marker='o',markersize=3)
    axes[2].set(xlabel='Training episode', ylabel='Time (s)')
    for ax in axes: ax.grid(alpha=.25)
    fig.suptitle(folder.name+' | REINFORCE, 64px | analytical PPA, not calibrated silicon')
    fig.savefig(folder/'training_curve.svg')
    fig.savefig(folder/'training_curve.png', dpi=180)
    plt.close(fig)
    return 'PASSED'


def export_model(folder, recorder, protocol, model, elapsed):
    sessions = recorder.sessions()
    train = [s for s in sessions if s['phase'] == 'train']
    winner = recorder.get('winner')
    save_json(folder/'training_curve.json', train)
    paired = policy_comparisons(sessions)
    save_json(folder/'policy_comparison.json', paired)
    if winner:
        with gzip.open(folder/'selected_design.json.gz', 'wt', encoding='utf-8', compresslevel=6) as stream:
            json.dump(winner, stream, allow_nan=False, separators=(',', ':'))
    for method in protocol['config'].get('comparisons',[]):
        comparison_winner=recorder.get(f'winner_{method}')
        if comparison_winner:
            with gzip.open(folder/f'{method}_selected_design.json.gz','wt',encoding='utf-8',compresslevel=6) as stream:
                json.dump(comparison_winner,stream,allow_nan=False,separators=(',',':'))
    checkpoint = recorder.checkpoint()
    if checkpoint:
        (folder/'checkpoint.pt').write_bytes(checkpoint.getvalue())
    result = {'model': model, 'smoke_only': protocol['smoke_only'],
              'training_episodes': len(train), 'training_steps': sum(s['steps'] for s in train),
              'policy_updates': sum(s['parameter_updates'] for s in train),
              'training_evaluations': sum(s['oracle_evaluations'] for s in train),
              'training_cache_hits': sum(s['cache_hits'] for s in train),
              'training_interaction_and_update_time_s': sum(s['wall_s'] for s in train),
              'evaluation_time_s': sum(s['wall_s'] for s in sessions if s['phase'] in ('baseline','periodic','final')),
              'comparison_time_s':sum(s['wall_s'] for s in sessions if s['phase'] in ('random','sa')),
              'current_process_wall_s': elapsed,
              'selected_assessment': winner['assessment'] if winner else None,
              'selected_compute_count': len(winner['hardware']['nodes']) if winner else None,
              'policy_comparisons': paired,'policy_summary':policy_summary(paired),
              'search_comparisons':search_comparisons(sessions),
              'periodic_evaluations':[s for s in sessions if s['phase']=='periodic'],
              'plot_status': plot_training(folder, train, sessions),
              'learning_claim': 'Paired frozen-policy rollout diagnostic only; one training seed, '
                                'no proof of convergence, generalization or algorithm superiority.'}
    save_json(folder/'result.json', result)


def value(x):
    return '—' if x is None else f'{x:.6g}'


def export_suite(run, protocol):
    models = []
    timings_path = run/'model_times.json'
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else []
    for name in protocol['config']['models']:
        path = run/name/'result.json'
        if path.exists():
            models.append(json.loads(path.read_text(encoding='utf-8')))
        else:
            sessions = read_sessions(run/name)
            train = [s for s in sessions if s['phase'] == 'train']
            models.append({'model': name, 'training_episodes': len(train),
                           'training_steps': sum(s['steps'] for s in train),
                           'selected_assessment': None, 'selected_compute_count': None,
                           'policy_comparisons': policy_comparisons(sessions),
                           'partial': True})
    for row in models:
        row['model_process_wall_total_s'] = sum(t['wall_s'] for t in timings if t['model'] == row['model'])
    save_json(run/'results.json', {'protocol_identity': protocol['identity'], 'models': models,
                                  'model_attempt_times':timings,
                                  'smoke_only': protocol['smoke_only']})
    table, details = [], []
    for row in models:
        name = row['model']
        a = row.get('selected_assessment') or {}
        metrics = a.get('metrics') or {}
        cells = [name, str(row['training_episodes']), str(row['training_steps']),
                 str(row.get('selected_compute_count') or '—'),
                 '是' if a.get('feasible') else ('否' if a else '未完成'),
                 value(metrics.get('average_power_w')), value(metrics.get('latency_s')),
                 value(metrics.get('area_mm2')), value(a.get('weighted_cost')),
                 value(row.get('training_interaction_and_update_time_s',0)/60),
                 value(row['model_process_wall_total_s']/60)]
        table.append('<tr>'+''.join(f'<td>{html.escape(c)}</td>' for c in cells)+'</tr>')
        comparisons = []
        for pair in row.get('policy_comparisons', []):
            pct = pair['objective_relative_improvement']
            comparisons.append(f'<li>測試 seed {pair["evaluation_seed"]}：'
                f'J {pair["before_objective"]:.6g} → {pair["after_objective"]:.6g}；'
                f'相對改善 {100*pct:.2f}%（負值表示變差）。前後各 {pair["before_steps"]} 次互動，參數保持凍結。</li>')
            comparisons.append('<li>原始 PPA（功耗 W／延遲 s／面積 mm²）：'+
                html.escape(str(pair['before']['metrics']))+' → '+html.escape(str(pair['after']['metrics']))+
                '；merit '+value(pair['before']['search_merit'])+' → '+value(pair['after']['search_merit'])+'</li>')
        method_rows=[]
        for method in row.get('search_comparisons',[]):
            method_rows.append('<tr>'+''.join('<td>'+html.escape(str(method[k]))+'</td>' for k in
                ('method','requests','new_evaluations','cache_hits','best_J','compliant_episodes','wall_s'))+'</tr>')
        method_table=('<p>比較已完成部分；只有 requests 與完成回合相同時，才可作等預算比較。額外凍結測試不送入策略、不納入搜尋最佳值；其查詢與時間另外計入整體成本。</p><div class="scroll"><table><tr>'+''.join('<th>'+k+'</th>' for k in
            ('方法','互動 requests','新 PPA 評估','快取','最佳 J','達標回合','耗時 s'))+'</tr>'+''.join(method_rows)+'</table></div>') if method_rows else ''
        plot = f'<img src="{name}/training_curve.svg" alt="{name} 實測訓練曲線">' if (run/name/'training_curve.svg').exists() else ''
        summary=row.get('policy_summary')
        details.append(f'<section><h2>{name}</h2><ul>{"".join(comparisons) or "<li>尚未完成前後測。</li>"}</ul>'
                       f'<pre>{html.escape(json.dumps(summary,ensure_ascii=False,indent=2)) if summary else ""}</pre>{method_table}'
                       f'{plot}<p><a href="{name}/training_curve.json">逐回合數值</a> · '
                       f'<a href="{name}/result.json">結果</a> · '
                       f'<a href="{name}/history.sqlite">完整精簡互動與接續點</a></p></section>')
    config = protocol['config']
    runtime_path = run/'status.json'
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    elapsed = runtime.get('elapsed_active_s',0)/3600
    title = '流程驗證，非 60 回合研究結果' if protocol['smoke_only'] else '模型訓練與策略前後測'
    report = f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>RACE-RL 訓練報告</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;background:#f3f7f6;color:#173238;max-width:1100px;margin:32px auto;padding:0 20px}}
section{{background:white;border:1px solid #dbe7e4;border-radius:12px;padding:20px;margin:18px 0}}
.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:9px;border-bottom:1px solid #dbe7e4;text-align:left}}
img{{max-width:100%;height:auto}}a{{color:#08776c}}</style><h1>{title}</h1>
<p>REINFORCE／每模型 {config['episodes']} 訓練回合／輸入 64×64／每回合 {config['steps_per_episode']} 次完整互動／float64／固定 mesh／RL 自選 1–16 顆／scope=2／停用提前停止。</p>
<p>目前批次狀態：{html.escape(runtime.get('status','RUNNING'))}；累計執行時間 {elapsed:.3f} 小時（不含中斷後的等待）。表內訓練時間包含互動與策略更新；模型總時間從子程序啟動量到結束，包含匯入、驗證、前後測及寫出報告。中斷重跑的執行時間也累計，見 model_times.json。</p>
<section><b>實驗條件</b><pre>{html.escape(json.dumps({k:v['requirements'] for k,v in config.get('tasks',{}).items()} or config['requirements'],ensure_ascii=False,indent=2))}</pre>
<p>研究任務由前次已量測可行設計衍生，具有公開的可行性憑證，並非廠商需求或 silicon 校準。憑證不提供給 Actor、不初始化候選、不更新策略。自訂 --requirements 則完全使用指定限制。</p>
<p>Reward={config.get('reward_variant','dense')}；每 {config.get('evaluation_interval',0)} 回合中途凍結測試。Return 對 dense/terminal 均為 -log(回合最佳 J)，不是獨立的學習證據。</p>
<p>訓練 seed={config['seed']}；前後測 seeds={config['evaluation_seeds']}。
前後測使用相同工作負載、初始配置、互動預算與抽樣 seed，過程不更新策略。不同策略可能產生不同動作與快取次數，因此同時記錄互動數與新評估數。</p>
<p>搜尋期間找到的最佳方案改善，不等於策略學習已改善；請同時閱讀凍結策略前後測。
只有一個訓練 seed，此報告是學習診斷，不證明多 seed 穩定性、收斂、泛化或核心假說。PPA 使用未校準的分析式模型。</p></section>
<section><div class="scroll"><table><thead><tr><th>模型</th><th>完成回合</th><th>訓練互動</th><th>顆數</th><th>達標</th><th>功耗 W</th><th>延遲 s</th><th>面積 mm²</th><th>加權成本</th><th>訓練分鐘</th><th>模型總分鐘</th></tr></thead>
<tbody>{''.join(table)}</tbody></table></div></section>{''.join(details)}
<section><p>每次 action、環境回報、設計狀態識別碼與 PPA 存於各模型的 history.sqlite。
它不重複保存每個候選的完整帳本；最終選中設計的完整硬體、mapping、排程及流量帳本存於 selected_design.json.gz。
中斷時接續最後已完整提交的回合，未完成回合會重跑。來源身份不同時拒絕接續。</p></section></html>'''
    (run/'report.html').write_text(report, encoding='utf-8')
