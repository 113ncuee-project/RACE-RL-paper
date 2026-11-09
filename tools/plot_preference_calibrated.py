#!/usr/bin/env python3
"""Report all paired RACE preference trials and render actual and zoomed PPA."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon
import numpy as np

PREFS = ('balanced', 'power', 'performance', 'area')
LABELS = {'balanced': '平衡', 'power': '功耗優先', 'performance': '效能優先', 'area': '面積優先'}
COLORS = {'balanced': '#285f93', 'power': '#d17823', 'performance': '#208a79', 'area': '#8c579b'}
NAMES = {'resnet18': 'ResNet18', 'mobilenet_v2': 'MobileNetV2', 'vgg16': 'VGG16'}
KEYS = ('average_power_w', 'latency_s', 'area_mm2')
LIMIT_KEYS = ('max_power_w', 'max_latency_s', 'max_area_mm2')
AXIS = {'power': 0, 'performance': 1, 'area': 2}
VERTICES = [(-math.sqrt(3)/2, -.5), (0, 1), (math.sqrt(3)/2, -.5)]


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def draw(model, seed, rows, limits, out):
    # All panels in each row use the same axis mapping; unavailable results
    # are explicitly labeled and excluded from zoom-range calculation.
    available = [np.array([rows[p]['metrics'][k] for k in KEYS], dtype=float)
                 for p in PREFS if rows[p]['feasible']]
    if not available:
        return
    matrix = np.vstack(available)
    mins, maxs = matrix.min(axis=0), matrix.max(axis=0)
    spans = maxs - mins
    fig, axes = plt.subplots(2, 4, figsize=(18, 11), squeeze=False)
    def project(values, zoom):
        if zoom:
            radius = np.divide(values - mins, spans, out=np.full(3, .5), where=spans > 1e-12)
        else:
            radius = values / limits
        return [(x*r, y*r) for (x, y), r in zip(VERTICES, radius)]
    base = np.array([rows['balanced']['metrics'][k] for k in KEYS], dtype=float)
    for r in range(2):
        zoom = r == 1
        for col, pref in enumerate(PREFS):
            ax = axes[r, col]
            for f in (.25, .5, .75, 1.):
                ax.add_patch(Polygon([(x*f, y*f) for x, y in VERTICES], closed=True, fill=False,
                                     edgecolor='#d8e0e7', linewidth=1))
            for x, y in VERTICES:
                ax.plot([0, x], [0, y], color='#d8e0e7', linewidth=.8)
            if rows['balanced']['feasible']:
                c = project(base, zoom)
                ax.plot([v[0] for v in c+[c[0]]], [v[1] for v in c+[c[0]]],
                        color='#445261', linestyle='--', linewidth=1.4)
            item = rows[pref]
            if item['feasible']:
                values = np.array([item['metrics'][k] for k in KEYS], dtype=float)
                c = project(values, zoom)
                ax.add_patch(Polygon(c, closed=True, facecolor=COLORS[pref], alpha=.27))
                ax.plot([v[0] for v in c+[c[0]]], [v[1] for v in c+[c[0]]],
                        color=COLORS[pref], marker='o', markersize=4, linewidth=1.8)
                ax.text(0, -.88, f'{values[0]:.5f} W / {values[1]:.5f} s / {values[2]:.1f} mm²',
                        ha='center', fontsize=9)
                if pref != 'balanced' and rows['balanced']['feasible']:
                    delta = 100 * (values / base - 1)
                    ax.text(0, -1.03, f'相對平衡: {delta[0]:+.2f}% / {delta[1]:+.2f}% / {delta[2]:+.2f}%',
                            ha='center', fontsize=9)
            else:
                ax.text(0, .1, '未達標', ha='center', fontsize=14, color='#a33')
            ax.text(0, 1.11, '延遲', ha='center', fontsize=11)
            ax.text(-.97, -.64, '功耗', ha='center', fontsize=11)
            ax.text(.97, -.64, '面積', ha='center', fontsize=11)
            ax.set_title(LABELS[pref], color=COLORS[pref], fontsize=15, pad=13)
            ax.set_aspect('equal')
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.14, 1.3)
            ax.axis('off')
    fig.suptitle(f'{NAMES[model]}　同一 seed {seed} 的四種偏好', fontsize=20)
    fig.text(.005, .73, '上排：相對原始上限，中心為 0', fontsize=12, rotation=90, va='center')
    fig.text(.005, .29, '下排：同軸同範圍放大，常數軸置中', fontsize=12, rotation=90, va='center')
    fig.text(.06, .025,
             f'放大範圍 功耗 {mins[0]:.5f}–{maxs[0]:.5f} W；延遲 {mins[1]:.5f}–{maxs[1]:.5f} s；面積 {mins[2]:.1f}–{maxs[2]:.1f} mm²。虛線：平衡。',
             fontsize=11)
    fig.tight_layout(rect=(.025, .05, 1, .90), h_pad=7)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', type=Path, required=True)
    args = p.parse_args()
    study = args.study
    manifest = read(study / 'manifest.json')
    data = read(study / 'results.json')
    if data['manifest_identity'] != manifest['identity']:
        raise ValueError('results do not match manifest')
    fonts = ['/System/Library/Fonts/STHeiti Medium.ttc', '/System/Library/Fonts/PingFang.ttc']
    for font in fonts:
        if Path(font).exists():
            font_manager.fontManager.addfont(font)
            plt.rcParams['font.family'] = font_manager.fontManager.ttflist[-1].name
            break
    plt.rcParams['axes.unicode_minus'] = False
    index = {(r['model'], r['seed'], r['preference']): r for r in data['rows']}
    out = study / 'figures'
    out.mkdir(exist_ok=True)
    expected = len(manifest['models']) * len(manifest['seeds']) * len(PREFS)
    lines = ['# RACE-RL 偏好權重與 PPA 取捨探索', '',
             f"已完成 {len(data['rows'])}/{expected} 組。每次 {manifest['budget']} 次 RACE 提案；"
             f"新 seed {', '.join(map(str, manifest['seeds']))}。", '',
             '本研究先用舊 seed 341–345 的已稽核、達標候選，按模型計算各 PPA 的 P05–P95 變動範圍。'
             '平衡對三個變動範圍各給 1/3；其他偏好在該範圍尺度上的主權重為 '
             f"{manifest['strength']*100:.1f}%，其餘兩項平分。硬上限、硬體與 PPA 模型未改。", '',
             '為沿用原本「數值除以限制」的實作，權重已依舊資料範圍換算；實際送入策略及評分器的權重如下。'
             '此平衡定義與舊版直接對三項限制各給 1/3 不同，不能把結果歸因為只改了偏好百分比。', '',
             '| 模型 | 偏好 | 範圍權重 功耗/延遲/面積 | 實際輸入的限制正規化權重 功耗/延遲/面積 |',
             '|---|---|---|---|']
    for model in manifest['models']:
        for pref in PREFS:
            spec = manifest['calibration'][model]['preferences'][pref]
            a = '/'.join(f'{x:.4f}' for x in spec['semantic_range_weights'])
            b = '/'.join(f'{x:.4f}' for x in manifest['requirements'][model][pref]['weights'])
            lines.append(f'| {NAMES[model]} | {LABELS[pref]} | {a} | {b} |')
    lines += ['', '此結果屬資料驅動的探索性研究；校準範圍、主權重及顯示方式是依舊資料選擇。'
              'RACE 使用不同偏好的同 seed 不保證產生相同搜尋軌跡。PPA 為未經晶片量測校準的分析模型。', '',
              '## 全部結果', '',
              '| 模型 | seed | 偏好 | 達標 | 功耗 W | 延遲 s | 面積 mm² | 請求 |',
              '|---|---:|---|---|---:|---:|---:|---:|']
    for model in manifest['models']:
        for seed in manifest['seeds']:
            for pref in PREFS:
                r = index.get((model, seed, pref))
                if r is None:
                    lines.append(f'| {NAMES[model]} | {seed} | {LABELS[pref]} | 尚未完成 | — | — | — | — |')
                    continue
                v = r['metrics']
                fmt = lambda x: '—' if x is None else f'{x:.6g}'
                lines.append(f"| {NAMES[model]} | {seed} | {LABELS[pref]} | {'是' if r['feasible'] else '否'} | "
                             f"{fmt(v['average_power_w'])} | {fmt(v['latency_s'])} | {fmt(v['area_mm2'])} | {r['requests']} |")
    lines += ['', '## 同種子相對平衡的目標變化', '',
              '負值表示目標下降，正值表示目標增加；未達標或尚未完成標為「—」。', '',
              '| 模型 | seed | 功耗優先功耗 | 效能優先延遲 | 面積優先面積 |',
              '|---|---:|---:|---:|---:|']
    for model in manifest['models']:
        for seed in manifest['seeds']:
            base = index.get((model, seed, 'balanced'))
            cells = []
            for pref in PREFS[1:]:
                r = index.get((model, seed, pref))
                if not base or not r or not base['feasible'] or not r['feasible']:
                    cells.append('—')
                else:
                    k = KEYS[AXIS[pref]]
                    cells.append(f"{100*(r['metrics'][k]/base['metrics'][k]-1):+.3f}%")
            lines.append(f"| {NAMES[model]} | {seed} | {' | '.join(cells)} |")
    lines += ['', '## 三角圖', '',
              '每個模型只選一個四種偏好都完成且平衡達標的共同 seed；選取方式為平衡結果距離'
              '平衡結果各 PPA 中位數最近（以校準範圍正規化），不逐偏好挑種子。所有種子見上表。'
              '上排以各項硬上限正規化；下排以同一圖內四種偏好的最小與最大值放大，三軸不共用同一數值單位，不能跨圖比較距離。']
    for model in manifest['models']:
        complete = []
        for seed in manifest['seeds']:
            group = {pref: index.get((model, seed, pref)) for pref in PREFS}
            if all(group.values()) and group['balanced']['feasible']:
                complete.append((seed, group))
        if not complete:
            lines.append(f'\n- {NAMES[model]}：尚無可畫出的完整配對。')
            continue
        limits = np.array([manifest['calibration'][model]['limits'][k] for k in LIMIT_KEYS])
        scales = np.array([manifest['calibration'][model]['ranges'][k] for k in KEYS])
        vectors = [np.array([g['balanced']['metrics'][k] for k in KEYS]) for _, g in complete]
        med = np.median(vectors, axis=0)
        seed, group = min(complete, key=lambda x: (sum(((np.array([x[1]['balanced']['metrics'][k]
                               for k in KEYS]) - med)/scales)**2), x[0]))
        image = out / f'{model}_paired_triangle.png'
        draw(model, seed, group, limits, image)
        lines.append(f'\n- {NAMES[model]}：共同 seed {seed}，圖檔 `figures/{image.name}`。')
    lines += ['', '放大圖中若某一軸四組數值相同，該軸固定畫在 50%，不表示實際數值為 50%。'
              '原始數據、校準候選雜湊、權重及每筆執行紀錄分別保存在 manifest.json、results.json 與 runs/。', '']
    (study / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
