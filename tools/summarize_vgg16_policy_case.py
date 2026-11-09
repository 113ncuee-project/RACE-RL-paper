#!/usr/bin/env python3
"""Summarize all original and held-out evaluations of the selected VGG16 case."""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


RUN = Path("experiments/runs/vgg16_full_dse_20261104")
FOLLOWUP = Path("experiments/runs/vgg16_balanced_policy_followup_20261104")
OUT = Path("experiments/runs/vgg16_balanced_case_figures_20261104")
METRICS = ("average_power_w", "latency_s", "area_mm2")


def main():
    original = json.loads((RUN / "vgg16/report.json").read_text())
    added = json.loads((FOLLOWUP / "results.json").read_text())
    plan = json.loads((FOLLOWUP / "protocol.json").read_text())
    summary = json.loads((OUT / "case_summary.json").read_text())
    if added["status"] != "COMPLETED":
        raise RuntimeError("the held-out evaluation has not completed")
    initial = {x["phase_seed"]: x["assessment"] for x in original["sessions"] if x["phase"] == "baseline"}
    final = {x["phase_seed"]: x["assessment"] for x in original["sessions"] if x["phase"] == "final"}
    if set(initial) != set(final):
        raise RuntimeError("original evaluation seeds are not paired")
    new_initial = {x["seed"]: x["assessment"] for x in added["rows"] if x["phase"] == "before"}
    new_final = {x["seed"]: x["assessment"] for x in added["rows"] if x["phase"] == "after"}
    if set(new_initial) != set(plan["new_evaluation_seeds"]) or set(new_final) != set(new_initial):
        raise RuntimeError("missing new evaluation results")
    groups = [("原始評估", plan["original_interaction_budget"], initial, final),
              ("新增評估", plan["new_interaction_budget_per_policy_per_seed"], new_initial, new_final)]
    for _, _, pre, post in groups:
        if not all(x["feasible"] for x in list(pre.values()) + list(post.values())):
            raise RuntimeError("at least one evaluation has no feasible design; report feasibility before means")

    font_path = Path("~/Library/Fonts/kaiu.ttf").expanduser()
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11, "savefig.dpi": 180})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7), sharey=True)
    fig.suptitle("凍結策略：訓練前後的成對評估", fontsize=18, x=.06, ha="left")
    fig.text(.06, .90, "每條線是一個評估種子；兩端使用同樣限制和提案次數，加權成本越低越好。", color="#475569")
    aggregations = []
    for ax, (label, budget, pre, post) in zip(axes, groups):
        seeds = sorted(pre)
        old = [pre[s]["weighted_cost"] for s in seeds]
        new = [post[s]["weighted_cost"] for s in seeds]
        wins = sum(a < b for a, b in zip(new, old))
        ties = sum(a == b for a, b in zip(new, old))
        before_mean, after_mean = statistics.mean(old), statistics.mean(new)
        gain = 100*(before_mean-after_mean)/before_mean
        aggregate = {"name":label,"budget":budget,"seeds":seeds,"wins":wins,"ties":ties,
                     "losses":len(seeds)-wins-ties,"before_mean_cost":before_mean,
                     "after_mean_cost":after_mean,"mean_cost_reduction_percent":gain,
                     "before_mean_metrics":{k:statistics.mean(pre[s]["metrics"][k] for s in seeds) for k in METRICS},
                     "after_mean_metrics":{k:statistics.mean(post[s]["metrics"][k] for s in seeds) for k in METRICS}}
        aggregations.append(aggregate)
        for s in seeds:
            a, b = pre[s]["weighted_cost"], post[s]["weighted_cost"]
            color = "#168675" if b < a else "#c9473d" if b > a else "#64748b"
            ax.plot([0,1], [a,b], marker="o", lw=2, ms=6, color=color, alpha=.9)
            ax.annotate(str(s), (1,b), xytext=(5,0), textcoords="offset points", va="center", fontsize=9, color=color)
        ax.set_title(f"{label}：{budget} 次提案／種子", loc="left", fontsize=13)
        ax.text(.04, .94, f"平均成本 {before_mean:.4f} → {after_mean:.4f}  ({-gain:+.1f}%)\n"
                f"{wins} 勝、{len(seeds)-wins-ties} 負、{ties} 平", transform=ax.transAxes,
                va="top", fontsize=10, bbox=dict(facecolor="white", edgecolor="#dce3e8", boxstyle="round,pad=.4"))
        ax.set_xticks([0,1], ["訓練前", "訓練後"])
        ax.set_xlim(-.2,1.28)
        ax.set_ylim(0, max([1.0] + old + new)*1.05)
        ax.grid(axis="y", color="#dce3e8", lw=.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("符合限制設計的加權成本")
    fig.text(.06, .03, "綠線：訓練後較低；紅線：訓練後較高。新增評估只有一個訓練種子，屬小樣本探索。", color="#475569", fontsize=10)
    fig.tight_layout(rect=(.04,.07,.99,.87))
    fig.savefig(OUT / "vgg16_balanced_frozen_policy.png", bbox_inches="tight")
    fig.savefig(OUT / "vgg16_balanced_frozen_policy.svg", bbox_inches="tight")
    plt.close(fig)
    (OUT / "evaluation_summary.json").write_text(json.dumps(aggregations, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

    first, last = summary["first_feasible"], summary["final_archive"]
    lines = [
        "# VGG16 平衡偏好：訓練過程與前後策略比較", "",
        "本案例是在檢查既有結果後選出，用於解說 RACE-RL 的搜尋過程；新增種子評估是在選出案例後才執行，並完整列出全部結果。", "",
        "## 設定", "",
        "- VGG16 輸入 1×3×64×64，訓練種子 7，平衡偏好，功耗、延遲、面積各佔 1/3。",
        "- 候選晶粒數 1–16，初始 2 顆；60 個訓練回合，每回合 64 次提案，REINFORCE 更新 60 次。",
        f"- 限制：平均功耗 ≤ {summary['limits']['max_power_w']:.6f} W，延遲 ≤ {summary['limits']['max_latency_s']:.6f} s，計算面積 ≤ {summary['limits']['max_area_mm2']:.1f} mm²。",
        "- 加權成本 J = (功耗／功耗上限 + 延遲／延遲上限 + 面積／面積上限)／3；符合限制時越低越好。", "",
        "## 訓練期間搜尋紀錄", "",
        "![訓練時的最佳設計](vgg16_balanced_training_archive.png)", "",
        f"第一個符合限制的設計出現在第 {first['episode']} 回合第 {first['step']} 次提案，使用 {first['compute_count']} 顆；"
        f"最終最佳設計首次出現在第 {last['episode']} 回合第 {last['step']} 次提案，使用 {last['compute_count']} 顆。", "",
        "| 項目 | 首個可行設計 | 最終最佳設計 | 變化 |", "|---|---:|---:|---:|",
    ]
    for k,label,unit,decimals in (("average_power_w","平均功耗","W",4),("latency_s","延遲","s",4),("area_mm2","計算面積","mm²",1)):
        lines.append(f"| {label} | {first['metrics'][k]:.{decimals}f} {unit} | {last['metrics'][k]:.{decimals}f} {unit} | {summary['percentage_change_first_feasible_to_final'][k]:+.1f}% |")
    lines.extend([f"| 加權成本 | {first['weighted_cost']:.4f} | {last['weighted_cost']:.4f} | {summary['cost_change_percent']:+.1f}% |", "",
                  "**解讀：**第一回合的改善發生在第一次策略更新之前，顯示搜尋過程找到更好的設計，不能單靠這條曲線證明策略學習有效；以下的凍結策略前後比較才是直接的學習診斷。", "",
                  "## 凍結策略前後評估", "", "![凍結策略前後評估](vgg16_balanced_frozen_policy.png)", "",
                  "原始與新增評估的每一對分別使用相同限制、相同模型、相同評估種子及同樣提案次數。兩組評估的預算不同，故分開呈現；訓練提案另外計算。", ""])
    for (label,budget,pre,post), agg in zip(groups,aggregations):
        lines.extend([f"### {label}：每個策略每個種子 {budget} 次提案", "",
                      "| 評估種子 | 訓練前 J | 訓練後 J | 成本變化 | 訓練前 P／L／A | 訓練後 P／L／A |",
                      "|---:|---:|---:|---:|---|---|"])
        for seed in sorted(pre):
            a,b=pre[seed],post[seed]
            am,bm=a["metrics"],b["metrics"]
            delta=100*(b["weighted_cost"]/a["weighted_cost"]-1)
            lines.append(f"| {seed} | {a['weighted_cost']:.4f} | {b['weighted_cost']:.4f} | {delta:+.1f}% | {am['average_power_w']:.4f} W／{am['latency_s']:.4f} s／{am['area_mm2']:.1f} mm² | {bm['average_power_w']:.4f} W／{bm['latency_s']:.4f} s／{bm['area_mm2']:.1f} mm² |")
        lines.extend(["", f"平均成本 {agg['before_mean_cost']:.4f} → {agg['after_mean_cost']:.4f}，變化 {-agg['mean_cost_reduction_percent']:+.1f}%；{agg['wins']} 勝、{agg['losses']} 負、{agg['ties']} 平。", ""])
    lines.extend(["## 限制與可重現性", "",
                  "- 這是檢視結果後挑選的示範案例，只有一個訓練種子，不能據此宣稱所有 CNN 或偏好都有效，也不是與其他搜尋方法的比較。",
                  "- 原始與新增評估的政策參數分別凍結，新增評估先比對程式雜湊、策略雜湊與原始前／後評估軌跡各 3 步，再執行先列出的新種子。",
                  "- PPA 為解析模型與宣告係數的估算，非晶片實測；VGG16 使用隨機初始化，未驗證資料集準確率。",
                  "- 原始紀錄：`experiments/runs/vgg16_full_dse_20261104/vgg16/report.json` 與 `history.sqlite`；新增評估：`experiments/runs/vgg16_balanced_policy_followup_20261104/protocol.json` 與 `results.json`；圖中逐互動資料見 `training_archive_every_interaction.csv`。", ""])
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(aggregations,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
