#!/usr/bin/env python3
"""Create Chinese, source-backed plots of one recorded training archive."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def main():
    root = Path("experiments/runs/preference_case_continue_alexnet_performance_414_ep20_20261105/analysis")
    data = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
    with (root / "archive_every_action.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [(int(x["proposal"]), float(x["archive_priority"])*1000) for x in rows if x["archive_priority"]]
    events, episodes = data["archive_events"], data["episodes"]
    font_path = str(Path("~/Library/Fonts/kaiu.ttf").expanduser())
    font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font_path).get_name(),
                         "axes.unicode_minus": False, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "axes.grid.axis": "y", "grid.alpha": .22, "savefig.dpi": 200})
    out = root / "brief"
    out.mkdir(parents=True, exist_ok=True)
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(4.3, 4.6), sharex=True, layout="constrained")
    for axis in (ax, bx):
        axis.step([x for x, _ in rows], [y for _, y in rows], where="post", lw=2, color="#007f88")
        axis.scatter([e["proposal"] for e in events], [e["metrics"]["latency_s"]*1000 for e in events],
                     s=26, color="#d05a35", zorder=3)
        axis.set_ylabel("延遲（毫秒）")
    ax.set_title("搜尋期間累積最佳方案的延遲")
    ax.set_ylim(0, 120)
    ax.axvline(16, color="#64748b", linestyle="--", lw=1)
    bx.set_ylim(60, 80)
    bx.set_title("局部放大：縱軸 60–80 毫秒")
    bx.set_xlabel("訓練期間的提案次數")
    fig.savefig(out / "延遲搜尋歷程.png"); plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(4.3, 5.0), sharex=True, layout="constrained")
    x = [e["episode"] for e in episodes]
    metrics = (("average_power_w", "功耗（W）", 1), ("latency_s", "延遲（毫秒）", 1000),
               ("area_mm2", "面積（mm²）", 1))
    for axis, (key, label, scale) in zip(axes, metrics):
        chosen = [e["episode_best"]["metrics"][key]*scale if e["feasible"] else float("nan") for e in episodes]
        retained = [e["archive"]["metrics"][key]*scale if e["archive"] else float("nan") for e in episodes]
        axis.plot(x, chosen, color="#4475a0", marker="o", markersize=2.8, lw=1, label="當回合最佳")
        axis.step(x, retained, where="post", color="#dc8435", lw=1.8, label="累積最佳方案")
        axis.set_ylabel(label); axis.set_ylim(bottom=0)
    axes[0].set_title("各回合與累積最佳方案的 PPA")
    axes[0].legend(frameon=False, fontsize=7)
    axes[-1].set_xlabel("訓練回合")
    fig.savefig(out / "三項PPA歷程.png"); plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
