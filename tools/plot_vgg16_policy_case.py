#!/usr/bin/env python3
"""Reconstruct a training archive curve from every saved VGG16 interaction."""
import argparse
import csv
import json
import math
import sqlite3
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


METRICS = ("average_power_w", "latency_s", "area_mm2")
LIMIT_NAMES = ("max_power_w", "max_latency_s", "max_area_mm2")
WEIGHTS = {"balanced": (1/3, 1/3, 1/3), "performance": (.2, .6, .2),
           "power": (.6, .2, .2), "area": (.2, .2, .6)}
LABELS = {"balanced": "平衡", "performance": "效能優先", "power": "功耗優先", "area": "面積優先"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("experiments/runs/vgg16_full_dse_20261104"))
    parser.add_argument("--out", type=Path, default=Path("experiments/runs/vgg16_balanced_case_figures_20261104"))
    args = parser.parse_args()
    run, out = args.run, args.out
    protocol = json.loads((run / "protocol.json").read_text())
    cfg = protocol["config"]
    preference = cfg["requirements"]["preference"]
    if preference not in WEIGHTS:
        raise ValueError("this figure expects a standard recorded preference")
    weights = WEIGHTS[preference]
    limits = [cfg["requirements"][key] for key in LIMIT_NAMES]
    db_path = (run / "vgg16/history.sqlite").resolve()
    db = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)
    episodes = db.execute("SELECT episode, stats, interactions FROM episodes WHERE phase='train' ORDER BY episode").fetchall()
    db.close()

    all_rows, first, current = [], None, None
    per_episode = []
    for episode, stats_json, blob in episodes:
        stats = json.loads(stats_json)
        interactions = json.loads(zlib.decompress(blob))
        for step, action in enumerate(interactions, 1):
            ppa = action.get("ppa")
            compliant = bool(ppa and action["physical_feasible"] and
                             all(ppa[m] <= limit for m, limit in zip(METRICS, limits)))
            if compliant:
                cost = sum(weight * ppa[m] / limit for m, limit, weight in zip(METRICS, limits, weights))
                if current is None or cost < current["weighted_cost"]:
                    current = {"episode": episode, "step": step, "metrics": ppa,
                               "weighted_cost": cost, "compute_count": action["compute_count"]}
                    if first is None:
                        first = current
            all_rows.append({"episode": episode, "step": step, "global_step": (episode-1)*cfg["interactions"]+step,
                             "candidate_compliant": compliant,
                             **({key: current["metrics"][key] for key in METRICS} if current else {}),
                             "weighted_cost": current["weighted_cost"] if current else None})
        saved = stats["archive_assessment"]
        if current is None or not saved["feasible"] or not math.isclose(current["weighted_cost"], saved["weighted_cost"], rel_tol=1e-10, abs_tol=1e-12):
            raise RuntimeError(f"reconstructed archive does not match saved episode {episode}")
        if any(not math.isclose(current["metrics"][key], saved["metrics"][key], rel_tol=1e-10, abs_tol=1e-12) for key in METRICS):
            raise RuntimeError(f"reconstructed PPA does not match saved episode {episode}")
        per_episode.append({"episode": episode, **saved["metrics"], "weighted_cost": saved["weighted_cost"]})
    if len(per_episode) != cfg["episodes"]:
        raise RuntimeError("training record is incomplete")

    out.mkdir(parents=True, exist_ok=True)
    with (out / "training_archive_every_interaction.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("episode", "step", "global_step", "candidate_compliant", *METRICS, "weighted_cost"))
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        "source": str(run), "preference": preference, "weights": weights,
        "training_seed": cfg["seed"], "evaluation_seeds": cfg["evaluation_seeds"],
        "training_episodes": len(per_episode), "interactions_per_episode": cfg["interactions"],
        "limits": cfg["requirements"], "first_feasible": first, "final_archive": current,
        "percentage_change_first_feasible_to_final": {k: 100*(current["metrics"][k]/first["metrics"][k]-1) for k in METRICS},
        "cost_change_percent": 100*(current["weighted_cost"]/first["weighted_cost"]-1),
        "interpretation": "Archive-best search history; it is not itself an independent evaluation of learning. First episode precedes first policy update. PPA is analytical estimate.",
    }
    (out / "case_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    font_path = Path("~/Library/Fonts/kaiu.ttf").expanduser()
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11, "savefig.dpi": 180})
    fig, axs = plt.subplots(3, 2, figsize=(14, 9), sharex="col")
    fig.suptitle(f"VGG16 {LABELS[preference]}偏好：訓練時找到的最佳可行設計", fontsize=18, fontweight="bold", x=.06, ha="left")
    fig.text(.06, .93, "同一加權成本選出的設計；功耗、延遲與面積可能互相取捨，點位均為實際紀錄。", fontsize=11, color="#475569")
    axs[0, 0].set_title("第 1 回合：前 64 次提案（尚未更新策略）", loc="left", fontsize=12)
    axs[0, 1].set_title("全部訓練：每回合結束的最佳設計", loc="left", fontsize=12)
    names = ("平均功耗 (W)", "推論延遲 (s)", "計算面積 (mm²)")
    first_rows = [x for x in all_rows if x["episode"] == 1 and x.get("weighted_cost") is not None]
    for row, (metric, label) in enumerate(zip(METRICS, names)):
        ax1, ax2 = axs[row]
        ax1.step([x["step"] for x in first_rows], [x[metric] for x in first_rows], where="post", color="#087f8c", lw=2.2)
        ax2.step([x["episode"] for x in per_episode], [x[metric] for x in per_episode], where="post", color="#087f8c", lw=2.2)
        ax2.scatter([x["episode"] for x in per_episode], [x[metric] for x in per_episode], s=9, color="#087f8c", zorder=3)
        ax1.scatter([first["step"]], [first["metrics"][metric]], s=44, color="#e08b1c", zorder=4)
        ax2.scatter([len(per_episode)], [current["metrics"][metric]], marker="*", s=95, color="#d44736", zorder=4)
        upper = max(x[metric] for x in first_rows + per_episode) * 1.16
        for ax in (ax1, ax2):
            ax.set_ylim(0, upper)
            ax.grid(axis="y", color="#dce3e8", lw=.7)
            ax.spines[["top", "right"]].set_visible(False)
        ax1.set_ylabel(label)
        delta = summary["percentage_change_first_feasible_to_final"][metric]
        ax2.text(.98, .91, f"首個可行 → 最終：{delta:+.1f}%", transform=ax2.transAxes,
                 ha="right", va="top", fontsize=10, color="#394e5e")
    axs[-1, 0].set_xlabel("第 1 回合的提案次數")
    axs[-1, 1].set_xlabel("訓練回合")
    fig.text(.06, .02, "橘點：第一個符合限制的設計；紅星：訓練期間的最終最佳設計。估算值非晶片實測。", fontsize=10, color="#475569")
    fig.tight_layout(rect=(.04, .05, .99, .91))
    fig.savefig(out / f"vgg16_{preference}_training_archive.png", bbox_inches="tight")
    fig.savefig(out / f"vgg16_{preference}_training_archive.svg", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
