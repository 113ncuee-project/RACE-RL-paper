#!/usr/bin/env python3
"""Summarize a completed VGG16 CNN hierarchy DSE run and make report figures."""
import argparse
import json
import math
from pathlib import Path
import sqlite3
import statistics
import zlib
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mac_count(operator, tensors):
    kind = operator["kind"]
    output = tensors[operator["output"]]["shape"]
    source = tensors[operator["inputs"][0]]["shape"]
    if kind == "Conv":
        return math.prod(output) * (source[1] // operator["groups"]) * math.prod(operator["kernel"])
    if kind == "Linear":
        return math.prod(output) * source[1]
    return 0


def plan_placement(operator, plan, tensors, count):
    """Enumerate output tiles exactly as cnn_design.output_tiles does."""
    shape = tensors[operator["output"]]["shape"]
    nodes = {}
    index = 0
    if len(shape) == 4:
        n, channels, height, width = shape
        starts = ((b, ch, y, x)
                  for b in range(n)
                  for ch in range(0, channels, plan["c"])
                  for y in range(0, height, plan["h"])
                  for x in range(0, width, plan["w"]))
        for _, ch, y, x in starts:
            out_channels = min(channels, ch + plan["c"]) - ch
            out_height = min(height, y + plan["h"]) - y
            out_width = min(width, x + plan["w"]) - x
            output_elements = out_channels * out_height * out_width
            node = (plan["target"] + index) % count if plan["spread"] else plan["target"]
            nodes[node] = nodes.get(node, 0) + 1
            index += 1
    else:
        n, channels = shape
        for _ in range(n):
            for ch in range(0, channels, plan["c"]):
                node = (plan["target"] + index) % count if plan["spread"] else plan["target"]
                nodes[node] = nodes.get(node, 0) + 1
                index += 1
    return nodes, index


def median_sd(values):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return None, None
    return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0


def extract_sessions(db_path):
    connection = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    rows = []
    try:
        for phase, seed, episode, stats, packed_trace in connection.execute(
                "SELECT phase, seed, episode, stats, interactions FROM episodes ORDER BY rowid"):
            trace = json.loads(zlib.decompress(packed_trace).decode("utf-8"))
            rows.append({"phase": phase, "seed": seed, "episode_number": episode,
                         "stats": json.loads(stats), "trace": trace})
    finally:
        connection.close()
    return rows


def make_figures(out_dir, result):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    navy, teal, orange, gray = "#183B56", "#0E8A8A", "#E39C35", "#83929B"

    fig, ax = plt.subplots(figsize=(11, 2.5))
    boxes = [
        ("MODEL INPUT", "VGG16 · 64×64\n40 operators"),
        ("QUANTIZE", "INT8 / INT32\n1 calibration sample"),
        ("BASELINE", "3 fixed seeds\n64 actions each"),
        ("REINFORCE", "60 × 64 actions\n1 update per episode"),
        ("FROZEN EVAL", "Same 3 seeds\n64 actions each"),
        ("DESIGN OUTPUT", "Training archive winner\nchiplet / tile mapping"),
    ]
    ax.set_xlim(0, 12); ax.set_ylim(0, 2); ax.axis("off")
    for i, (title, body) in enumerate(boxes):
        x = 0.15 + i * 1.97
        ax.text(x + .78, 1.08, title + "\n" + body, ha="center", va="center",
                color="#14232D", fontsize=8.5,
                bbox={"boxstyle": "round,pad=0.55", "fc": "#F2F6F8" if i not in (3, 5) else "#E5F3F2",
                      "ec": teal if i in (3, 5) else "#B8C8D0", "lw": 1.1})
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + 1.74, 1.08), xytext=(x + 1.37, 1.08),
                        arrowprops={"arrowstyle": "-|>", "lw": 1.1, "color": gray})
    fig.tight_layout(pad=0.2)
    fig.savefig(out_dir / "workflow.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    evaluations = result["evaluations"]
    metrics = [("average_power_w", "Average power", "W"), ("latency_s", "Inference latency", "s"),
               ("area_mm2", "Compute die area", "mm²")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.35))
    for ax, (key, label, unit) in zip(axes, metrics):
        for xpos, phase, color, marker in [(0, "baseline", orange, "o"), (1, "final", teal, "s")]:
            vals = [row["metrics"].get(key) for row in evaluations.get(phase, [])]
            vals = [v for v in vals if v is not None]
            if vals:
                ax.scatter([xpos] * len(vals), vals, color=color, marker=marker, s=37, zorder=3)
                mean, sd = median_sd(vals)
                ax.errorbar([xpos], [mean], yerr=[sd], fmt="D", color=navy, capsize=4, lw=1.3,
                            markersize=5, zorder=4)
                ax.set_ylim(0, max(vals + [mean + sd]) * 1.12)
        ax.set_xticks([0, 1], ["Baseline", "Final"])
        ax.set_ylabel(unit)
        ax.set_title(label, fontsize=10, color=navy)
        ax.grid(axis="y", color="#E5EAED", lw=.7)
    fig.suptitle("Frozen-policy comparison on the same evaluation seeds", color=navy, fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(out_dir / "baseline_final.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    episodes = result["training_episodes"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)
    for ax, (key, label, unit) in zip(axes, metrics):
        xs, ys = [], []
        for row in episodes:
            metric_value = row.get("archive_assessment", {}).get("metrics", {}).get(key)
            if metric_value is not None:
                xs.append(row["episode"]); ys.append(metric_value)
        if xs:
            ax.plot(xs, ys, color=teal, lw=1.6)
            ax.scatter(xs, ys, s=8, color=teal)
        ax.set_ylabel(unit)
        ax.set_title("Training archive best | " + label, loc="left", fontsize=9, color=navy)
        ax.grid(axis="y", color="#E5EAED", lw=.7)
    axes[-1].set_xlabel("Training episode")
    axes[-1].set_xlim(1, max(1, len(episodes)))
    fig.tight_layout()
    fig.savefig(out_dir / "training_archive.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    episode_counts = result["training_episode_best_counts"]
    proposal_counts = result["training_proposal_count_histogram"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.65, 1]})
    ax = axes[0]
    episodes_x = [row["episode"] for row in episode_counts]
    counts_y = [row["compute_count"] for row in episode_counts]
    ax.step(episodes_x, counts_y, where="mid", color=teal, lw=1.5, alpha=.8)
    ax.scatter(episodes_x, counts_y, color=teal, s=20, zorder=3, label="Episode-best action")
    ax.axhline(result["configuration"]["initial_count"], color=orange, ls="--", lw=1.2,
               label=f"Initial design: {result['configuration']['initial_count']} chiplets")
    ax.axhline(result["topology"]["compute_count"], color=navy, ls=":", lw=1.5,
               label=f"Final archive: {result['topology']['compute_count']} chiplets")
    ax.set_xlim(1, max(1, len(episodes_x)))
    ax.set_ylim(.5, max(result["configuration"]["candidate_compute_counts"]) + .5)
    ax.set_yticks(result["configuration"]["candidate_compute_counts"])
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Chiplet count")
    ax.set_title("Episode-best action count", color=navy)
    ax.grid(axis="y", color="#E5EAED", lw=.7)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    xs = sorted(int(key) for key in proposal_counts)
    ys = [proposal_counts[str(count)] for count in xs]
    axes[1].bar(xs, ys, color=[orange if count == result["topology"]["compute_count"] else teal for count in xs], width=.72)
    axes[1].set_xticks(xs)
    axes[1].set_xlabel("Proposed chiplet count")
    axes[1].set_ylabel("Training actions")
    axes[1].set_title("All training proposals", color=navy)
    axes[1].grid(axis="y", color="#E5EAED", lw=.7)
    for x, y in zip(xs, ys):
        axes[1].text(x, y + max(ys + [1]) * .012, str(y), ha="center", va="bottom", fontsize=6.5, color="#435963")
    fig.suptitle("Chiplet-count search trajectory and proposal coverage", color=navy, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out_dir / "training_count_trajectory.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    topology = result["topology"]
    nodes = topology["nodes"]
    nrows, ncols = topology["mesh_shape"]
    fig, ax = plt.subplots(figsize=(max(5, ncols * 1.45), max(3.6, nrows * 1.3)))
    ax.set_xlim(-2.1, ncols - .3); ax.set_ylim(-.7, nrows - .3); ax.set_aspect("equal"); ax.axis("off")
    node_map = {node["id"]: node for node in nodes}
    for link in topology.get("links", []):
        a = node_map[link["a"]]; b = node_map[link["b"]]
        ax.plot([a["grid_x"], b["grid_x"]], [nrows - 1 - a["grid_y"], nrows - 1 - b["grid_y"]],
                color="#A7B7BF", lw=1.5, zorder=1)
        mid_x = (a["grid_x"] + b["grid_x"]) / 2
        mid_y = nrows - 1 - (a["grid_y"] + b["grid_y"]) / 2
        ax.text(mid_x, mid_y + .28,
                f"c{int(link['a'][1:])}→c{int(link['b'][1:])} {link.get('a_to_b_bytes', 0) / 1e6:.1f} MB\n"
                f"c{int(link['b'][1:])}→c{int(link['a'][1:])} {link.get('b_to_a_bytes', 0) / 1e6:.1f} MB",
                ha="center", va="bottom", fontsize=7, color="#435963",
                bbox={"boxstyle": "round,pad=.2", "fc": "white", "ec": "none", "alpha": .95}, zorder=2)
    for node in nodes:
        x, y = node["grid_x"], nrows - 1 - node["grid_y"]
        mac_share = node.get("mac_share", 0.0)
        label = f"{node['id']}\n{node['tile_count']} tiles ({node.get('tile_share', 0):.1%})\n{mac_share:.1%} MACs"
        ax.text(x, y, label, ha="center", va="center", fontsize=8, color="#132A38", zorder=3,
                bbox={"boxstyle": "round,pad=.42", "fc": "#DFF1EF" if node["tile_count"] else "#F2F4F5",
                      "ec": teal if node["tile_count"] else "#C4CDD1", "lw": 1.1})
    for port in topology.get("memory_ports", []):
        owner = node_map.get(port.get("attached_to"))
        if owner is None:
            continue
        y = nrows - 1 - owner["grid_y"]
        mem_x = -1.25
        ax.plot([mem_x + .43, owner["grid_x"] - .48], [y, y], color="#78909C", lw=1.3, zorder=1)
        ax.text(mem_x, y, f"DRAM0\nread {port.get('read_bytes', 0) / 1e6:.1f} MB\n"
                f"write {port.get('write_bytes', 0) / 1e6:.1f} MB",
                ha="center", va="center", fontsize=7, color="#132A38",
                bbox={"boxstyle": "round,pad=.36", "fc": "#F5F2E8", "ec": "#C7B989", "lw": 1.0}, zorder=3)
    ax.set_title(f"Training archive winner | {topology['compute_count']} chiplets · {nrows}×{ncols} mesh",
                 color=navy, fontsize=11, pad=12)
    fig.tight_layout()
    fig.savefig(out_dir / "topology.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    nodes = topology["nodes"]
    chiplet_labels = [node["id"] for node in nodes]
    tile_shares = [node["tile_share"] * 100 for node in nodes]
    mac_shares = [node["mac_share"] * 100 for node in nodes]
    fig, axes = plt.subplots(1, 2, figsize=(10, max(3.4, len(nodes) * .48)), sharey=True)
    for ax, values, title, color in (
            (axes[0], tile_shares, "Output tile share", teal),
            (axes[1], mac_shares, "Logical MAC share", navy)):
        bars = ax.barh(chiplet_labels, values, color=color, height=.58)
        ax.invert_yaxis()
        ax.set_xlabel("Share of total (%)")
        ax.set_title(title, color=navy)
        ax.set_xlim(0, max(values + [1]) * 1.25)
        ax.grid(axis="x", color="#E5EAED", lw=.7)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + max(values + [1]) * .025,
                    bar.get_y() + bar.get_height() / 2, f"{value:.1f}%",
                    va="center", ha="left", fontsize=8, color="#243B47")
    fig.suptitle("Workload distribution in the selected archive design", color=navy, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(out_dir / "chiplet_shares.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("experiments/runs/vgg16_full_dse_20261104"))
    args = parser.parse_args()
    run = args.run.resolve()
    protocol = read_json(run / "protocol.json")
    status = read_json(run / "vgg16/status.json")
    if not status or status.get("status") != "COMPLETED":
        raise SystemExit("DSE 尚未完成；請等待 status.json 的 status=COMPLETED。")
    report = read_json(run / "vgg16/report.json")
    workload = read_json(run / "vgg16/workload.json")
    numerical = read_json(run / "vgg16/numerical_validation.json")
    design_view = read_json(run / "vgg16/design_view.json")
    sessions = extract_sessions(run / "vgg16/history.sqlite")
    config = protocol["config"]
    tensor_map = {tensor["id"]: tensor for tensor in workload["tensors"]}
    design = report["selected_design"]
    count = design["compute_count"]
    by_id = {row["operator"]: row for row in report["selected_model_feedback"]["operator_placement"]}
    cycle_by_id = {row["operator"]: row for row in report["selected_model_feedback"]["layers"]}
    nodes = [{"id": f"c{i}", "tile_count": 0, "macs": 0, "mac_share": 0.0, "workloads": []}
             for i in range(count)]
    layer_rows = []
    for i, operator in enumerate(workload["operators"]):
        plan = design["plans"][i]
        placement, tile_total = plan_placement(operator, plan, tensor_map, count)
        # Aliases are zero-work graph nodes and are omitted by the placement engine.
        if operator["kind"] in ("Flatten", "Identity"):
            placement, tile_total = {}, 0
        recorded = by_id.get(operator["id"], {})
        if recorded and tile_total != recorded.get("tile_count"):
            raise ValueError(f"tile accounting mismatch for {operator['id']}")
        mac_total = mac_count(operator, tensor_map)
        assigned_macs = {node: 0 for node in placement}
        if mac_total and tile_total:
            # Exact output-cell distribution across the same tiles used by the mapper.
            shape = tensor_map[operator["output"]]["shape"]
            index = 0
            if len(shape) == 4:
                n, channels, height, width = shape
                tiles = ((ch, y, x) for _b in range(n)
                         for ch in range(0, channels, plan["c"])
                         for y in range(0, height, plan["h"])
                         for x in range(0, width, plan["w"]))
                for ch, y, x in tiles:
                    elements = (min(channels, ch + plan["c"]) - ch) * (min(height, y + plan["h"]) - y) * (min(width, x + plan["w"]) - x)
                    node = (plan["target"] + index) % count if plan["spread"] else plan["target"]
                    source = tensor_map[operator["inputs"][0]]["shape"]
                    assigned_macs[node] = assigned_macs.get(node, 0) + elements * (source[1] // operator["groups"]) * math.prod(operator["kernel"])
                    index += 1
            else:
                _, channels = shape
                source = tensor_map[operator["inputs"][0]]["shape"]
                for _b in range(shape[0]):
                    for ch in range(0, channels, plan["c"]):
                        elements = min(channels, ch + plan["c"]) - ch
                        node = (plan["target"] + index) % count if plan["spread"] else plan["target"]
                        assigned_macs[node] = assigned_macs.get(node, 0) + elements * source[1]
                        index += 1
            if sum(assigned_macs.values()) != mac_total:
                raise ValueError(f"MAC accounting mismatch for {operator['id']}")
        node_tiles = {f"c{node}": amount for node, amount in placement.items()}
        if not (operator["kind"] in ("Flatten", "Identity")):
            for node, amount in placement.items():
                row = nodes[node]
                row["tile_count"] += amount
                row["macs"] += assigned_macs.get(node, 0)
                row["workloads"].append({"operator": operator["id"], "kind": operator["kind"],
                                         "tiles": amount, "macs": assigned_macs.get(node, 0)})
        op_cycle = cycle_by_id.get(operator["id"], {})
        layer_rows.append({"index": i + 1, "operator": operator["id"], "kind": operator["kind"],
                           "input_shape": tensor_map[operator["inputs"][0]]["shape"],
                           "output_shape": tensor_map[operator["output"]]["shape"],
                           "plan": plan, "tile_count": tile_total, "placement": node_tiles,
                           "logical_macs": mac_total, "modeled_cycles": op_cycle.get("cycles", 0),
                           "node_service_cycles": op_cycle.get("node_service_cycles", []),
                           "dram_service_cycles": op_cycle.get("dram_service_cycles", 0),
                           "vector_ops": op_cycle.get("vector_ops", 0),
                           "payload_service_bytes": op_cycle.get("payload_service_bytes", 0),
                           "alias_of": op_cycle.get("alias_of")})
    total_macs = sum(node["macs"] for node in nodes)
    total_tiles = sum(node["tile_count"] for node in nodes)
    for node in nodes:
        node["tile_share"] = node["tile_count"] / total_tiles if total_tiles else 0.0
        node["mac_share"] = node["macs"] / total_macs if total_macs else 0.0
    grid_shape = report["selected_model_feedback"].get("mesh_shape", [1, count])
    for node in nodes:
        idx = int(node["id"][1:])
        node["grid_x"] = idx % grid_shape[1]
        node["grid_y"] = idx // grid_shape[1]
    topology = {**design_view, "nodes": nodes, "compute_count": count, "mesh_shape": grid_shape,
                "logical_macs": total_macs, "links": design_view.get("links", [])}

    evaluation = {"baseline": [], "final": []}
    training = []
    phase_groups = {}
    for session in sessions:
        s = session["stats"]
        phase = session["phase"]
        assessment = s.get("assessment", {})
        best_step = None
        if session["trace"]:
            best_step = min(enumerate(session["trace"], 1),
                            key=lambda pair: pair[1].get("best_objective", 1.0))
        best_action_step = best_step[0] if best_step else None
        best_action_count = best_step[1].get("compute_count") if best_step else None
        item = {"seed": session["seed"], "episode": session["episode_number"],
                "feasible": assessment.get("feasible"), "metrics": assessment.get("metrics", {}),
                "search_merit": assessment.get("search_merit"), "oracle_requests": s.get("oracle_requests"),
                "oracle_evaluations": s.get("oracle_evaluations"), "cache_hits": s.get("cache_hits"),
                "wall_time_s": s.get("wall_time_s"), "return": s.get("return"),
                "best_objective": s.get("best_objective"), "parameter_updates": s.get("parameter_updates"),
                "nonzero_reward_steps": s.get("nonzero_reward_steps"),
                "archive_assessment": s.get("archive_assessment", {}),
                "selected_compute_count": best_action_count, "best_action_step": best_action_step}
        if phase in evaluation:
            evaluation[phase].append(item)
        elif phase == "train":
            training.append(item)
    training.sort(key=lambda x: x["episode"])
    proposal_histogram = Counter(
        int(action["compute_count"])
        for session in sessions if session["phase"] == "train"
        for action in session["trace"]
        if action.get("compute_count") is not None)
    episode_best_counts = [
        {"episode": row["episode"], "compute_count": row["selected_compute_count"]}
        for row in training if row["selected_compute_count"] is not None]
    summary = {}
    for phase, rows in evaluation.items():
        summary[phase] = {key: {"mean": median_sd([row["metrics"].get(key) for row in rows])[0],
                                "sd": median_sd([row["metrics"].get(key) for row in rows])[1]}
                          for key in ("average_power_w", "latency_s", "area_mm2")}
        summary[phase]["feasible_count"] = sum(bool(row["feasible"]) for row in rows)
    deltas = {}
    for key in ("average_power_w", "latency_s", "area_mm2"):
        base = summary["baseline"][key]["mean"]
        final = summary["final"][key]["mean"]
        deltas[key] = ((final - base) / base) if base else None
    paired_changes = {}
    baseline_by_seed = {row["seed"]: row for row in evaluation["baseline"]}
    final_by_seed = {row["seed"]: row for row in evaluation["final"]}
    for seed in sorted(set(baseline_by_seed) & set(final_by_seed)):
        paired_changes[str(seed)] = {}
        for key in ("average_power_w", "latency_s", "area_mm2"):
            base = baseline_by_seed[seed]["metrics"].get(key)
            final = final_by_seed[seed]["metrics"].get(key)
            paired_changes[str(seed)][key] = (final - base) / base if base else None

    # Locate the exact coupled action that first placed the final archive winner.
    winner_hash = __import__("hashlib").sha256(json.dumps(
        report["selected_design"], sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()
    winner_action = None
    for session in sessions:
        if session["phase"] != "train":
            continue
        for step, action in enumerate(session["trace"], 1):
            if action.get("design_hash") == winner_hash:
                winner_action = {"phase": session["phase"], "seed": session["seed"],
                    "episode": session["episode_number"], "step": step,
                    "compute_count": action.get("compute_count"), "scope": action.get("scope"),
                    "edits": action.get("edits"), "reward": action.get("reward"),
                    "ppa": action.get("ppa"), "status": action.get("status")}
                break
        if winner_action:
            break
    episode_blocks = []
    for start in range(1, len(training) + 1, 10):
        block = training[start - 1:start + 9]
        last = block[-1]
        episode_blocks.append({"start_episode": start, "end_episode": start + len(block) - 1,
            "feasible_episode_count": sum(bool(row["feasible"]) for row in block),
            "requests": sum(row["oracle_requests"] or 0 for row in block),
            "evaluations": sum(row["oracle_evaluations"] or 0 for row in block),
            "cache_hits": sum(row["cache_hits"] or 0 for row in block),
            "parameter_updates": sum(row["parameter_updates"] or 0 for row in block),
            "mean_episode_wall_s": statistics.mean(row["wall_time_s"] or 0 for row in block),
            "archive_assessment_at_end": last["archive_assessment"]})

    result = {"status": report["status"], "run_path": str(run), "protocol": protocol,
              "configuration": config, "run_report": report, "numerical_validation": numerical,
              "workload_summary": {"name": workload["name"], "operator_count": len(workload["operators"]),
                  "tensor_count": len(workload["tensors"]), "logical_macs": total_macs,
                  "operator_kinds": {kind: sum(op["kind"] == kind for op in workload["operators"])
                                     for kind in sorted({op["kind"] for op in workload["operators"]})},
                  "source_metadata": workload.get("source_metadata", [])},
              "summary": summary, "paired_mean_relative_change": deltas,
              "paired_seed_relative_changes": paired_changes,
              "evaluations": evaluation, "training_episodes": training,
              "training_episode_best_counts": episode_best_counts,
              "training_proposal_count_histogram": {str(k): proposal_histogram.get(k, 0)
                                                     for k in config["candidate_compute_counts"]},
              "training_episode_blocks": episode_blocks, "winner_action": winner_action,
              "training_updates": sum(item["parameter_updates"] or 0 for item in training),
              "topology": topology, "layers": layer_rows,
              "feedback": report["selected_model_feedback"],
              "sessions_total": len(sessions), "sessions_by_phase": {
                  phase: sum(row["phase"] == phase for row in sessions) for phase in ("baseline", "train", "final")}}
    out_path = run / "analysis.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    make_figures(run / "figures", result)
    print(f"分析完成：{out_path}")
    print(f"訓練回合 {len(training)}；前測 {len(evaluation['baseline'])}；後測 {len(evaluation['final'])}；" 
          f"拓撲 {count} chiplets / {grid_shape[0]}×{grid_shape[1]} mesh")


if __name__ == "__main__":
    main()
