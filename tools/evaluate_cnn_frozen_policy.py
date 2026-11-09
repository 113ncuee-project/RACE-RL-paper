#!/usr/bin/env python3
"""Evaluate saved CNN policy against its original initialization on new seeds.

The input run is opened read-only. Each policy receives the same environment,
evaluation seeds, and interaction budget. The output includes every attempted
seed and a short replay check against the source run before new evaluation.
"""
import argparse
from dataclasses import asdict
import hashlib
import io
import inspect
import json
import random
import sqlite3
import time
from pathlib import Path

import numpy as np
import torch
import race_rl

from race_rl.accelerators.cnn_environment import GemminiCNNEnvironment
from race_rl.accelerators.cnn_ppa import load_cnn_cost_profile
from race_rl.accelerators.gemmini_model import PlatformScenario
from race_rl.accelerators.memory_hierarchy import MemoryHierarchy
from race_rl.model_suite import frozen_episode
from race_rl.recording import policy_hash, unpacked
from race_rl.requirements import PPARequirements
from race_rl.serialization import code_hashes, digest, save_json
from race_rl.training import RACETrainer
from race_rl.workload import Operator, Tensor, Workload


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--replay-steps", type=int, default=3)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output directory must be empty")
    if args.budget < 1 or args.replay_steps < 1 or len(args.seeds) != len(set(args.seeds)):
        parser.error("use a positive budget and distinct seeds")

    source = args.run.resolve()
    protocol_path = source / "protocol.json"
    model_dir = source / args.model
    report_path = model_dir / "report.json"
    workload_path = model_dir / "workload.json"
    db_path = model_dir / "history.sqlite"
    protocol, report, workload_data = map(read_json, (protocol_path, report_path, workload_path))
    cfg = protocol["config"]
    if protocol["source_hashes"] != code_hashes():
        raise RuntimeError("current source hashes differ from the source experiment")
    prior_seeds = set(cfg["evaluation_seeds"]) | {cfg["seed"]}
    if prior_seeds & set(args.seeds):
        raise ValueError("new evaluation seeds must be disjoint from original seeds")
    workload = Workload(
        tensors=tuple(Tensor(**x) for x in workload_data["tensors"]),
        operators=tuple(Operator(**x) for x in workload_data["operators"]),
        outputs=tuple(workload_data["outputs"]), name=workload_data["name"],
        source_metadata=tuple(tuple(x) for x in workload_data["source_metadata"]),
    )
    if digest(workload) != digest(workload_data):
        raise RuntimeError("reconstructed workload differs from the saved workload")
    memory = MemoryHierarchy.from_record(cfg["memory"])
    platform = PlatformScenario.from_record(cfg["platform"])
    profile = load_cnn_cost_profile(cfg["cost_profile"], memory=memory)
    requirements = PPARequirements(**cfg["requirements"])

    def make_env(budget):
        options = dict(requirements=requirements, memory=memory, platform=platform,
                       cost_profile=profile, budget=budget, max_scope=cfg["max_scope"])
        if "candidate_compute_counts" in inspect.signature(GemminiCNNEnvironment).parameters:
            options.update(candidate_compute_counts=cfg["candidate_compute_counts"],
                           initial_count=cfg["initial_count"])
        elif cfg["candidate_compute_counts"] != list(range(1, 17)) or cfg["initial_count"] != 2:
            raise RuntimeError("historical environment supports only its recorded default counts and start")
        return GemminiCNNEnvironment(workload, **options)

    db = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)
    identity = unpacked(db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0])
    checkpoint = db.execute("SELECT value FROM metadata WHERE key='checkpoint'").fetchone()[0]
    after = RACETrainer.load(io.BytesIO(checkpoint), experiment_identity=identity)
    before = RACETrainer(seed=cfg["seed"], width=32, precision="float64", policy_backend="gemmini_cnn")
    baseline_session = next(x for x in report["sessions"] if x["phase"] == "baseline")
    baseline_hash, trained_hash = policy_hash(before.policy), policy_hash(after.policy)
    if baseline_hash != baseline_session["policy_hash_before"] or trained_hash != report["policy_hash"]:
        raise RuntimeError("policy hashes do not match source experiment")

    args.out.mkdir(parents=True, exist_ok=True)
    plan = {
        "purpose": "Selected example; independent seed follow-up, not a pre-registered population study",
        "selection": "Chosen after inspecting existing results because the balanced VGG16 case had improving archive and mean frozen evaluation; all new seeds are reported",
        "source_run": str(source), "source_model": args.model,
        "source_package": str(Path(race_rl.__file__).resolve()),
        "source_protocol_sha256": sha_file(protocol_path), "source_report_sha256": sha_file(report_path),
        "workload_sha256": sha_file(workload_path), "workload_identity": digest(workload),
        "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
        "script_sha256": sha_file(Path(__file__)), "source_code_hashes": protocol["source_hashes"],
        "training_seed": cfg["seed"], "training_episodes": after.episodes,
        "original_evaluation_seeds": cfg["evaluation_seeds"], "new_evaluation_seeds": args.seeds,
        "original_interaction_budget": cfg["interactions"], "new_interaction_budget_per_policy_per_seed": args.budget,
        "requirements": cfg["requirements"], "candidate_compute_counts": cfg["candidate_compute_counts"],
        "initial_count": cfg["initial_count"], "max_scope": cfg["max_scope"],
        "initial_policy_sha256": baseline_hash, "trained_policy_sha256": trained_hash,
        "policy_comparison": "same seeds and fixed proposal budget, frozen policy; training cost is separate",
        "limitations": ["one trained policy seed", "small held-out action-seed sample", "analytical proxy PPA, not physical measurements", "new budget differs from original evaluation"],
    }
    save_json(args.out / "protocol.json", plan)

    # Replay a prefix under the original budget, so budget-aware observations match.
    replay_seed = cfg["evaluation_seeds"][0]
    replay_results = []
    for phase, trainer in (("baseline", before), ("final", after)):
        raw = db.execute("SELECT interactions FROM episodes WHERE phase=? AND seed=? AND episode=1", (phase, replay_seed)).fetchone()
        if raw is None:
            raise RuntimeError("source replay trace missing")
        expected = unpacked(raw[0])
        env = make_env(cfg["interactions"])
        obs = env.reset()
        torch.manual_seed(replay_seed)
        np.random.seed(replay_seed)
        random.seed(replay_seed)
        for i in range(args.replay_steps):
            action, _, _ = trainer.policy.select(obs)
            obs, _, _ = env.step(action)
            actual = env.trace[-1]
            if actual["design_hash"] != expected[i]["design_hash"] or actual["ppa"] != expected[i]["ppa"]:
                raise RuntimeError(f"source trace replay differs for {phase}, step {i + 1}")
        replay_results.append({"phase": phase, "seed": replay_seed, "steps": args.replay_steps, "matched": True})
    db.close()
    save_json(args.out / "replay_check.json", replay_results)
    print("Source trace replay matched; starting all declared new seed pairs.", flush=True)

    rows = []
    start = time.perf_counter()
    for seed in args.seeds:
        for phase, trainer, expected_hash in (("before", before, baseline_hash), ("after", after, trained_hash)):
            env = make_env(args.budget)
            t0 = time.perf_counter()
            stats = frozen_episode(trainer, env, seed)
            if policy_hash(trainer.policy) != expected_hash or stats["parameter_updates"] != 0:
                raise RuntimeError("frozen policy changed during evaluation")
            row = {
                "seed": seed, "phase": phase, "assessment": env.scorer.assess(env.best_record),
                "best_compute_count": env.best_design.compute_count if env.best_design else None,
                "best_design": asdict(env.best_design) if env.best_design else None,
                "policy_hash": expected_hash, "stats": stats, "trace": env.trace,
                "wall_time_s": time.perf_counter() - t0,
            }
            rows.append(row)
            save_json(args.out / "results.json", {"status": "RUNNING", "rows": rows})
            a = row["assessment"]
            print(f"seed={seed} {phase}: {stats['oracle_requests']} proposals, "
                  f"cost={a['weighted_cost']}, feasible={a['feasible']}, {row['wall_time_s']:.1f}s", flush=True)
    save_json(args.out / "results.json", {"status": "COMPLETED", "rows": rows, "wall_time_s": time.perf_counter()-start})


if __name__ == "__main__":
    main()
