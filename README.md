# RACE-RL

Source code and experiment data for the paper:

**RACE-RL: Relation-Aware Coupled-Edit Reinforcement Learning for Preference-Aware Design Space Exploration of Chiplet-Based DNN Accelerators**

RACE-RL jointly searches compute-chiplet count, operator tiling and placement, and data residency for homogeneous chiplet-based DNN accelerators. A relational graph encoder feeds a conditional Joint Actor, which emits one complete Coupled-Edit action before each candidate evaluation. The policy is trained with REINFORCE.

> **PPA values are model predictions.** Power, latency, and area come from an analytical model that has not been calibrated against RTL or silicon. The 45 nm, 500 MHz, and 5 mm² per-chiplet parameters are declared assumptions or derived proxies; see `configs/ppa_profiles/`.

## Repository layout

| Path | Contents |
|---|---|
| `src/race_rl/` | Environment, relational encoder, Joint Actor, REINFORCE trainer, analytical PPA model |
| `tools/` | Experiment entry points, baseline adapters, analysis scripts |
| `configs/` | Hardware, memory, PPA profile, and model registry |
| `experiments/gemini_*_extension_20261104/` | Source snapshots with operator extensions for Inception-ResNet-v1, PNASNet-5-Large, and Transformer |
| `experiments/runs/*/requirements/` | Per-workload PPA limits used by the experiments |
| `third_party/` | Only the third-party files these experiments need; see `THIRD_PARTY_NOTICES.md` |
| `data/` | Compressed raw results for the paper |
| `scripts/` | Data extraction, result checks, and rerun commands |

The directory layout matches the one used to produce the results, because result files record paths and source hashes.

## Installation

Tested on macOS arm64 with Python 3.12. Other platforms may need different wheels, and floating-point results may differ.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src:tools:third_party
```

## Check the paper's numbers from the raw data

```sh
sh scripts/extract_data.sh
python scripts/reproduce_paper_numbers.py
```

The check recomputes the values reported for Fig. 5, Fig. 6, Fig. 7, Fig. 8, and the frozen-policy comparison, compares them with the manuscript, and exits non-zero on a mismatch.

## Data

| Archive | Paper section | Contents |
|---|---|---|
| `01_vgg16_training.tar.xz` | IV-D, Fig. 7 | VGG16 training (seed 7), interaction history in SQLite, frozen-policy evaluations |
| `02_coupled_edit_ablation.tar.xz` | IV-B, Fig. 5 | 960 trials with arms A (scope 1), B (dependent scope 2), C (random non-adjacent scope 2), and D (scope 4) |
| `03_preference_tradeoffs.tar.xz` | IV-C, Fig. 6, Table A.I | All 132 searches (11 workloads × 4 preferences × seeds 351–353) with calibration data |
| `04_search_baselines.tar.xz` | IV-E, Fig. 8 | ResNet18, AlexNet, and MobileNetV2 balanced runs for seeds 411–415 |

`04_search_baselines` also contains `results.json`, which summarizes every method and workload in that batch. The batch ran 15 methods and stopped at 2805 of 3300 runs. The paper reports five methods on three workloads.

## Rerun the experiments

Results are deterministic for a given seed on the tested platform.

| Paper | Command |
|---|---|
| Coupled-Edit scopes (Fig. 5) | `sh scripts/run_coupled_edit.sh` |
| Preference trade-offs (Fig. 6) | `sh scripts/run_preferences.sh` (requires extracted data) |
| VGG16 training (Fig. 7) | `sh scripts/run_vgg16_training.sh` |
| Search baselines (Fig. 8) | `sh scripts/run_baselines.sh` |

Runs write to `runs_repro/`. A full rerun takes many CPU hours.

`tools/common_dse_trial.py` also accepts methods the paper does not report. Those methods need third-party sources that this repository does not redistribute, so they fail when selected.

## Citation

Citation details will be added after publication.

## License

The original code and result data are licensed under the [Apache License 2.0](LICENSE). See `NOTICE` for attribution. Third-party files keep their own licenses, listed in `THIRD_PARTY_NOTICES.md`.
