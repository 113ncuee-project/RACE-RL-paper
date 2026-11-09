#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH=src:tools:third_party OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for model in resnet18 alexnet mobilenet_v2; do
  for seed in 411 412 413 414 415; do
    python -u tools/preference_seven_entry.py --entry trial --model "$model" --input-size 64 --model-seed 7 \
      --out "runs_repro/baselines/$model/$seed" --seed "$seed" --budget 64 --rl-episode-size 16 \
      --requirements "experiments/runs/fresh_all_models_dse_20261104/requirements/$model/balanced.json" \
      --profile configs/ppa_profiles/cnn_45nm_reference.json \
      --methods reinforce local_random sa gemini_common_restricted bayesopt_gp_ei
  done
done
