#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH=src:tools:third_party OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python tools/run_cnn_hierarchy.py --train --models vgg16 --input-size 64 --episodes "${EPISODES:-60}" --interactions 64 --max-scope 2 --seed 7 \
  --evaluation-seeds 17 29 43 --ppa-profile configs/ppa_profiles/cnn_45nm_reference.json \
  --requirements experiments/runs/common_dse_study_v2/requirements/vgg16/balanced.json --out "${OUT:-runs_repro/vgg16_training}"
