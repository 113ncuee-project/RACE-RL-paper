#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH=src:tools:third_party OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for model in resnet18 vgg16 mobilenet_v2; do
  python tools/coupled_edit_ablation.py run --model "$model" --out runs_repro/coupled_edit
done
