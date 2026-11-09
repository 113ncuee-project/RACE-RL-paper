#!/bin/sh
# Requires scripts/extract_data.sh: the studies read earlier candidates for scale calibration.
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH=src:tools:third_party OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python tools/preference_calibrated_study.py --out runs_repro/preference_calibrated
python tools/preference_additional_models.py --out runs_repro/preference_additional
python tools/preference_resnext_study.py --out runs_repro/preference_resnext
python tools/preference_inception_study.py --out runs_repro/preference_inception
python tools/preference_pnas_study.py --out runs_repro/preference_pnas
python tools/preference_transformer_study.py --out runs_repro/preference_transformer
