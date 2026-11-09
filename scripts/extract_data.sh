#!/bin/sh
# Extract the result archives and create the paths the preference studies expect.
set -eu
cd "$(dirname "$0")/.."
for archive in data/*.tar.xz; do tar -xJf "$archive"; done
D="../paper_sources/RACE-RL_v20261106-0100/D_十一模型偏好"
for study in preference_calibrated_v1_20261104 preference_additional_four_20261104 preference_resnext_20261104 preference_inception_20261104 preference_pnas_20261104 preference_transformer_20261104; do
  ln -sfn "$D/$study" "experiments/runs/$study"
done
mkdir -p experiments/runs/preference_tradeoff_validation_20261031
ln -sfn "../$D/calibration_source_reference_v2" experiments/runs/preference_tradeoff_validation_20261031/reference_v2
