#!/usr/bin/env bash

# Wave-13 single-variable replay: use 4x4-per-crop current visual tokens while
# keeping the Wave-12 folds, scorer, optimization and fixed stop epoch intact.

set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
private_root="${NO_VQA_DENSE_TRAIN_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_multiview_trainval_pool4_tiles4_v1_8shard}"
strict_python=/mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/envs/navsim_py39_exact/bin/python
export DRIVEVLA_PYTHON="${DRIVEVLA_PYTHON:-${strict_python}}"
export PYTHONPATH="/mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/envs/transformers_4_48_3:/mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/envs/lightning_2_2_1:${repo_root}:${repo_root}/nuplan-devkit:/mnt/project/DriveVLA-M0-env/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
[[ -x "${DRIVEVLA_PYTHON}" ]] || {
  echo "missing strict Python runtime: ${DRIVEVLA_PYTHON}" >&2
  exit 2
}
[[ "$("${DRIVEVLA_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "3.9" ]] || {
  echo "Wave-13 requires Python 3.9" >&2
  exit 2
}
[[ -f "${private_root}/.complete" ]] || {
  echo "incomplete Wave-13 dense train cache: ${private_root}" >&2
  exit 2
}

export REPO_ROOT="${repo_root}"
export NO_VQA_MULTIVIEW_TRAIN_ROOT="${private_root}"
export NO_VQA_WAVE12_FOLD_ROOT="${NO_VQA_WAVE13_FOLD_ROOT:-${repo_root}/reports/m0_independent_scorer_representation/no_vqa_e35_risk_cv_wave12_v1/folds}"
export NO_VQA_WAVE12_RUN_ROOT="${NO_VQA_WAVE13_RUN_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_dense_risk_cv_wave13_v2}"
export NO_VQA_WAVE12_LOG_ROOT="${NO_VQA_WAVE13_LOG_ROOT:-/root/scorer_pdms93_logs/no_vqa_e35_dense_risk_cv_wave13_v2}"

exec bash "${repo_root}/local_stage2/run_no_vqa_e35_risk_cv_wave12_remote.sh"
