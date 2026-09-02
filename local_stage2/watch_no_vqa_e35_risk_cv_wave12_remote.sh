#!/usr/bin/env bash

# Wait for all five fixed-epoch folds, enforce the predeclared robust gate,
# then (only on PASS) refit all Navtrain logs and run strict complete Navtest.

set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
python_bin="${DRIVEVLA_PYTHON:-/mnt/project/DriveVLA-M0-env/bin/python}"
run_root="${NO_VQA_WAVE12_RUN_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_risk_cv_wave12_v1}"
fold_root="${NO_VQA_WAVE12_FOLD_ROOT:-${repo_root}/reports/m0_independent_scorer_representation/no_vqa_e35_risk_cv_wave12_v1/folds}"
report_root="${NO_VQA_WAVE12_REPORT_ROOT:-${repo_root}/reports/m0_independent_scorer_representation/no_vqa_e35_risk_cv_wave12_v1}"
log_root="${NO_VQA_WAVE12_POST_LOG_ROOT:-/root/scorer_pdms93_logs/no_vqa_e35_risk_cv_wave12_post_v1}"
selection_artifact="${NO_VQA_WAVE12_SELECTION_ARTIFACT:-/root/scorer_pdms93_artifacts/no_vqa_e35_risk_cv_wave12_v1/CV_SELECTION_ARTIFACT.pt}"
refit_root="${NO_VQA_WAVE12_REFIT_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_risk_cv_wave12_all_log_refit_v1}"
campaign="${NO_VQA_WAVE12_REFIT_CAMPAIGN:-no_vqa_e35_risk_cv_wave12_all_log_refit_v1}"
private_navtest_root="${NO_VQA_MULTIVIEW_TEST_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_multiview_navtest_pool2_tiles4_v1_8shard}"
gpu="${NO_VQA_WAVE12_POST_GPU:-0}"

mkdir -p "${log_root}" "$(dirname "${selection_artifact}")"
until [[ -f "${run_root}/.wave12_folds_complete" ]]; do
  echo "NO_VQA_WAVE12_POST waiting_for_folds utc=$(date -u +%FT%TZ)"
  sleep 30
done
# Shared summaries are atomically replaced at the end of every epoch. Require
# a stable second read before constructing immutable CV evidence.
sleep 30
for fold in 0 1 2 3 4; do
  "${python_bin}" - "${run_root}/fold_${fold}/training_summary.json" <<'PY'
import json, sys
payload=json.load(open(sys.argv[1]))
assert len(payload.get("history", [])) == 8
assert payload.get("last_artifact_epoch") == 7
PY
done

"${python_bin}" "${repo_root}/local_stage2/summarize_m0_residual_cv.py" \
  --run-root "${run_root}" \
  --fold-root "${fold_root}" \
  --locked-epoch 7 \
  --output-json "${report_root}/CV_RESULTS.json" \
  --output-md "${report_root}/CV_RESULTS.md" \
  --selection-artifact "${selection_artifact}" \
  >"${log_root}/cv_summary.log" 2>&1

gate="$("${python_bin}" - "${report_root}/CV_RESULTS.json" <<'PY'
import json, sys
print("1" if json.load(open(sys.argv[1]))["aggregate"]["robust_refit_gate_passed"] else "0")
PY
)"
if [[ "${gate}" != "1" ]]; then
  echo "NO_VQA_WAVE12_ROBUST_GATE_FAIL report=${report_root}/CV_RESULTS.md"
  exit 0
fi
[[ -f "${selection_artifact}" ]] || { echo "missing CV selection artifact" >&2; exit 2; }

while true; do
  used="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
  [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= 1024 )) && break
  echo "NO_VQA_WAVE12_POST waiting_for_gpu=${gpu} utc=$(date -u +%FT%TZ)"
  sleep 30
done
export CUDA_VISIBLE_DEVICES="${gpu}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${repo_root}:${repo_root}/nuplan-devkit${PYTHONPATH:+:${PYTHONPATH}}"
"${python_bin}" "${repo_root}/local_stage2/refit_m0_private_residual_scorer.py" \
  --selection-artifact "${selection_artifact}" \
  --output-dir "${refit_root}" \
  --device cuda \
  >"${log_root}/all_log_refit.log" 2>&1

export M0_SINGLE_REFIT_PRIVATE_OBSERVATION_ROOT="${private_navtest_root}"
exec bash "${repo_root}/local_stage2/watch_single_m0_all_log_refit_navtest.sh" \
  "${refit_root}" "${campaign}" "${gpu}"
