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
fixed_policy_artifact="${NO_VQA_WAVE12_FIXED_POLICY_ARTIFACT:-/root/scorer_pdms93_artifacts/no_vqa_e35_risk_cv_wave12_v1/FIXED_POLICY_DIAGNOSTIC_ARTIFACT.pt}"
sweep_root="${NO_VQA_WAVE12_SWEEP_ROOT:-${run_root}/common_policy_sweeps}"
refit_root="${NO_VQA_WAVE12_REFIT_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_risk_cv_wave12_all_log_refit_v1}"
campaign="${NO_VQA_WAVE12_REFIT_CAMPAIGN:-no_vqa_e35_risk_cv_wave12_all_log_refit_v1}"
private_navtest_root="${NO_VQA_MULTIVIEW_TEST_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_multiview_navtest_pool2_tiles4_v1_8shard}"
gpu="${NO_VQA_WAVE12_POST_GPU:-0}"
source_root="${NO_VQA_SOURCE_ROOT:-/root/scorer_pdms93_cache/no_vqa_e35_features_full_v1}"
label_root="${NO_VQA_LABEL_ROOT:-/root/scorer_pdms93_cache/no_vqa_e35_labels_full_v1}"
read -r -a sweep_gpu_ids <<<"${NO_VQA_WAVE12_SWEEP_GPU_IDS:-0 1 2 3 4}"
sweep_max_parallel="${NO_VQA_WAVE12_SWEEP_MAX_PARALLEL:-${#sweep_gpu_ids[@]}}"

(( ${#sweep_gpu_ids[@]} > 0 )) || { echo "empty sweep GPU list" >&2; exit 2; }
[[ "${sweep_max_parallel}" =~ ^[1-9][0-9]*$ ]] || {
  echo "invalid sweep parallelism: ${sweep_max_parallel}" >&2
  exit 2
}
(( sweep_max_parallel <= ${#sweep_gpu_ids[@]} )) || {
  echo "sweep parallelism exceeds distinct GPU count" >&2
  exit 2
}
for path in "${source_root}" "${label_root}"; do
  [[ -e "${path}" ]] || { echo "missing scorer cache: ${path}" >&2; exit 2; }
done

# Do not create ``sweep_root`` here when it is nested under ``run_root``.
# The fold launcher intentionally refuses any pre-existing run directory, and
# starting this watcher before the launcher used to create an empty
# ``run_root/common_policy_sweeps`` directory that made every fold abort.
mkdir -p "${log_root}" "$(dirname "${selection_artifact}")"
export PYTHONPATH="${repo_root}:${repo_root}/nuplan-devkit${PYTHONPATH:+:${PYTHONPATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
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

mkdir -p "${sweep_root}"

if [[ ! -f "${report_root}/FIXED_POLICY_CV_RESULTS.json" ]]; then
  "${python_bin}" "${repo_root}/local_stage2/summarize_m0_residual_cv.py" \
    --run-root "${run_root}" \
    --fold-root "${fold_root}" \
    --locked-epoch 7 \
    --output-json "${report_root}/FIXED_POLICY_CV_RESULTS.json" \
    --output-md "${report_root}/FIXED_POLICY_CV_RESULTS.md" \
    --selection-artifact "${fixed_policy_artifact}" \
    >"${log_root}/fixed_policy_cv_summary.log" 2>&1
fi

# Freeze every epoch-7 model, sweep the exact same predeclared deployment grid
# independently on each held-out fold, then select one policy jointly.  Five
# GPUs only shorten evaluation wall time; no fold receives another fold's
# neural weights and Navtest is not available to either program.
sweep_pids=()
sweep_launch_index=0
for fold in 0 1 2 3 4; do
  output="${sweep_root}/fold_${fold}.json"
  [[ -f "${output}" ]] && continue
  gpu_slot="${sweep_gpu_ids[$((sweep_launch_index % ${#sweep_gpu_ids[@]}))]}"
  while true; do
    used="$(nvidia-smi --id="${gpu_slot}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
    [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= 1024 )) && break
    echo "NO_VQA_WAVE12_POST waiting_for_sweep_gpu=${gpu_slot} utc=$(date -u +%FT%TZ)"
    sleep 30
  done
  CUDA_VISIBLE_DEVICES="${gpu_slot}" "${python_bin}" \
    "${repo_root}/local_stage2/sweep_m0_conservative_reference_fold.py" \
    --artifact "${run_root}/fold_${fold}/last_m0_private_residual_scorer.pt" \
    --source no_vqa_e35 \
      "${source_root}" \
      "${label_root}" \
    --private-observation-root \
      "${NO_VQA_WAVE12_PRIVATE_TRAIN_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_multiview_trainval_pool2_tiles4_v1_8shard}" \
    --split-manifest "${fold_root}/fold_${fold}.json" \
    --output "${output}" \
    --eval-batch-size 64 \
    --device cuda \
    >"${log_root}/common_policy_fold_${fold}.log" 2>&1 &
  sweep_pids+=("$!")
  sweep_launch_index=$((sweep_launch_index + 1))
  if (( ${#sweep_pids[@]} == sweep_max_parallel )); then
    for pid in "${sweep_pids[@]}"; do
      wait "${pid}"
    done
    sweep_pids=()
  fi
done
for pid in "${sweep_pids[@]}"; do
  wait "${pid}"
done

fold_result_args=()
for fold in 0 1 2 3 4; do
  fold_result_args+=(--fold-result "${sweep_root}/fold_${fold}.json")
done
if [[ ! -f "${report_root}/CV_RESULTS.json" ]]; then
  "${python_bin}" \
    "${repo_root}/local_stage2/summarize_m0_conservative_reference_cv.py" \
    "${fold_result_args[@]}" \
    --output-json "${report_root}/CV_RESULTS.json" \
    --output-md "${report_root}/CV_RESULTS.md" \
    --selection-artifact "${selection_artifact}" \
    --bootstrap-replicates 10000 \
    --bootstrap-seed 20260902 \
    --safety-tolerance 0.0005 \
    >"${log_root}/common_policy_cv_summary.log" 2>&1
fi

gate="$("${python_bin}" - "${report_root}/CV_RESULTS.json" <<'PY'
import json, sys
print("1" if json.load(open(sys.argv[1]))["robust_refit_gate_passed"] else "0")
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
"${python_bin}" "${repo_root}/local_stage2/refit_m0_private_residual_scorer.py" \
  --selection-artifact "${selection_artifact}" \
  --output-dir "${refit_root}" \
  --device cuda \
  >"${log_root}/all_log_refit.log" 2>&1

export M0_SINGLE_REFIT_PRIVATE_OBSERVATION_ROOT="${private_navtest_root}"
exec bash "${repo_root}/local_stage2/watch_single_m0_all_log_refit_navtest.sh" \
  "${refit_root}" "${campaign}" "${gpu}"
