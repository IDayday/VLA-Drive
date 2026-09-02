#!/usr/bin/env bash

# Refit three validation-selected No-VQA scorer families on all 103,288
# legal trainval scenes.  No validation or Navtest value is read during refit.

set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
python_bin="${DRIVEVLA_PYTHON:-/mnt/project/DriveVLA-M0-env/bin/python}"
selection_root="${NO_VQA_REFIT_SELECTION_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_scene_token_wave3_calibrated_v1}"
run_root="${NO_VQA_REFIT_RUN_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_all_log_refit_wave3_v1}"
log_root="${NO_VQA_REFIT_LOG_ROOT:-/root/scorer_pdms93_logs/no_vqa_e35_all_log_refit_wave3_v1}"

names=(
  candidate_only_top16_factor_safety5_all_logs
  combined_top16_hybrid_safety5_all_logs
  factorized_top16_cv_hybrid_safety5_all_logs
)
selection_names=(
  candidate_only_top16_factor_safety5_seed2__calibrated
  combined_top16_hybrid_safety5_seed2__calibrated
  factorized_top16_cv_hybrid_safety5_seed2__calibrated
)
gpus=(1 3 5)

if [[ -e "${run_root}" || -e "${log_root}" ]]; then
  echo "refusing existing all-log refit output: ${run_root} or ${log_root}" >&2
  exit 2
fi
mkdir -p "${run_root}" "${log_root}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${repo_root}:${repo_root}/nuplan-devkit${PYTHONPATH:+:${PYTHONPATH}}"

pids=()
for index in "${!names[@]}"; do
  name="${names[${index}]}"
  selection="${selection_root}/${selection_names[${index}]}/best_m0_private_residual_scorer.pt"
  output="${run_root}/${name}"
  [[ -f "${selection}" ]] || { echo "missing selection artifact: ${selection}" >&2; exit 2; }
  (
    export CUDA_VISIBLE_DEVICES="${gpus[${index}]}"
    exec "${python_bin}" "${repo_root}/local_stage2/refit_m0_private_residual_scorer.py" \
      --selection-artifact "${selection}" \
      --output-dir "${output}" \
      --device cuda
  ) >"${log_root}/${name}.log" 2>&1 &
  pids+=("$!")
  echo "M0_ALL_LOG_REFIT_STARTED gpu=${gpus[${index}]} name=${name} pid=$!"
done

failure=0
for index in "${!pids[@]}"; do
  name="${names[${index}]}"
  if wait "${pids[${index}]}"; then
    [[ -f "${run_root}/${name}/refit_m0_private_residual_scorer.pt" ]] || failure=1
    echo "M0_ALL_LOG_REFIT_COMPLETE name=${name}"
  else
    status=$?
    echo "M0_ALL_LOG_REFIT_FAILED name=${name} status=${status}" >&2
    failure=1
  fi
done
(( failure == 0 )) || exit 1
touch "${run_root}/.refit_complete"
echo "M0_ALL_LOG_REFIT_CAMPAIGN_COMPLETE root=${run_root}"
