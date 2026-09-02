#!/usr/bin/env bash

# Refit three predeclared Wave-5 conservative-reference variants on every
# legal trainval physical log.  The selected epoch and all architecture/loss
# settings are reconstructed from each held-out-log-selected artifact.

set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
python_bin="${DRIVEVLA_PYTHON:-/mnt/project/DriveVLA-M0-env/bin/python}"
selection_root="${NO_VQA_WAVE5_SELECTION_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_wave5_reference_selection_v1}"
run_root="${NO_VQA_REFIT_RUN_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_all_log_refit_wave5_reference_v1}"
log_root="${NO_VQA_REFIT_LOG_ROOT:-/root/scorer_pdms93_logs/no_vqa_e35_all_log_refit_wave5_reference_v1}"
poll_seconds="${NO_VQA_REFIT_POLL_SECONDS:-30}"

names=(
  combined_top8_reference_q50_strict_actor_all_logs
  combined_top16_reference_q50_strict_actor_all_logs
  combined_top16_reference_q10_balanced_actor_all_logs
)
selection_names=(
  combined_top8_reference_q50_strict_actor_seed2
  combined_top16_reference_q50_strict_actor_seed2
  combined_top16_reference_q10_balanced_actor_seed2
)
gpus=(1 2 3)

for selection_name in "${selection_names[@]}"; do
  selection="${selection_root}/${selection_name}.pt"
  while [[ ! -f "${selection}" ]]; do
    echo "M0_WAVE5_ALL_LOG_REFIT waiting_for_selection=${selection_name} utc=$(date -u +%FT%TZ)"
    sleep "${poll_seconds}"
  done
done
if [[ -e "${run_root}" || -e "${log_root}" ]]; then
  echo "refusing existing Wave-5 all-log refit output" >&2
  exit 2
fi

while true; do
  mapfile -t gpu_memory < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  )
  ready=1
  for gpu in "${gpus[@]}"; do
    used="${gpu_memory[${gpu}]//[[:space:]]/}"
    if [[ ! "${used}" =~ ^[0-9]+$ ]] || (( used > 1024 )); then
      ready=0
    fi
  done
  (( ready == 1 )) && break
  echo "M0_WAVE5_ALL_LOG_REFIT waiting_for_gpus utc=$(date -u +%FT%TZ)"
  sleep "${poll_seconds}"
done

mkdir -p "${run_root}" "${log_root}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${repo_root}:${repo_root}/nuplan-devkit${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

pids=()
for index in "${!names[@]}"; do
  name="${names[${index}]}"
  selection="${selection_root}/${selection_names[${index}]}.pt"
  (
    export CUDA_VISIBLE_DEVICES="${gpus[${index}]}"
    exec "${python_bin}" "${repo_root}/local_stage2/refit_m0_private_residual_scorer.py" \
      --selection-artifact "${selection}" \
      --output-dir "${run_root}/${name}" \
      --device cuda
  ) >"${log_root}/${name}.log" 2>&1 &
  pids+=("$!")
  echo "M0_WAVE5_ALL_LOG_REFIT_STARTED gpu=${gpus[${index}]} name=${name} pid=$!"
done

failure=0
for index in "${!pids[@]}"; do
  name="${names[${index}]}"
  if wait "${pids[${index}]}" && \
    [[ -f "${run_root}/${name}/refit_m0_private_residual_scorer.pt" ]]; then
    echo "M0_WAVE5_ALL_LOG_REFIT_COMPLETE name=${name}"
  else
    echo "M0_WAVE5_ALL_LOG_REFIT_FAILED name=${name}" >&2
    failure=1
  fi
done
(( failure == 0 )) || exit 1
touch "${run_root}/.refit_complete"
echo "M0_WAVE5_ALL_LOG_REFIT_CAMPAIGN_COMPLETE root=${run_root}"
