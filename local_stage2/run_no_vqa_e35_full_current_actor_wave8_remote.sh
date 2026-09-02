#!/usr/bin/env bash

# Matched Wave-8 ablation for the 100%-coverage current-frame actor target.
# The frozen No-VQA generator and current four-camera token cache are identical
# to Wave-6/7.  Only the training-only actor supervision source changes from
# the legacy sparse Gate-C table to the audited 103,288-scene current-frame
# table.  No actor target is available to inference.

set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
python_bin="${DRIVEVLA_PYTHON:-/mnt/project/DriveVLA-M0-env/bin/python}"
source_root="${NO_VQA_SOURCE_ROOT:-/root/scorer_pdms93_cache/no_vqa_e35_features_full_v1}"
label_root="${NO_VQA_LABEL_ROOT:-/root/scorer_pdms93_cache/no_vqa_e35_labels_full_v1}"
private_root="${NO_VQA_MULTIVIEW_TRAIN_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_multiview_trainval_pool2_tiles4_v1_8shard}"
actor_root="${NO_VQA_FULL_CURRENT_ACTOR_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_full_current_actor_targets_v1}"
run_root="${NO_VQA_WAVE8_RUN_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_full_current_actor_wave8_v1}"
log_root="${NO_VQA_WAVE8_LOG_ROOT:-/root/scorer_pdms93_logs/no_vqa_e35_full_current_actor_wave8_v1}"
split_manifest="${NO_VQA_SPLIT_MANIFEST:-${repo_root}/reports/scorer_pdms93/OFFICIAL_SCORER_SPLIT.json}"
poll_seconds="${NO_VQA_WAVE8_POLL_SECONDS:-30}"

# GPU 0 is occupied. GPU 7 is intentionally left to pre-existing shared-future
# watchers on rl-zt4.  This campaign never preempts another task.
gpu_ids=(1 2 3 4 5 6)
names=(
  fullactor_rawcombined_top16_reference_q50_strict_seed2
  fullactor_rawcombined_top32_reference_q50_strict_seed2
  fullactor_rawprivate_top16_reference_q50_strict_seed2
  fullactor_rawcontextcombined_top16_reference_q50_strict_seed2
  fullactor_rawpointcombined_top16_reference_q50_strict_seed2
  fullactor_rawpointcombined_top32_reference_q50_strict_seed2
)

for path in "${source_root}" "${label_root}" "${split_manifest}"; do
  [[ -e "${path}" ]] || { echo "missing Wave-8 input: ${path}" >&2; exit 2; }
done
for root in "${private_root}" "${actor_root}"; do
  [[ -f "${root}/.complete" ]] || { echo "incomplete Wave-8 cache: ${root}" >&2; exit 2; }
done
if [[ -e "${run_root}" || -e "${log_root}" ]]; then
  echo "Wave-8 output already exists; refusing overwrite" >&2
  exit 2
fi

while true; do
  mapfile -t gpu_memory < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  ready=1
  for gpu in "${gpu_ids[@]}"; do
    used="${gpu_memory[${gpu}]//[[:space:]]/}"
    if [[ ! "${used}" =~ ^[0-9]+$ ]] || (( used > 1024 )); then
      ready=0
    fi
  done
  (( ready == 1 )) && break
  echo "NO_VQA_WAVE8 waiting_for_gpus utc=$(date -u +%FT%TZ)"
  sleep "${poll_seconds}"
done

mkdir -p "${run_root}" "${log_root}"
export PYTHONPATH="${repo_root}:${repo_root}/nuplan-devkit${PYTHONPATH:+:${PYTHONPATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

common_args=(
  --source no_vqa_e35 "${source_root}" "${label_root}"
  --private-observation-root "${private_root}"
  --split-manifest "${split_manifest}"
  --selection-source no_vqa_e35
  --epochs 8
  --batch-size 32
  --eval-batch-size 64
  --num-workers 0
  --learning-rate 3e-4
  --weight-decay 1e-4
  --bootstrap-replicates 1000
  --model-dim 256
  --dynamic-queries 16
  --private-layers 2
  --trajectory-layers 2
  --candidate-layers 1
  --fine-layers 2
  --private-fine-top-k 16
  --residual-layers 2
  --max-residual 0.5
  --minimum-pair-delta 0.02
  --factor-rank-minimum-delta 0.05
  --score-mode hybrid
  --seed 2
  --conservative-reference
  --reference-hidden-dim 512
  --reference-layers 2
  --reference-gain-quantile-index 1
  --reference-minimum-lcb-gain 0
  --reference-maximum-safety-worse-probability 0.1
  --reference-minimum-safe-improvement-probability 0.7
  --reference-weight 1
  --reference-quantile-weight 1
  --reference-median-rank-weight 0.25
  --reference-safety-weight 1
  --reference-improvement-weight 0.5
  --reference-false-switch-weight 0.5
  --reference-missed-improvement-weight 0
  --reference-safety-worse-positive-weight 10
  --reference-safe-improvement-positive-weight 3
  --reference-switch-margin-temperature 0.05
  --reference-minimum-improvement-target 0.005
  --reference-factor-epsilon 1e-6
  --pairwise-weight 0
  --base-pairwise-weight 0
  --listwise-weight 0
  --top-set-weight 0
  --expected-regret-weight 0
  --top-regret-weight 0
  --factor-weight 0
  --private-factor-weight 0.25
  --factor-rank-weight 0
  --relative-safety-weight 0
  --residual-l2-weight 0
  --safety-negative-weight 1
  --factor-loss-scope topk
  --current-actor-target-root "${actor_root}"
  --current-actor-weight 0.5
)

pids=()
for index in "${!names[@]}"; do
  gpu="${gpu_ids[${index}]}"
  name="${names[${index}]}"
  variant_args=()
  case "${name}" in
    fullactor_rawcombined_top16_reference_q50_strict_seed2)
      variant_args+=(--m0-candidate-fusion --residual-top-k 16)
      ;;
    fullactor_rawcombined_top32_reference_q50_strict_seed2)
      variant_args+=(--m0-candidate-fusion --residual-top-k 32)
      ;;
    fullactor_rawprivate_top16_reference_q50_strict_seed2)
      variant_args+=(--residual-top-k 16)
      ;;
    fullactor_rawcontextcombined_top16_reference_q50_strict_seed2)
      variant_args+=(--m0-context-fusion --m0-candidate-fusion --residual-top-k 16)
      ;;
    fullactor_rawpointcombined_top16_reference_q50_strict_seed2)
      variant_args+=(--trajectory-observation-attention --m0-candidate-fusion --residual-top-k 16)
      ;;
    fullactor_rawpointcombined_top32_reference_q50_strict_seed2)
      variant_args+=(--trajectory-observation-attention --m0-candidate-fusion --residual-top-k 32)
      ;;
    *)
      echo "unhandled Wave-8 variant: ${name}" >&2
      exit 2
      ;;
  esac
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    exec "${python_bin}" "${repo_root}/local_stage2/train_m0_private_residual_scorer.py" \
      "${common_args[@]}" "${variant_args[@]}" \
      --output-dir "${run_root}/${name}"
  ) >"${log_root}/${name}.log" 2>&1 &
  pids+=("$!")
  echo "NO_VQA_WAVE8_TRAIN_STARTED gpu=${gpu} name=${name} pid=$!"
done

failure=0
for index in "${!pids[@]}"; do
  if wait "${pids[${index}]}"; then
    echo "NO_VQA_WAVE8_TRAIN_COMPLETE name=${names[${index}]}"
  else
    echo "NO_VQA_WAVE8_TRAIN_FAILED name=${names[${index}]}" >&2
    failure=1
  fi
done
(( failure == 0 )) || exit 1
touch "${run_root}/.wave8_complete"
echo "NO_VQA_WAVE8_COMPLETE run_root=${run_root}"
