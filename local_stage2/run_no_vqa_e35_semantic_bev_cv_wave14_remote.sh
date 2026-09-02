#!/usr/bin/env bash

# Wave-14: train one locked semantic-BEV scorer-private model on five
# disjoint physical-log folds. The M0 proposal/VLM checkpoint stays frozen.

set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
python_bin="${DRIVEVLA_PYTHON:-/mnt/project/DriveVLA-M0-env/bin/python}"
source_root="${NO_VQA_SOURCE_ROOT:-/root/scorer_pdms93_cache/no_vqa_e35_features_full_v1}"
label_root="${NO_VQA_LABEL_ROOT:-/root/scorer_pdms93_cache/no_vqa_e35_labels_full_v1}"
private_root="${NO_VQA_DENSE_TRAIN_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_multiview_trainval_pool4_tiles4_v1_8shard}"
actor_root="${NO_VQA_FULL_CURRENT_ACTOR_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_full_current_actor_targets_v1}"
semantic_root="${NO_VQA_SEMANTIC_BEV_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_full_current_semantic_bev_targets_v1}"
fold_root="${NO_VQA_WAVE14_FOLD_ROOT:-${repo_root}/reports/m0_independent_scorer_representation/no_vqa_e35_risk_cv_wave12_v1/folds}"
run_root="${NO_VQA_WAVE14_RUN_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_semantic_bev_cv_wave14_v1}"
log_root="${NO_VQA_WAVE14_LOG_ROOT:-/root/scorer_pdms93_logs/no_vqa_e35_semantic_bev_cv_wave14_v1}"
poll_seconds="${NO_VQA_WAVE14_POLL_SECONDS:-30}"

gpu_ids=(0 1 2 3 4)
for path in "${source_root}" "${label_root}" "${fold_root}/index.json"; do
  [[ -e "${path}" ]] || { echo "missing Wave-14 input: ${path}" >&2; exit 2; }
done
for fold in 0 1 2 3 4; do
  [[ -f "${fold_root}/fold_${fold}.json" ]] || {
    echo "missing Wave-14 fold ${fold}" >&2
    exit 2
  }
done
for root in "${private_root}" "${actor_root}" "${semantic_root}"; do
  [[ -f "${root}/.complete" ]] || { echo "incomplete Wave-14 cache: ${root}" >&2; exit 2; }
done
if [[ -e "${run_root}" || -e "${log_root}" ]]; then
  echo "Wave-14 output already exists; refusing overwrite" >&2
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
  echo "NO_VQA_WAVE14 waiting_for_gpus utc=$(date -u +%FT%TZ)"
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
  --selection-source no_vqa_e35
  --epochs 8
  --save-last-artifact
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
  --trajectory-observation-attention
  --semantic-bev-target-root "${semantic_root}"
  --semantic-bev-weight 0.5
  --semantic-bev-fusion
  --candidate-layers 1
  --fine-layers 2
  --private-fine-top-k 16
  --residual-layers 2
  --max-residual 0.5
  --minimum-pair-delta 0.02
  --factor-rank-minimum-delta 0.05
  --score-mode hybrid
  --seed 2
  --scene-sampling-mode risk_balanced
  --risk-scene-max-multiplier 4
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
  --m0-candidate-fusion
  --residual-top-k 32
)

pids=()
for fold in 0 1 2 3 4; do
  gpu="${gpu_ids[${fold}]}"
  output="${run_root}/fold_${fold}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    exec "${python_bin}" "${repo_root}/local_stage2/train_m0_private_residual_scorer.py" \
      "${common_args[@]}" \
      --split-manifest "${fold_root}/fold_${fold}.json" \
      --output-dir "${output}"
  ) >"${log_root}/fold_${fold}.log" 2>&1 &
  pids+=("$!")
  echo "NO_VQA_WAVE14_FOLD_STARTED fold=${fold} gpu=${gpu} pid=$!"
done

failure=0
for fold in 0 1 2 3 4; do
  if wait "${pids[${fold}]}"; then
    echo "NO_VQA_WAVE14_FOLD_COMPLETE fold=${fold}"
  else
    echo "NO_VQA_WAVE14_FOLD_FAILED fold=${fold}" >&2
    failure=1
  fi
done
(( failure == 0 )) || exit 1
touch "${run_root}/.wave12_folds_complete"
echo "NO_VQA_WAVE14_FOLDS_COMPLETE root=${run_root}"
