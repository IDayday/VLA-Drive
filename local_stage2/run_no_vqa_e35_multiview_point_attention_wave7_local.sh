#!/usr/bin/env bash

# Predeclared path-local scorer representation wave.  Every proposal waypoint
# queries the same uncompressed current CAM_F0/L0/R0/B0 token memory.  The
# underlying No-VQA generator remains frozen and no future/evaluator tensor is
# available to the forward path.  Variants are locked before Wave-6 or Wave-7
# validation/Navtest results are read.

set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
python_bin="${DRIVEVLA_PYTHON:-/mnt/project/DriveVLA-M0-env/bin/python}"
source_root="${NO_VQA_SOURCE_ROOT:-/root/scorer_pdms93_cache/no_vqa_e35_features_full_v1}"
label_root="${NO_VQA_LABEL_ROOT:-/root/scorer_pdms93_cache/no_vqa_e35_labels_full_v1}"
private_root="${NO_VQA_MULTIVIEW_TRAIN_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_multiview_trainval_pool2_tiles4_v1_8shard}"
run_root="${NO_VQA_WAVE7_RUN_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_multiview_point_attention_wave7_v1}"
log_root="${NO_VQA_WAVE7_LOG_ROOT:-/root/scorer_pdms93_logs/no_vqa_e35_multiview_point_attention_wave7_v1}"
split_manifest="${NO_VQA_SPLIT_MANIFEST:-${repo_root}/reports/scorer_pdms93/OFFICIAL_SCORER_SPLIT.json}"
current_actor_root="${NO_VQA_CURRENT_ACTOR_ROOT:-/mnt/project/DriveVLA-M0-gate-c/outputs/shared_future_candidate_consequence_gate_c/all/oracle_store}"
poll_seconds="${NO_VQA_WAVE7_POLL_SECONDS:-30}"

# GPU 0 is intentionally excluded because a pre-existing DrivOR diagnostic owns
# it.  This experiment never stops, replaces, or shares that task.
gpu_ids=(1 2 3 4 5 6 7)
names=(
  rawpointcombined_top16_hybrid_standard_actor_seed2
  rawpointcombined_top8_hybrid_topregret_actor_seed2
  rawpointcombined_top16_reference_q50_strict_actor_seed2
  rawpointcombined_top32_reference_q50_strict_actor_seed2
  rawpointprivate_top16_reference_q50_strict_actor_seed2
  rawpointcontextcombined_top16_reference_q50_strict_actor_seed2
  rawpointcombined_top16_reference_q50_balanced_actor_seed2
)

for path in "${source_root}" "${label_root}" "${split_manifest}" "${current_actor_root}"; do
  [[ -e "${path}" ]] || { echo "missing wave-7 input: ${path}" >&2; exit 2; }
done
while [[ ! -f "${private_root}/.complete" ]]; do
  echo "NO_VQA_WAVE7 waiting_for_trainval_multiview_cache utc=$(date -u +%FT%TZ)"
  sleep "${poll_seconds}"
done
if [[ -e "${run_root}" || -e "${log_root}" ]]; then
  echo "wave-7 output already exists; refusing overwrite" >&2
  exit 2
fi

while true; do
  mapfile -t gpu_memory < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  )
  ready=1
  for gpu in "${gpu_ids[@]}"; do
    used="${gpu_memory[${gpu}]//[[:space:]]/}"
    if [[ ! "${used}" =~ ^[0-9]+$ ]] || (( used > 1024 )); then
      ready=0
    fi
  done
  (( ready == 1 )) && break
  echo "NO_VQA_WAVE7 waiting_for_gpus utc=$(date -u +%FT%TZ)"
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
  --trajectory-observation-attention
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
)

reference_args=(
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
)

standard_args=(
  --pairwise-weight 1
  --base-pairwise-weight 1
  --listwise-weight 0.1
  --top-set-weight 0.5
  --expected-regret-weight 1
  --factor-weight 1
  --private-factor-weight 0.25
  --factor-rank-weight 0.5
  --relative-safety-weight 0.5
  --residual-l2-weight 0.01
  --safety-negative-weight 5
  --factor-loss-scope all
)

pids=()
for index in "${!names[@]}"; do
  gpu="${gpu_ids[${index}]}"
  name="${names[${index}]}"
  variant_args=()
  case "${name}" in
    rawpointcombined_top16_hybrid_standard_actor_seed2)
      variant_args+=("${standard_args[@]}" --top-regret-weight 0 --m0-candidate-fusion --residual-top-k 16 --current-actor-target-root "${current_actor_root}" --current-actor-weight 0.5)
      ;;
    rawpointcombined_top8_hybrid_topregret_actor_seed2)
      variant_args+=("${standard_args[@]}" --top-regret-weight 1 --m0-candidate-fusion --residual-top-k 8 --current-actor-target-root "${current_actor_root}" --current-actor-weight 0.5)
      ;;
    rawpointcombined_top16_reference_q50_strict_actor_seed2)
      variant_args+=("${reference_args[@]}" --m0-candidate-fusion --residual-top-k 16 --current-actor-target-root "${current_actor_root}" --current-actor-weight 0.5)
      ;;
    rawpointcombined_top32_reference_q50_strict_actor_seed2)
      variant_args+=("${reference_args[@]}" --m0-candidate-fusion --residual-top-k 32 --current-actor-target-root "${current_actor_root}" --current-actor-weight 0.5)
      ;;
    rawpointprivate_top16_reference_q50_strict_actor_seed2)
      variant_args+=("${reference_args[@]}" --residual-top-k 16 --current-actor-target-root "${current_actor_root}" --current-actor-weight 0.5)
      ;;
    rawpointcontextcombined_top16_reference_q50_strict_actor_seed2)
      variant_args+=("${reference_args[@]}" --m0-context-fusion --m0-candidate-fusion --residual-top-k 16 --current-actor-target-root "${current_actor_root}" --current-actor-weight 0.5)
      ;;
    rawpointcombined_top16_reference_q50_balanced_actor_seed2)
      variant_args+=("${reference_args[@]}" --m0-candidate-fusion --residual-top-k 16 --reference-maximum-safety-worse-probability 0.25 --reference-minimum-safe-improvement-probability 0.5 --reference-missed-improvement-weight 0.25 --current-actor-target-root "${current_actor_root}" --current-actor-weight 0.5)
      ;;
    *)
      echo "unhandled wave-7 variant: ${name}" >&2
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
  echo "NO_VQA_WAVE7_TRAIN_STARTED gpu=${gpu} name=${name} pid=$!"
done

failure=0
for index in "${!pids[@]}"; do
  if wait "${pids[${index}]}"; then
    echo "NO_VQA_WAVE7_TRAIN_COMPLETE name=${names[${index}]}"
  else
    echo "NO_VQA_WAVE7_TRAIN_FAILED name=${names[${index}]}" >&2
    failure=1
  fi
done
(( failure == 0 )) || exit 1
touch "${run_root}/.wave7_complete"
echo "NO_VQA_WAVE7_COMPLETE run_root=${run_root}"
