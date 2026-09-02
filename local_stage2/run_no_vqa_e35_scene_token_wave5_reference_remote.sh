#!/usr/bin/env bash

# Fifth No-VQA scorer wave: M0-owned scorer-private/current-observation
# representations with an uncertainty-aware policy-improvement head.  The
# released No-VQA scorer choice is an exact fallback; a candidate may replace
# it only when predicted relative gain and safety gates both pass.  Variants
# are predeclared here and all held-out-log-positive artifacts are promoted.

set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
python_bin="${DRIVEVLA_PYTHON:-/mnt/project/DriveVLA-M0-env/bin/python}"
source_root="${NO_VQA_SOURCE_ROOT:-/root/scorer_pdms93_cache/no_vqa_e35_features_full_v1}"
label_root="${NO_VQA_LABEL_ROOT:-/root/scorer_pdms93_cache/no_vqa_e35_labels_full_v1}"
run_root="${NO_VQA_WAVE5_RUN_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_scene_token_wave5_reference_v1}"
log_root="${NO_VQA_WAVE5_LOG_ROOT:-/root/scorer_pdms93_logs/no_vqa_e35_scene_token_wave5_reference_v1}"
split_manifest="${NO_VQA_SPLIT_MANIFEST:-${repo_root}/reports/scorer_pdms93/OFFICIAL_SCORER_SPLIT.json}"
current_actor_root="${NO_VQA_CURRENT_ACTOR_ROOT:-/mnt/project/DriveVLA-M0-gate-c/outputs/shared_future_candidate_consequence_gate_c/all/oracle_store}"

gpu_ids=(0 1 2 3 4 5 6 7)
names=(
  combined_top8_reference_q10_strict_actor_seed2
  combined_top8_reference_q50_strict_actor_seed2
  combined_top16_reference_q10_strict_actor_seed2
  combined_top16_reference_q50_strict_actor_seed2
  combined_top32_reference_q10_strict_actor_seed2
  private_top16_reference_q10_strict_actor_seed2
  candidateonly_top16_reference_q10_strict_seed2
  combined_top16_reference_q10_balanced_actor_seed2
)

for path in "${source_root}" "${label_root}" "${split_manifest}" "${current_actor_root}"; do
  [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 2; }
done
if [[ -e "${run_root}" || -e "${log_root}" ]]; then
  echo "wave-5 output already exists; refusing overwrite" >&2
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
  echo "NO_VQA_WAVE5 waiting_for_gpus utc=$(date -u +%FT%TZ)"
  sleep 15
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

"${python_bin}" "${repo_root}/local_stage2/verify_no_vqa_scorer_cache.py" \
  --source-root "${source_root}" \
  --label-root "${label_root}" \
  --expected-checkpoint-sha256 72c74a113c557df27c86a320f66d4ff2a79fc1a19e678337d5a142a520359309 \
  --expected-config-sha256 5f70b74293883bebb80fc1feffaf3786556f909645a248374495dfadbf7cd1c3 \
  --expected-scenes 103288 \
  --output-json "${log_root}/CACHE_VERIFICATION.json" \
  --output-md "${log_root}/CACHE_VERIFICATION.md" \
  >"${log_root}/cache_verification.log" 2>&1

common_args=(
  --source no_vqa_e35 "${source_root}" "${label_root}"
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
  --pairwise-weight 0
  --base-pairwise-weight 0
  --listwise-weight 0
  --top-set-weight 0
  --expected-regret-weight 0
  --top-regret-weight 0
  --factor-weight 0
  --factor-rank-weight 0
  --relative-safety-weight 0
  --residual-l2-weight 0
  --safety-negative-weight 1
  --factor-loss-scope topk
  --score-mode hybrid
  --seed 2
  --conservative-reference
  --reference-hidden-dim 512
  --reference-layers 2
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
)

train_pids=()
for index in "${!gpu_ids[@]}"; do
  gpu="${gpu_ids[${index}]}"
  name="${names[${index}]}"
  variant_args=()
  case "${name}" in
    combined_top8_reference_q10_strict_actor_seed2)
      variant_args+=(--m0-candidate-fusion --residual-top-k 8 --reference-gain-quantile-index 0 --reference-maximum-safety-worse-probability 0.1 --reference-minimum-safe-improvement-probability 0.7 --private-factor-weight 0.25 --current-actor-target-root "${current_actor_root}" --current-actor-weight 0.5)
      ;;
    combined_top8_reference_q50_strict_actor_seed2)
      variant_args+=(--m0-candidate-fusion --residual-top-k 8 --reference-gain-quantile-index 1 --reference-maximum-safety-worse-probability 0.1 --reference-minimum-safe-improvement-probability 0.7 --private-factor-weight 0.25 --current-actor-target-root "${current_actor_root}" --current-actor-weight 0.5)
      ;;
    combined_top16_reference_q10_strict_actor_seed2)
      variant_args+=(--m0-candidate-fusion --residual-top-k 16 --reference-gain-quantile-index 0 --reference-maximum-safety-worse-probability 0.1 --reference-minimum-safe-improvement-probability 0.7 --private-factor-weight 0.25 --current-actor-target-root "${current_actor_root}" --current-actor-weight 0.5)
      ;;
    combined_top16_reference_q50_strict_actor_seed2)
      variant_args+=(--m0-candidate-fusion --residual-top-k 16 --reference-gain-quantile-index 1 --reference-maximum-safety-worse-probability 0.1 --reference-minimum-safe-improvement-probability 0.7 --private-factor-weight 0.25 --current-actor-target-root "${current_actor_root}" --current-actor-weight 0.5)
      ;;
    combined_top32_reference_q10_strict_actor_seed2)
      variant_args+=(--m0-candidate-fusion --residual-top-k 32 --reference-gain-quantile-index 0 --reference-maximum-safety-worse-probability 0.1 --reference-minimum-safe-improvement-probability 0.7 --private-factor-weight 0.25 --current-actor-target-root "${current_actor_root}" --current-actor-weight 0.5)
      ;;
    private_top16_reference_q10_strict_actor_seed2)
      variant_args+=(--residual-top-k 16 --reference-gain-quantile-index 0 --reference-maximum-safety-worse-probability 0.1 --reference-minimum-safe-improvement-probability 0.7 --private-factor-weight 0.25 --current-actor-target-root "${current_actor_root}" --current-actor-weight 0.5)
      ;;
    candidateonly_top16_reference_q10_strict_seed2)
      variant_args+=(--m0-candidate-fusion --m0-candidate-only --residual-top-k 16 --reference-gain-quantile-index 0 --reference-maximum-safety-worse-probability 0.1 --reference-minimum-safe-improvement-probability 0.7 --private-factor-weight 0 --current-actor-weight 0)
      ;;
    combined_top16_reference_q10_balanced_actor_seed2)
      variant_args+=(--m0-candidate-fusion --residual-top-k 16 --reference-gain-quantile-index 0 --reference-maximum-safety-worse-probability 0.25 --reference-minimum-safe-improvement-probability 0.5 --reference-missed-improvement-weight 0.25 --private-factor-weight 0.25 --current-actor-target-root "${current_actor_root}" --current-actor-weight 0.5)
      ;;
    *)
      echo "unhandled wave-5 variant: ${name}" >&2
      exit 2
      ;;
  esac
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    exec "${python_bin}" "${repo_root}/local_stage2/train_m0_private_residual_scorer.py" \
      "${common_args[@]}" \
      "${variant_args[@]}" \
      --output-dir "${run_root}/${name}"
  ) >"${log_root}/${name}.log" 2>&1 &
  train_pids+=("$!")
  echo "NO_VQA_WAVE5_TRAIN_STARTED gpu=${gpu} name=${name} pid=$!"
done

failure=0
for index in "${!train_pids[@]}"; do
  if wait "${train_pids[${index}]}"; then
    echo "NO_VQA_WAVE5_TRAIN_COMPLETE name=${names[${index}]}"
  else
    status=$?
    echo "NO_VQA_WAVE5_TRAIN_FAILED name=${names[${index}]} status=${status}" >&2
    failure=1
  fi
done
(( failure == 0 )) || exit 1
touch "${run_root}/.wave5_complete"
echo "NO_VQA_WAVE5_COMPLETE run_root=${run_root}"
