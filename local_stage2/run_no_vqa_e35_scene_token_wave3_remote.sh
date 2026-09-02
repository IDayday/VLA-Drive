#!/usr/bin/env bash

# Third No-VQA scorer wave for the explicitly idle rl-zt4 GPUs 1/3/5/6/7.
# This wave isolates candidate-relative shared-future relabeling and a
# train-data-preregistered safety-class weight.  It never uses DrivOR or any
# future/evaluator tensor as an inference input.

set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
python_bin="${DRIVEVLA_PYTHON:-/mnt/project/DriveVLA-M0-env/bin/python}"
source_root="${NO_VQA_SOURCE_ROOT:-/root/scorer_pdms93_cache/no_vqa_e35_features_full_v1}"
label_root="${NO_VQA_LABEL_ROOT:-/root/scorer_pdms93_cache/no_vqa_e35_labels_full_v1}"
run_root="${NO_VQA_WAVE3_RUN_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_scene_token_wave3_v1}"
calibrated_root="${NO_VQA_WAVE3_CALIBRATED_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_scene_token_wave3_calibrated_v1}"
log_root="${NO_VQA_WAVE3_LOG_ROOT:-/root/scorer_pdms93_logs/no_vqa_e35_scene_token_wave3_v1}"
split_manifest="${NO_VQA_SPLIT_MANIFEST:-${repo_root}/reports/scorer_pdms93/OFFICIAL_SCORER_SPLIT.json}"
current_actor_root="${NO_VQA_CURRENT_ACTOR_ROOT:-/mnt/project/DriveVLA-M0-gate-c/outputs/shared_future_candidate_consequence_gate_c/all/oracle_store}"
shared_future_root="${NO_VQA_SHARED_FUTURE_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/shared_future_target_table_v1}"

gpu_ids=(1 3 5 6 7)
names=(
  factorized_top16_cv_hybrid_safety5_seed2
  factorized_top16_cv_hybrid_safety1_seed2
  factorized_top8_cv_hybrid_safety5_seed2
  combined_top16_hybrid_safety5_seed2
  candidate_only_top16_factor_safety5_seed2
)

for path in \
  "${source_root}" \
  "${label_root}" \
  "${split_manifest}" \
  "${current_actor_root}" \
  "${shared_future_root}"; do
  [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 2; }
done
if [[ -e "${run_root}" || -e "${calibrated_root}" || -e "${log_root}" ]]; then
  echo "wave-3 output already exists; refusing overwrite" >&2
  exit 2
fi

# Do not occupy rl-zt4 GPUs that became busy after this wave was scheduled.
mapfile -t gpu_memory < <(
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
)
for gpu in "${gpu_ids[@]}"; do
  used="${gpu_memory[${gpu}]//[[:space:]]/}"
  if [[ ! "${used}" =~ ^[0-9]+$ ]] || (( used > 1024 )); then
    echo "wave-3 GPU ${gpu} is no longer idle (memory.used=${used} MiB)" >&2
    exit 2
  fi
done

mkdir -p "${run_root}" "${calibrated_root}" "${log_root}"
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
echo "NO_VQA_WAVE3_CACHE_VERIFICATION=PASS"

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
  --m0-candidate-fusion
  --max-residual 0.5
  --minimum-pair-delta 0.02
  --factor-rank-minimum-delta 0.05
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
)

train_pids=()
for index in "${!gpu_ids[@]}"; do
  gpu="${gpu_ids[${index}]}"
  name="${names[${index}]}"
  variant_args=()
  case "${name}" in
    factorized_top16_cv_hybrid_safety5_seed2)
      variant_args+=(--score-mode hybrid --seed 2 --residual-top-k 16 --safety-negative-weight 5 --current-actor-target-root "${current_actor_root}" --current-actor-weight 0.5 --shared-future-target-root "${shared_future_root}" --shared-future-weight 0.5 --shared-future-relabeling --shared-future-constant-velocity-residual --candidate-relative-weight 1)
      ;;
    factorized_top16_cv_hybrid_safety1_seed2)
      variant_args+=(--score-mode hybrid --seed 2 --residual-top-k 16 --safety-negative-weight 1 --current-actor-target-root "${current_actor_root}" --current-actor-weight 0.5 --shared-future-target-root "${shared_future_root}" --shared-future-weight 0.5 --shared-future-relabeling --shared-future-constant-velocity-residual --candidate-relative-weight 1)
      ;;
    factorized_top8_cv_hybrid_safety5_seed2)
      variant_args+=(--score-mode hybrid --seed 2 --residual-top-k 8 --safety-negative-weight 5 --current-actor-target-root "${current_actor_root}" --current-actor-weight 0.5 --shared-future-target-root "${shared_future_root}" --shared-future-weight 0.5 --shared-future-relabeling --shared-future-constant-velocity-residual --candidate-relative-weight 1)
      ;;
    combined_top16_hybrid_safety5_seed2)
      variant_args+=(--score-mode hybrid --seed 2 --residual-top-k 16 --safety-negative-weight 5 --current-actor-target-root "${current_actor_root}" --current-actor-weight 0.5)
      ;;
    candidate_only_top16_factor_safety5_seed2)
      variant_args+=(--score-mode factor --seed 2 --residual-top-k 16 --safety-negative-weight 5 --m0-candidate-only --private-factor-weight 0)
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
  echo "NO_VQA_WAVE3_TRAIN_STARTED gpu=${gpu} name=${name} pid=$!"
done

failure=0
for index in "${!train_pids[@]}"; do
  if wait "${train_pids[${index}]}"; then
    echo "NO_VQA_WAVE3_TRAIN_COMPLETE name=${names[${index}]}"
  else
    status=$?
    echo "NO_VQA_WAVE3_TRAIN_FAILED name=${names[${index}]} status=${status}" >&2
    failure=1
  fi
done
(( failure == 0 )) || exit 1

calibration_pids=()
for index in "${!gpu_ids[@]}"; do
  gpu="${gpu_ids[${index}]}"
  name="${names[${index}]}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    exec "${python_bin}" "${repo_root}/local_stage2/calibrate_m0_private_residual_policy.py" \
      --source no_vqa_e35 "${source_root}" "${label_root}" \
      --split-manifest "${split_manifest}" \
      --selection-source no_vqa_e35 \
      --artifact "${run_root}/${name}/best_m0_private_residual_scorer.pt" \
      --output-dir "${calibrated_root}/${name}__calibrated" \
      --seed "$((20260932 + index))" \
      --eval-batch-size 64 \
      --bootstrap-replicates 1000 \
      --device cuda
  ) >"${log_root}/calibrate_${name}.log" 2>&1 &
  calibration_pids+=("$!")
  echo "NO_VQA_WAVE3_CALIBRATION_STARTED gpu=${gpu} name=${name} pid=$!"
done

for index in "${!calibration_pids[@]}"; do
  if wait "${calibration_pids[${index}]}"; then
    echo "NO_VQA_WAVE3_CALIBRATION_COMPLETE name=${names[${index}]}"
  else
    status=$?
    echo "NO_VQA_WAVE3_CALIBRATION_FAILED name=${names[${index}]} status=${status}" >&2
    failure=1
  fi
done
(( failure == 0 )) || exit 1
touch "${run_root}/.wave3_complete"
echo "NO_VQA_WAVE3_COMPLETE run_root=${run_root} calibrated_root=${calibrated_root}"
