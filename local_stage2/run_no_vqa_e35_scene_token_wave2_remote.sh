#!/usr/bin/env bash

# Second, predeclared No-VQA scorer-representation wave for an idle 8-GPU host.
# It isolates the frozen M0 candidate hidden state and Base-top-K conservative
# ranking.  The source/label cache is copied to host-local storage by the
# companion launcher before this script starts.

set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
python_bin="${DRIVEVLA_PYTHON:-/mnt/project/DriveVLA-M0-env/bin/python}"
source_root="${NO_VQA_SOURCE_ROOT:-/root/scorer_pdms93_cache/no_vqa_e35_features_full_v1}"
label_root="${NO_VQA_LABEL_ROOT:-/root/scorer_pdms93_cache/no_vqa_e35_labels_full_v1}"
run_root="${NO_VQA_WAVE2_RUN_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_scene_token_wave2_v1}"
calibrated_root="${NO_VQA_WAVE2_CALIBRATED_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_scene_token_wave2_calibrated_v1}"
log_root="${NO_VQA_WAVE2_LOG_ROOT:-/root/scorer_pdms93_logs/no_vqa_e35_scene_token_wave2_v1}"
split_manifest="${NO_VQA_SPLIT_MANIFEST:-${repo_root}/reports/scorer_pdms93/OFFICIAL_SCORER_SPLIT.json}"
current_actor_root="${NO_VQA_CURRENT_ACTOR_ROOT:-/mnt/project/DriveVLA-M0-gate-c/outputs/shared_future_candidate_consequence_gate_c/all/oracle_store}"

for path in "${source_root}" "${label_root}" "${split_manifest}"; do
  [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 2; }
done
if [[ -e "${run_root}" || -e "${calibrated_root}" || -e "${log_root}" ]]; then
  echo "wave-2 output already exists; refusing overwrite" >&2
  exit 2
fi
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
echo "NO_VQA_WAVE2_CACHE_VERIFICATION=PASS"

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
  --safety-negative-weight 1
)

names=(
  candidate_only_top64_hybrid_seed2
  candidate_only_top16_hybrid_seed2
  candidate_only_top8_hybrid_seed2
  combined_top16_hybrid_actor050_seed2
  combined_top8_hybrid_actor050_seed2
  combined_top4_hybrid_actor050_seed2
  combined_top16_factor_actor050_seed2
  combined_top16_hybrid_actor050_seed11
)

train_pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  name="${names[${gpu}]}"
  variant_args=()
  case "${name}" in
    candidate_only_top64_hybrid_seed2)
      variant_args+=(--score-mode hybrid --seed 2 --residual-top-k 64 --m0-candidate-only --private-factor-weight 0)
      ;;
    candidate_only_top16_hybrid_seed2)
      variant_args+=(--score-mode hybrid --seed 2 --residual-top-k 16 --m0-candidate-only --private-factor-weight 0)
      ;;
    candidate_only_top8_hybrid_seed2)
      variant_args+=(--score-mode hybrid --seed 2 --residual-top-k 8 --m0-candidate-only --private-factor-weight 0)
      ;;
    combined_top16_hybrid_actor050_seed11)
      variant_args+=(--score-mode hybrid --seed 11 --residual-top-k 16 --current-actor-target-root "${current_actor_root}" --current-actor-weight 0.5)
      ;;
    combined_top16_factor_actor050_seed2)
      variant_args+=(--score-mode factor --seed 2 --residual-top-k 16 --current-actor-target-root "${current_actor_root}" --current-actor-weight 0.5)
      ;;
    combined_top8_hybrid_actor050_seed2)
      variant_args+=(--score-mode hybrid --seed 2 --residual-top-k 8 --current-actor-target-root "${current_actor_root}" --current-actor-weight 0.5)
      ;;
    combined_top4_hybrid_actor050_seed2)
      variant_args+=(--score-mode hybrid --seed 2 --residual-top-k 4 --current-actor-target-root "${current_actor_root}" --current-actor-weight 0.5)
      ;;
    *)
      variant_args+=(--score-mode hybrid --seed 2 --residual-top-k 16 --current-actor-target-root "${current_actor_root}" --current-actor-weight 0.5)
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
  echo "NO_VQA_WAVE2_TRAIN_STARTED gpu=${gpu} name=${name} pid=$!"
done

failure=0
for index in "${!train_pids[@]}"; do
  if wait "${train_pids[${index}]}"; then
    echo "NO_VQA_WAVE2_TRAIN_COMPLETE name=${names[${index}]}"
  else
    status=$?
    echo "NO_VQA_WAVE2_TRAIN_FAILED name=${names[${index}]} status=${status}" >&2
    failure=1
  fi
done
(( failure == 0 )) || exit 1

calibration_pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  name="${names[${gpu}]}"
  calibrated_name="${name}__calibrated"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    exec "${python_bin}" "${repo_root}/local_stage2/calibrate_m0_private_residual_policy.py" \
      --source no_vqa_e35 "${source_root}" "${label_root}" \
      --split-manifest "${split_manifest}" \
      --selection-source no_vqa_e35 \
      --artifact "${run_root}/${name}/best_m0_private_residual_scorer.pt" \
      --output-dir "${calibrated_root}/${calibrated_name}" \
      --seed "$((20260922 + gpu))" \
      --eval-batch-size 64 \
      --bootstrap-replicates 1000 \
      --device cuda
  ) >"${log_root}/calibrate_${name}.log" 2>&1 &
  calibration_pids+=("$!")
  echo "NO_VQA_WAVE2_CALIBRATION_STARTED gpu=${gpu} name=${name} pid=$!"
done

for index in "${!calibration_pids[@]}"; do
  if wait "${calibration_pids[${index}]}"; then
    echo "NO_VQA_WAVE2_CALIBRATION_COMPLETE name=${names[${index}]}"
  else
    status=$?
    echo "NO_VQA_WAVE2_CALIBRATION_FAILED name=${names[${index}]} status=${status}" >&2
    failure=1
  fi
done
(( failure == 0 )) || exit 1
touch "${run_root}/.wave2_complete"
echo "NO_VQA_WAVE2_COMPLETE run_root=${run_root} calibrated_root=${calibrated_root}"
