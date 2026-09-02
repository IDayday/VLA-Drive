#!/usr/bin/env bash

# Wait for the exact No-VQA epoch-35 replay cache, verify it, then use all idle
# local GPUs for a preregistered scorer-private scene-token campaign.  In
# parallel, score the matching full Navtest proposal cache on CPU so promoted
# rankers can be evaluated without rerunning the 2B VLM or mixing proposal
# banks.

set -uo pipefail

repo_root="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
python_bin="${DRIVEVLA_PYTHON:-/mnt/project/DriveVLA-M0-env/bin/python}"
source_root="${NO_VQA_SOURCE_ROOT:-/root/scorer_pdms93_cache/no_vqa_e35_features_full_v1}"
label_root="${NO_VQA_LABEL_ROOT:-/root/scorer_pdms93_cache/no_vqa_e35_labels_full_v1}"
run_root="${NO_VQA_SCORER_RUN_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_scene_token_v1}"
log_root="${NO_VQA_SCORER_LOG_ROOT:-/root/scorer_pdms93_logs/no_vqa_e35_scene_token_v1}"
report_root="${repo_root}/reports/m0_independent_scorer_representation/no_vqa_e35_scene_token_v1"
split_manifest="${NO_VQA_SPLIT_MANIFEST:-${repo_root}/reports/scorer_pdms93/OFFICIAL_SCORER_SPLIT.json}"
current_actor_root="${NO_VQA_CURRENT_ACTOR_ROOT:-/mnt/project/DriveVLA-M0-gate-c/outputs/shared_future_candidate_consequence_gate_c/all/oracle_store}"
shared_future_root="${NO_VQA_SHARED_FUTURE_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/shared_future_target_table_v1}"
checkpoint="${NO_VQA_CHECKPOINT:-/mnt/project/DriveVLA-M0-no-vqa/runs/training/no_vqa_full_ft_seed0_e36/lightning_logs/version_0/checkpoints/best-epoch=35-step=232416.ckpt}"
checkpoint_sha="72c74a113c557df27c86a320f66d4ff2a79fc1a19e678337d5a142a520359309"
config_sha="5f70b74293883bebb80fc1feffaf3786556f909645a248374495dfadbf7cd1c3"
navtest_features="${NO_VQA_NAVTEST_FEATURES:-/mnt/project/DriveVLA-M0-stage2/runs/ke_candidate_audit/no_vqa_e35_navtest_scorer_features_fp32_v1/proposal_predictions.pkl}"
navtest_scores="${NO_VQA_NAVTEST_SCORES:-/root/scorer_pdms93_cache/no_vqa_e35_navtest_candidate_scores_fp32_v1}"
wait_seconds="${NO_VQA_WATCH_POLL_SECONDS:-30}"

mkdir -p "${run_root}" "${log_root}" "${report_root}"

cache_ready() {
  [[ "$(find "${source_root}" -mindepth 2 -maxdepth 2 -name manifest.json -type f 2>/dev/null | wc -l)" -eq 8 ]] || return 1
  [[ "$(find "${label_root}" -mindepth 1 -maxdepth 1 -name 'worker_manifest_*.json' -type f 2>/dev/null | wc -l)" -eq 4 ]] || return 1
  "${python_bin}" - "${label_root}" <<'PY' >/dev/null 2>&1
import json
import sys
from pathlib import Path

paths = sorted(Path(sys.argv[1]).glob("worker_manifest_*.json"))
if len(paths) != 4:
    raise SystemExit(1)
for path in paths:
    payload = json.loads(path.read_text())
    if not payload.get("source_complete") or not payload.get("worker_complete"):
        raise SystemExit(1)
    if int(payload.get("failed_chunk_count", -1)):
        raise SystemExit(1)
PY
}

while ! cache_ready; do
  echo "NO_VQA_SCORER_WATCH waiting_for_verified_cache utc=$(date -u +%FT%TZ)"
  sleep "${wait_seconds}"
done

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
  --expected-checkpoint-sha256 "${checkpoint_sha}" \
  --expected-config-sha256 "${config_sha}" \
  --expected-scenes 103288 \
  --output-json "${report_root}/CACHE_VERIFICATION.json" \
  --output-md "${report_root}/CACHE_VERIFICATION.md" \
  >"${log_root}/cache_verification.log" 2>&1
echo "NO_VQA_SCORER_WATCH cache_verification=PASS"

# The matching Navtest feature cache is a separate FP32 forward of the same
# No-VQA checkpoint.  Score it once; do not reuse the older candidate matrix
# whose trajectories differ numerically.
(
  export DRIVEVLA_REPO_ROOT="${repo_root}"
  export DRIVEVLA_PYTHON="${python_bin}"
  export DRIVEVLA_SCORE_AGGREGATE_WHEN_COMPLETE=false
  bash "${repo_root}/local_stage2/score_cached_navtest_proposals.sh" \
    "${navtest_features}" "${navtest_scores}" 1 0 48
  bash "${repo_root}/local_stage2/aggregate_cached_navtest_proposals.sh" \
    "${navtest_features}" "${navtest_scores}" "${checkpoint}"
) >"${log_root}/navtest_candidate_scoring.log" 2>&1 &
navtest_pid=$!
echo "NO_VQA_NAVTEST_CANDIDATE_SCORING_STARTED pid=${navtest_pid}"

# Do not collide with a task that may have claimed a GPU after cache export.
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')" ]]; do
  echo "NO_VQA_SCORER_WATCH waiting_for_idle_gpus utc=$(date -u +%FT%TZ)"
  sleep "${wait_seconds}"
done

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
  --residual-top-k 64
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
  primary_hybrid_actor050_seed2
  control_hybrid_no_actor_seed2
  factor_actor050_seed2
  direct_actor050_seed2
  primary_hybrid_actor050_seed11
  primary_hybrid_actor050_seed23
  hybrid_actor050_deep_seed2
  hybrid_actor050_future025_seed2
)

train_pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  name="${names[${gpu}]}"
  output_dir="${run_root}/${name}"
  if [[ -e "${output_dir}" ]]; then
    echo "NO_VQA_SCORER_WATCH refusing_existing_output=${output_dir}" >&2
    exit 2
  fi
  variant_args=()
  case "${name}" in
    control_hybrid_no_actor_seed2)
      variant_args+=(--score-mode hybrid --seed 2)
      ;;
    factor_actor050_seed2)
      variant_args+=(--score-mode factor --seed 2 --current-actor-target-root "${current_actor_root}" --current-actor-weight 0.5)
      ;;
    direct_actor050_seed2)
      variant_args+=(--score-mode direct --seed 2 --current-actor-target-root "${current_actor_root}" --current-actor-weight 0.5)
      ;;
    primary_hybrid_actor050_seed11)
      variant_args+=(--score-mode hybrid --seed 11 --current-actor-target-root "${current_actor_root}" --current-actor-weight 0.5)
      ;;
    primary_hybrid_actor050_seed23)
      variant_args+=(--score-mode hybrid --seed 23 --current-actor-target-root "${current_actor_root}" --current-actor-weight 0.5)
      ;;
    hybrid_actor050_deep_seed2)
      variant_args+=(--score-mode hybrid --seed 2 --private-layers 3 --residual-layers 3 --current-actor-target-root "${current_actor_root}" --current-actor-weight 0.5)
      ;;
    hybrid_actor050_future025_seed2)
      variant_args+=(--score-mode hybrid --seed 2 --current-actor-target-root "${current_actor_root}" --current-actor-weight 0.5 --shared-future-target-root "${shared_future_root}" --shared-future-weight 0.25)
      ;;
    *)
      variant_args+=(--score-mode hybrid --seed 2 --current-actor-target-root "${current_actor_root}" --current-actor-weight 0.5)
      ;;
  esac
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    exec "${python_bin}" "${repo_root}/local_stage2/train_m0_private_residual_scorer.py" \
      "${common_args[@]}" \
      "${variant_args[@]}" \
      --output-dir "${output_dir}"
  ) >"${log_root}/${name}.log" 2>&1 &
  train_pids+=("$!")
  echo "NO_VQA_SCORER_TRAIN_STARTED gpu=${gpu} name=${name} pid=$!"
done

failure=0
for index in "${!train_pids[@]}"; do
  if wait "${train_pids[${index}]}"; then
    echo "NO_VQA_SCORER_TRAIN_COMPLETE name=${names[${index}]}"
  else
    status=$?
    echo "NO_VQA_SCORER_TRAIN_FAILED name=${names[${index}]} status=${status}" >&2
    failure=1
  fi
done

if wait "${navtest_pid}"; then
  echo "NO_VQA_NAVTEST_CANDIDATE_SCORING_COMPLETE output=${navtest_scores}"
else
  status=$?
  echo "NO_VQA_NAVTEST_CANDIDATE_SCORING_FAILED status=${status}" >&2
  failure=1
fi

if (( failure != 0 )); then
  echo "NO_VQA_SCENE_TOKEN_CAMPAIGN_FAILED" >&2
  exit 1
fi
echo "NO_VQA_SCENE_TOKEN_CAMPAIGN_COMPLETE runs=${run_root} navtest=${navtest_scores}"
