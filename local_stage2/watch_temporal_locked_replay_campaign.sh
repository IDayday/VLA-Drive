#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/local_stage2/common.sh"

discovery_root="${TEMPORAL_DISCOVERY_ROOT:-${DRIVEVLA_STAGE2_RUN_ROOT}/scorer_pdms93/temporal_consequence_cv5_dedicated_v2}"
replay_root="${TEMPORAL_REPLAY_ROOT:-${DRIVEVLA_STAGE2_RUN_ROOT}/scorer_pdms93/temporal_consequence_cv5_locked_replay_v3}"
policy_root="${TEMPORAL_POLICY_ROOT:-${DRIVEVLA_STAGE2_RUN_ROOT}/scorer_pdms93/temporal_consequence_cv5_common_policy_v3}"
full_root="${TEMPORAL_FULL_ROOT:-${DRIVEVLA_STAGE2_RUN_ROOT}/scorer_pdms93/temporal_consequence_full_data_v3}"
navtest_root="${TEMPORAL_NAVTEST_ROOT:-${DRIVEVLA_STAGE2_RUN_ROOT}/scorer_pdms93/temporal_consequence_locked_replay_navtest_v3}"

feature_cache="${TEMPORAL_NAVTEST_FEATURE_CACHE:-${DRIVEVLA_STAGE2_RUN_ROOT}/ke_candidate_audit/public_base_navtest_scorer_features_full_fp32_v2/proposal_predictions.pkl}"
feature_manifest="${TEMPORAL_NAVTEST_FEATURE_MANIFEST:-${DRIVEVLA_STAGE2_RUN_ROOT}/ke_candidate_audit/public_base_navtest_scorer_features_full_fp32_v2/proposal_cache_manifest.json}"
candidate_matrix="${TEMPORAL_NAVTEST_CANDIDATE_MATRIX:-${DRIVEVLA_STAGE2_RUN_ROOT}/ke_candidate_audit/public_base_navtest_all_candidate_factors_fp32_v1/candidate_scores.npz}"
candidate_summary="${TEMPORAL_NAVTEST_CANDIDATE_SUMMARY:-${DRIVEVLA_STAGE2_RUN_ROOT}/ke_candidate_audit/public_base_navtest_all_candidate_factors_fp32_v1/summary.json}"

IFS=',' read -r -a replay_gpus <<< "${TEMPORAL_REPLAY_GPUS:-0,1,2,3,4}"
IFS=',' read -r -a replay_wait_pids <<< "${TEMPORAL_REPLAY_WAIT_PIDS:-0,0,0,0,0}"
IFS=',' read -r -a full_gpus <<< "${TEMPORAL_FULL_GPUS:-0,1,2}"
eval_gpu="${TEMPORAL_EVAL_GPU:-5}"
minimum_available_gib="${TEMPORAL_MIN_AVAILABLE_GIB:-40}"
minimum_available_bytes=$((minimum_available_gib * 1024 * 1024 * 1024))

if [[ "${#replay_gpus[@]}" -ne 5 || "${#replay_wait_pids[@]}" -ne 5 ]]; then
  printf 'TEMPORAL_CONFIG_ERROR replay GPU and wait-PID lists must each have 5 entries\n' >&2
  exit 64
fi
if [[ "${#full_gpus[@]}" -ne 3 ]]; then
  printf 'TEMPORAL_CONFIG_ERROR full-data GPU list must have 3 entries\n' >&2
  exit 64
fi

for required in "${feature_cache}" "${feature_manifest}" "${candidate_matrix}" "${candidate_summary}"; do
  if [[ ! -f "${required}" ]]; then
    printf 'TEMPORAL_INPUT_MISSING %s\n' "${required}" >&2
    exit 66
  fi
done

mkdir -p \
  "${discovery_root}/logs" \
  "${replay_root}/logs" \
  "${policy_root}" \
  "${full_root}/logs"

wait_for_resources() {
  local gpu="$1"
  local wait_pid="${2:-0}"
  if [[ "${wait_pid}" != "0" ]]; then
    while kill -0 "${wait_pid}" 2>/dev/null; do
      printf 'WAIT_PID %s gpu=%s pid=%s\n' "$(date -u +%FT%TZ)" "${gpu}" "${wait_pid}"
      sleep 30
    done
  fi
  while true; do
    local gpu_memory
    local available_memory
    gpu_memory="$(
      nvidia-smi -i "${gpu}" --query-gpu=memory.used \
        --format=csv,noheader,nounits | tr -d ' '
    )"
    available_memory="$(free -b | awk '$1 == "Mem:" {print $7}')"
    if [[ "${gpu_memory}" -lt 512 && "${available_memory}" -ge "${minimum_available_bytes}" ]]; then
      return
    fi
    printf 'WAIT_RESOURCE %s gpu=%s gpu_mib=%s available_bytes=%s\n' \
      "$(date -u +%FT%TZ)" "${gpu}" "${gpu_memory}" "${available_memory}"
    sleep 60
  done
}

while [[ "$(find "${discovery_root}" -mindepth 2 -maxdepth 2 -name training_results.json | wc -l)" -lt 5 ]]; do
  printf 'WAIT_DISCOVERY %s results=%s/5\n' \
    "$(date -u +%FT%TZ)" \
    "$(find "${discovery_root}" -mindepth 2 -maxdepth 2 -name training_results.json | wc -l)"
  sleep 30
done

discovery_summary="${discovery_root}/cv_discovery_summary.json"
"${DRIVEVLA_PYTHON}" "${repo_root}/local_stage2/summarize_temporal_consequence_cv.py" \
  --run-root "${discovery_root}" \
  --output "${discovery_summary}" \
  --report "${discovery_root}/cv_discovery_summary.md"

common_epoch="$(
  "${DRIVEVLA_PYTHON}" -c \
    'import json,sys; x=json.load(open(sys.argv[1])); assert x["fold_audit"]["complete"]; print(int(x["common_epoch"]["epoch"]))' \
    "${discovery_summary}"
)"
discovery_scheduler_epochs="$(
  "${DRIVEVLA_PYTHON}" -c \
    'import glob,json,sys; paths=glob.glob(sys.argv[1]+"/fold_*/training_results.json"); values={int(json.load(open(p))["metadata"]["training_args"]["epochs"]) for p in paths}; assert len(paths)==5 and len(values)==1; print(values.pop())' \
    "${discovery_root}"
)"
replay_epochs=$((common_epoch + 1))
if [[ "${replay_epochs}" -gt "${discovery_scheduler_epochs}" ]]; then
  printf 'TEMPORAL_EPOCH_ERROR replay=%s scheduler=%s\n' \
    "${replay_epochs}" "${discovery_scheduler_epochs}" >&2
  exit 65
fi
printf 'LOCKED_EPOCH %s epoch=%s replay_epochs=%s scheduler_epochs=%s\n' \
  "$(date -u +%FT%TZ)" "${common_epoch}" "${replay_epochs}" \
  "${discovery_scheduler_epochs}"

launch_replay_fold() {
  local fold="$1"
  local gpu="$2"
  local wait_pid="$3"
  local output="${replay_root}/fold_${fold}_seed2"
  local log="${replay_root}/logs/fold_${fold}_seed2.log"
  if [[ -f "${output}/training_results.json" ]]; then
    printf 'SKIP_REPLAY %s fold=%s complete\n' "$(date -u +%FT%TZ)" "${fold}"
    return
  fi
  if [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    printf 'REFUSE_PARTIAL_REPLAY %s\n' "${output}" >&2
    return 73
  fi
  wait_for_resources "${gpu}" "${wait_pid}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONHASHSEED=2 \
    "${DRIVEVLA_PYTHON}" "${repo_root}/local_stage2/train_temporal_consequence_scorer.py" \
      --source-root "${DRIVEVLA_STAGE2_RUN_ROOT}/scorer_pdms93/public_base_features_full_v1" \
      --factor-root "${DRIVEVLA_STAGE2_RUN_ROOT}/scorer_pdms93/public_base_labels_full_v1" \
      --consequence-root "${DRIVEVLA_STAGE2_RUN_ROOT}/scorer_pdms93/public_base_consequence_labels_top16_v1" \
      --base-checkpoint "${DRIVEVLA_PUBLIC_BASE}" \
      --output-dir "${output}" \
      --fold-index "${fold}" \
      --num-folds 5 \
      --fold-seed 20260901 \
      --seed 2 \
      --epochs "${replay_epochs}" \
      --scheduler-epochs "${discovery_scheduler_epochs}" \
      --retained-epoch "${common_epoch}" \
      --batch-size 128 \
      --eval-batch-size 256 \
      --device cuda >"${log}" 2>&1
}

replay_pids=()
for fold in 0 1 2 3 4; do
  launch_replay_fold \
    "${fold}" "${replay_gpus[fold]}" "${replay_wait_pids[fold]}" &
  replay_pids+=("$!")
done
for pid in "${replay_pids[@]}"; do
  wait "${pid}"
done

cv_summary="${replay_root}/cv_summary_final.json"
"${DRIVEVLA_PYTHON}" "${repo_root}/local_stage2/summarize_temporal_consequence_cv.py" \
  --run-root "${replay_root}" \
  --output "${cv_summary}" \
  --report "${replay_root}/cv_summary_final.md"
"${DRIVEVLA_PYTHON}" -c \
  'import json,sys; x=json.load(open(sys.argv[1])); assert x["fold_audit"]["complete"]; assert x["common_epoch_weights_aligned"]; assert x["robust_deployment_available"]' \
  "${cv_summary}"

artifact_args=()
for fold in 0 1 2 3 4; do
  artifact_args+=(
    --artifact "${replay_root}/fold_${fold}_seed2/best_temporal_consequence_scorer.pt"
  )
done
"${DRIVEVLA_PYTHON}" "${repo_root}/local_stage2/materialize_temporal_cv_policy.py" \
  --cv-summary "${cv_summary}" \
  --output-root "${policy_root}" \
  "${artifact_args[@]}"

launch_full_data() {
  local seed="$1"
  local gpu="$2"
  local output="${full_root}/seed_${seed}"
  local log="${full_root}/logs/seed_${seed}.log"
  if [[ -f "${output}/training_results.json" ]]; then
    printf 'SKIP_FULL %s seed=%s complete\n' "$(date -u +%FT%TZ)" "${seed}"
    return
  fi
  if [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    printf 'REFUSE_PARTIAL_FULL %s\n' "${output}" >&2
    return 73
  fi
  wait_for_resources "${gpu}" 0
  CUDA_VISIBLE_DEVICES="${gpu}" \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONHASHSEED="${seed}" \
    bash "${repo_root}/local_stage2/train_temporal_consequence_scorer.sh" \
      --output-dir "${output}" \
      --train-all \
      --cv-summary "${cv_summary}" \
      --seed "${seed}" \
      --epochs "${replay_epochs}" \
      --scheduler-epochs "${discovery_scheduler_epochs}" \
      --batch-size 128 \
      --eval-batch-size 256 \
      --device cuda >"${log}" 2>&1
}

full_pids=()
for seed in 0 1 2; do
  launch_full_data "${seed}" "${full_gpus[seed]}" &
  full_pids+=("$!")
done
for pid in "${full_pids[@]}"; do
  wait "${pid}"
done

if [[ -f "${navtest_root}/campaign_summary.json" ]]; then
  printf 'SKIP_NAVTEST %s complete=%s\n' "$(date -u +%FT%TZ)" "${navtest_root}"
  exit 0
fi
if [[ -d "${navtest_root}" && -n "$(find "${navtest_root}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  printf 'REFUSE_PARTIAL_NAVTEST %s\n' "${navtest_root}" >&2
  exit 73
fi

evaluation_args=()
for fold in 0 1 2 3 4; do
  evaluation_args+=(
    --artifact "common_policy_fold_${fold}=${policy_root}/fold_${fold}_seed2/common_policy_temporal_consequence_scorer.pt"
  )
done
for seed in 0 1 2; do
  evaluation_args+=(
    --artifact "full_data_seed_${seed}=${full_root}/seed_${seed}/best_temporal_consequence_scorer.pt"
  )
done

wait_for_resources "${eval_gpu}" 0
CUDA_VISIBLE_DEVICES="${eval_gpu}" \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "${DRIVEVLA_PYTHON}" "${repo_root}/local_stage2/evaluate_cached_navtest_scorers.py" \
    --feature-cache "${feature_cache}" \
    --feature-manifest "${feature_manifest}" \
    --candidate-matrix "${candidate_matrix}" \
    --candidate-summary "${candidate_summary}" \
    --output-dir "${navtest_root}" \
    --device cuda \
    --batch-size 128 \
    --bootstrap-iterations 10000 \
    "${evaluation_args[@]}"

printf 'TEMPORAL_LOCKED_CAMPAIGN_COMPLETE %s output=%s\n' \
  "$(date -u +%FT%TZ)" "${navtest_root}"
