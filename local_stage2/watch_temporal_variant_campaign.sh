#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/local_stage2/common.sh"

variant="${TEMPORAL_VARIANT_NAME:?TEMPORAL_VARIANT_NAME is required}"
score_mode="${TEMPORAL_SCORE_MODE:-residual}"
use_base_candidate_features="${TEMPORAL_USE_BASE_CANDIDATE_FEATURES:-false}"
use_relative_safety_head="${TEMPORAL_USE_RELATIVE_SAFETY_HEAD:-false}"
safety_gate_mode="${TEMPORAL_SAFETY_GATE_MODE:-absolute}"
predecessor_pid="${TEMPORAL_PREDECESSOR_PID:-0}"
minimum_available_gib="${TEMPORAL_MIN_AVAILABLE_GIB:-40}"
minimum_available_bytes=$((minimum_available_gib * 1024 * 1024 * 1024))
IFS=',' read -r -a campaign_gpus <<< "${TEMPORAL_REPLAY_GPUS:-0,1,2,6,5}"

case "${variant}" in
  *[!A-Za-z0-9_.-]*|'')
    printf 'TEMPORAL_CONFIG_ERROR invalid variant name: %s\n' "${variant}" >&2
    exit 64
    ;;
esac
case "${score_mode}" in
  residual|factor_aggregate|hybrid) ;;
  *)
    printf 'TEMPORAL_CONFIG_ERROR invalid score mode: %s\n' "${score_mode}" >&2
    exit 64
    ;;
esac
case "${use_base_candidate_features}" in
  true|false) ;;
  *)
    printf 'TEMPORAL_CONFIG_ERROR invalid base-feature flag: %s\n' \
      "${use_base_candidate_features}" >&2
    exit 64
    ;;
esac
case "${use_relative_safety_head}" in
  true|false) ;;
  *)
    printf 'TEMPORAL_CONFIG_ERROR invalid relative-safety flag: %s\n' \
      "${use_relative_safety_head}" >&2
    exit 64
    ;;
esac
case "${safety_gate_mode}" in
  absolute|relative) ;;
  *)
    printf 'TEMPORAL_CONFIG_ERROR invalid safety gate mode: %s\n' \
      "${safety_gate_mode}" >&2
    exit 64
    ;;
esac
if [[ "${safety_gate_mode}" == "relative" && "${use_relative_safety_head}" != "true" ]]; then
  printf 'TEMPORAL_CONFIG_ERROR relative gate requires relative-safety head\n' >&2
  exit 64
fi
if [[ "${#campaign_gpus[@]}" -ne 5 ]]; then
  printf 'TEMPORAL_CONFIG_ERROR campaign GPU list must have 5 entries\n' >&2
  exit 64
fi

run_root="${DRIVEVLA_STAGE2_RUN_ROOT}/scorer_pdms93"
discovery_root="${TEMPORAL_DISCOVERY_ROOT:-${run_root}/${variant}_discovery}"
replay_root="${TEMPORAL_REPLAY_ROOT:-${run_root}/${variant}_locked_replay}"
policy_root="${TEMPORAL_POLICY_ROOT:-${run_root}/${variant}_common_policy}"
full_root="${TEMPORAL_FULL_ROOT:-${run_root}/${variant}_full_data}"
navtest_root="${TEMPORAL_NAVTEST_ROOT:-${run_root}/${variant}_navtest}"
controller_log="${discovery_root}/logs/controller.log"

mkdir -p "${discovery_root}/logs"

if [[ "${predecessor_pid}" != "0" ]]; then
  while kill -0 "${predecessor_pid}" 2>/dev/null; do
    printf 'WAIT_PREDECESSOR %s pid=%s\n' \
      "$(date -u +%FT%TZ)" "${predecessor_pid}"
    sleep 30
  done
fi

wait_for_resources() {
  local gpu="$1"
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

model_args=()
if [[ "${use_base_candidate_features}" == "true" ]]; then
  model_args+=(--use-base-candidate-features)
fi
if [[ "${use_relative_safety_head}" == "true" ]]; then
  model_args+=(--use-relative-safety-head)
fi
model_args+=(--safety-gate-mode "${safety_gate_mode}")

launch_discovery_fold() {
  local fold="$1"
  local gpu="$2"
  local output="${discovery_root}/fold_${fold}_seed2"
  local log="${discovery_root}/logs/fold_${fold}_seed2.log"
  if [[ -f "${output}/training_results.json" ]]; then
    printf 'SKIP_DISCOVERY %s fold=%s complete\n' \
      "$(date -u +%FT%TZ)" "${fold}"
    return
  fi
  if [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    printf 'REFUSE_PARTIAL_DISCOVERY %s\n' "${output}" >&2
    return 73
  fi
  wait_for_resources "${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONHASHSEED=2 \
    "${DRIVEVLA_PYTHON}" "${repo_root}/local_stage2/train_temporal_consequence_scorer.py" \
      --source-root "${run_root}/public_base_features_full_v1" \
      --factor-root "${run_root}/public_base_labels_full_v1" \
      --consequence-root "${run_root}/public_base_consequence_labels_top16_v1" \
      --base-checkpoint "${DRIVEVLA_PUBLIC_BASE}" \
      --output-dir "${output}" \
      --fold-index "${fold}" \
      --num-folds 5 \
      --fold-seed 20260901 \
      --seed 2 \
      --epochs 12 \
      --score-mode "${score_mode}" \
      "${model_args[@]}" \
      --batch-size 128 \
      --eval-batch-size 256 \
      --device cuda >"${log}" 2>&1
}

discovery_pids=()
for fold in 0 1 2 3 4; do
  launch_discovery_fold "${fold}" "${campaign_gpus[fold]}" &
  discovery_pids+=("$!")
done
for pid in "${discovery_pids[@]}"; do
  wait "${pid}"
done

TEMPORAL_DISCOVERY_ROOT="${discovery_root}" \
TEMPORAL_REPLAY_ROOT="${replay_root}" \
TEMPORAL_POLICY_ROOT="${policy_root}" \
TEMPORAL_FULL_ROOT="${full_root}" \
TEMPORAL_NAVTEST_ROOT="${navtest_root}" \
TEMPORAL_SCORE_MODE="${score_mode}" \
TEMPORAL_USE_BASE_CANDIDATE_FEATURES="${use_base_candidate_features}" \
TEMPORAL_USE_RELATIVE_SAFETY_HEAD="${use_relative_safety_head}" \
TEMPORAL_SAFETY_GATE_MODE="${safety_gate_mode}" \
TEMPORAL_REPLAY_GPUS="$(IFS=,; printf '%s' "${campaign_gpus[*]}")" \
TEMPORAL_REPLAY_WAIT_PIDS=0,0,0,0,0 \
TEMPORAL_FULL_GPUS="${campaign_gpus[0]},${campaign_gpus[1]},${campaign_gpus[2]}" \
TEMPORAL_EVAL_GPU="${campaign_gpus[4]}" \
TEMPORAL_MIN_AVAILABLE_GIB="${minimum_available_gib}" \
  bash "${repo_root}/local_stage2/watch_temporal_locked_replay_campaign.sh" \
    >>"${controller_log}" 2>&1

printf 'TEMPORAL_VARIANT_CAMPAIGN_COMPLETE %s variant=%s\n' \
  "$(date -u +%FT%TZ)" "${variant}"
