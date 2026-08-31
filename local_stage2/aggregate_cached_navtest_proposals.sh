#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 PREDICTIONS_PKL OUTPUT_DIR [CHECKPOINT] [HYDRA_OVERRIDES...]" >&2
  exit 2
fi

predictions_path="$1"
output_dir="$2"
checkpoint_path="${3:-}"
if [[ $# -ge 3 ]]; then
  shift 3
else
  shift 2
fi

if [[ ! -f "${predictions_path}" ]]; then
  echo "Prediction cache not found: ${predictions_path}" >&2
  exit 2
fi

extra_overrides=()
if [[ -n "${checkpoint_path}" ]]; then
  escaped_checkpoint="${checkpoint_path//=/\\=}"
  extra_overrides+=("+proposal_score_checkpoint_path=${escaped_checkpoint}")
fi

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUPLAN_MAPS_ROOT="${DRIVEVLA_MAP_ROOT}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"

"${DRIVEVLA_PYTHON}" "${DRIVEVLA_REPO_ROOT}/local_stage2/score_cached_navtest_proposals.py" \
  train_test_split=navtest \
  "output_dir=${output_dir}" \
  "+proposal_predictions_path=${predictions_path}" \
  +proposal_score_shard_count=1 \
  +proposal_score_shard_index=0 \
  +proposal_score_allow_partial=false \
  +proposal_score_aggregate_only=true \
  +proposal_score_aggregate_when_complete=true \
  "metric_cache_path=${DRIVEVLA_NAVTEST_METRIC_CACHE}" \
  "navsim_log_path=${DRIVEVLA_DATA_ROOT}/navsim_logs/test" \
  worker=ray_distributed_no_torch \
  worker.threads_per_node=1 \
  worker.log_to_driver=false \
  logger_level=warning \
  "${extra_overrides[@]}" \
  "$@"
