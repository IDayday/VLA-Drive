#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 PREDICTIONS_PKL OUTPUT_DIR SHARD_COUNT SHARD_INDEX [CPU_WORKERS] [HYDRA_OVERRIDES...]" >&2
  exit 2
fi

predictions_path="$1"
output_dir="$2"
shard_count="$3"
shard_index="$4"
cpu_workers="${5:-64}"
if [[ $# -ge 5 ]]; then
  shift 5
else
  shift 4
fi

if [[ ! -f "${predictions_path}" ]]; then
  echo "Prediction cache not found: ${predictions_path}" >&2
  exit 2
fi

# A Ray task owns one logical CPU.  Prevent NumPy/PyTorch BLAS libraries inside
# every worker from spawning another thread pool and oversubscribing the host.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export DRIVEVLA_SCORE_RAY=0
export NAVSIM_TRAIN_METRIC_CACHE="${DRIVEVLA_NAVTEST_METRIC_CACHE}"
aggregate_when_complete="${DRIVEVLA_SCORE_AGGREGATE_WHEN_COMPLETE:-false}"

mkdir -p "${output_dir}"

"${DRIVEVLA_PYTHON}" "${DRIVEVLA_REPO_ROOT}/local_stage2/score_cached_navtest_proposals.py" \
  train_test_split=navtest \
  "output_dir=${output_dir}" \
  "+proposal_predictions_path=${predictions_path}" \
  "+proposal_score_shard_count=${shard_count}" \
  "+proposal_score_shard_index=${shard_index}" \
  +proposal_score_allow_partial=false \
  "+proposal_score_aggregate_when_complete=${aggregate_when_complete}" \
  "metric_cache_path=${DRIVEVLA_NAVTEST_METRIC_CACHE}" \
  "navsim_log_path=${DRIVEVLA_DATA_ROOT}/navsim_logs/test" \
  worker=ray_distributed_no_torch \
  "worker.threads_per_node=${cpu_workers}" \
  worker.log_to_driver=false \
  logger_level=warning \
  "$@"
