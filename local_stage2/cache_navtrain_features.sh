#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

mkdir -p "${DRIVEVLA_NAVTRAIN_FEATURE_CACHE}"

"${DRIVEVLA_PYTHON}" "${DRIVEVLA_REPO_ROOT}/navsim/planning/script/run_dataset_caching.py" \
  train_test_split=navtrain \
  "cache_path=${DRIVEVLA_NAVTRAIN_FEATURE_CACHE}" \
  "navsim_log_path=${DRIVEVLA_DATA_ROOT}/navsim_logs/trainval" \
  "sensor_blobs_path=${DRIVEVLA_SENSOR_ROOT}/trainval" \
  "force_cache_computation=${DRIVEVLA_FORCE_FEATURE_RECACHE:-false}" \
  agent.cache_data=true \
  agent.checkpoint_path=null \
  agent.stage1_checkpoint_path=null \
  worker=ray_distributed_no_torch \
  worker.threads_per_node=128 \
  worker.log_to_driver=false

"${DRIVEVLA_PYTHON}" "$(dirname "${BASH_SOURCE[0]}")/verify_cache.py"
"${DRIVEVLA_PYTHON}" "$(dirname "${BASH_SOURCE[0]}")/verify_feature_cache_semantics.py" \
  "${DRIVEVLA_NAVTRAIN_FEATURE_CACHE}" "${DRIVEVLA_SENSOR_ROOT}/trainval"
"${DRIVEVLA_PYTHON}" "$(dirname "${BASH_SOURCE[0]}")/verify_feature_cache_integrity.py" \
  "${DRIVEVLA_NAVTRAIN_FEATURE_CACHE}"
