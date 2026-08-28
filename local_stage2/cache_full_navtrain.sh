#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

mkdir -p "${DRIVEVLA_NAVTRAIN_METRIC_CACHE}" "${DRIVEVLA_NAVTRAIN_FEATURE_CACHE}"

"${DRIVEVLA_PYTHON}" "${DRIVEVLA_REPO_ROOT}/navsim/planning/script/run_train_metric_caching.py" \
  train_test_split=navtrain \
  "cache.cache_path=${DRIVEVLA_NAVTRAIN_METRIC_CACHE}" \
  "navsim_log_path=${DRIVEVLA_DATA_ROOT}/navsim_logs/trainval" \
  worker=ray_distributed_no_torch \
  worker.threads_per_node=128 \
  worker.log_to_driver=false

"$(dirname "${BASH_SOURCE[0]}")/cache_navtrain_features.sh"
