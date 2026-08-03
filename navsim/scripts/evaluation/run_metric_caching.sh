#!/usr/bin/env bash

set -euo pipefail

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtest}"
CACHE_PATH="${CACHE_PATH:-$NAVSIM_EXP_ROOT/metric_cache}"
CACHE_WORKER="${CACHE_WORKER:-single_machine_thread_pool}"
CACHE_WORKERS="${CACHE_WORKERS:-2}"
CACHE_USE_PROCESS_POOL="${CACHE_USE_PROCESS_POOL:-true}"
FORCE_FEATURE_COMPUTATION="${FORCE_FEATURE_COMPUTATION:-false}"
MAX_SCENES="${MAX_SCENES:-}"

args=(
    "train_test_split=$TRAIN_TEST_SPLIT"
    "metric_cache_path=$CACHE_PATH"
    "force_feature_computation=$FORCE_FEATURE_COMPUTATION"
    "worker=$CACHE_WORKER"
)

if [[ "$CACHE_WORKER" == "single_machine_thread_pool" ]]; then
    args+=(
        "worker.max_workers=$CACHE_WORKERS"
        "worker.use_process_pool=$CACHE_USE_PROCESS_POOL"
    )
fi

if [[ -n "$MAX_SCENES" ]]; then
    args+=("train_test_split.scene_filter.max_scenes=$MAX_SCENES")
fi

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" "${args[@]}"
