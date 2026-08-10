#!/usr/bin/env bash
# Build the NAVSIM navtrain metric cache used only by offline consequence labels.

set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
source "$project_root/env.sh"

export NAVSIM_METRIC_CACHE_ROOT="${NAVSIM_METRIC_CACHE_ROOT:-$DRIVEDREAMER_SHARED_ROOT/groundedworld_cache/navsim_metric_navtrain}"
export CACHE_PATH="$NAVSIM_METRIC_CACHE_ROOT"
export TRAIN_TEST_SPLIT=navtrain
export CACHE_WORKER="${CACHE_WORKER:-single_machine_thread_pool}"
export CACHE_WORKERS="${CACHE_WORKERS:-16}"
export CACHE_USE_PROCESS_POOL="${CACHE_USE_PROCESS_POOL:-true}"
export FORCE_FEATURE_COMPUTATION="${FORCE_FEATURE_COMPUTATION:-false}"

mkdir -p "$NAVSIM_METRIC_CACHE_ROOT"
bash "$NAVSIM_DEVKIT_ROOT/scripts/evaluation/run_metric_caching.sh"

metadata_count="$(find "$NAVSIM_METRIC_CACHE_ROOT/metadata" -maxdepth 1 -type f -name '*.csv' | wc -l)"
entry_count="$(find "$NAVSIM_METRIC_CACHE_ROOT" -type f -name 'metric_cache.pkl' | wc -l)"
if (( metadata_count < 1 || entry_count < 1 )); then
  echo "[groundedworld] NAVSIM navtrain metric cache is incomplete: $NAVSIM_METRIC_CACHE_ROOT" >&2
  exit 3
fi
echo "[groundedworld] NAVSIM navtrain metric cache entries=$entry_count"
