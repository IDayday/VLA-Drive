#!/usr/bin/env bash
# Build the exact NAVSIM-v2 metric cache required by navtest or navhard evaluation.

set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
source "$project_root/env.sh"

suite="${EVAL_SUITE:-navtest}"
case "$suite" in
  navtest|navhard_two_stage) ;;
  *) echo "[groundedworld-eval-cache] EVAL_SUITE must be navtest|navhard_two_stage" >&2; exit 2 ;;
esac
if [ "$suite" = "navhard_two_stage" ]; then
  for path in \
    "$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs" \
    "$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles"; do
    if [ ! -d "$path" ]; then
      echo "[groundedworld-eval-cache] missing navhard asset: $path" >&2
      exit 2
    fi
  done
fi

export TRAIN_TEST_SPLIT="$suite"
export CACHE_PATH="${METRIC_CACHE_PATH:-$DRIVEDREAMER_SHARED_ROOT/groundedworld_cache/navsim_v2_metric_${suite}}"
export CACHE_WORKER="${CACHE_WORKER:-single_machine_thread_pool}"
export CACHE_WORKERS="${CACHE_WORKERS:-16}"
export CACHE_USE_PROCESS_POOL="${CACHE_USE_PROCESS_POOL:-true}"
export FORCE_FEATURE_COMPUTATION="${FORCE_FEATURE_COMPUTATION:-false}"
mkdir -p "$CACHE_PATH"
bash "$NAVSIM_DEVKIT_ROOT/scripts/evaluation/run_metric_caching.sh"

metadata_count="$(find "$CACHE_PATH/metadata" -maxdepth 1 -type f -name '*.csv' | wc -l)"
entry_count="$(find "$CACHE_PATH" -type f -name 'metric_cache.pkl' | wc -l)"
if (( metadata_count < 1 || entry_count < 1 )); then
  echo "[groundedworld-eval-cache] incomplete cache: $CACHE_PATH" >&2
  exit 3
fi
echo "[groundedworld-eval-cache] suite=$suite entries=$entry_count path=$CACHE_PATH"
