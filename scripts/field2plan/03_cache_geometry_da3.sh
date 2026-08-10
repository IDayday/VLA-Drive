#!/usr/bin/env bash
# CPU-only conversion of existing DA3 depth pickles into strict Field2Plan cache.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
source "$project_root/env.sh"

split="${FIELD2PLAN_GEOMETRY_SPLIT:-train}"
datalist="${FIELD2PLAN_DATALIST_PATH:-$project_root/train_meta.json}"
source_root="${FIELD2PLAN_DA3_SOURCE_ROOT:-$DATA_ROOT/meta/$split}"
cache_root="${FIELD2PLAN_GEOMETRY_CACHE_ROOT:-$DRIVEDREAMER_SHARED_ROOT/field2plan_cache/geometry_da3_metric_v1}"
workers="${FIELD2PLAN_GEOMETRY_WORKERS:-112}"
max_samples="${FIELD2PLAN_GEOMETRY_MAX_SAMPLES:-0}"

for integer in workers max_samples; do
  value="${!integer}"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "[field2plan-geometry-cache] $integer must be a non-negative integer, got $value" >&2
    exit 2
  fi
done
if (( workers < 1 )); then
  echo "[field2plan-geometry-cache] workers must be positive" >&2
  exit 2
fi
available_cpus="$(getconf _NPROCESSORS_ONLN)"
if (( workers > available_cpus )); then
  echo "[field2plan-geometry-cache] workers=$workers exceeds available_cpus=$available_cpus" >&2
  exit 2
fi
for required in "$datalist" "$source_root"; do
  if [ ! -e "$required" ]; then
    echo "[field2plan-geometry-cache] missing required path: $required" >&2
    exit 2
  fi
done

resolved_cache="$(readlink -m "$cache_root")"
resolved_parent="$(readlink -m "$DRIVEDREAMER_SHARED_ROOT/field2plan_cache")"
case "$resolved_cache" in
  "$resolved_parent"/*) ;;
  *)
    echo "[field2plan-geometry-cache] cache must be below shared project directory: $resolved_parent" >&2
    exit 2
    ;;
esac
cache_root="$resolved_cache"
mkdir -p "$cache_root/logs"
run_id="${FIELD2PLAN_GEOMETRY_RUN_ID:-geometry-cache-$(date +'%Y%m%d_%H%M%S')}"
log_path="$cache_root/logs/$run_id.log"

# Each conversion worker performs zlib compression. Keep BLAS libraries from
# spawning additional threads and oversubscribing the 128-CPU DLC container.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

exec > >(tee -a "$log_path") 2>&1
echo "[field2plan-geometry-cache] project_root=$project_root"
echo "[field2plan-geometry-cache] mode=offline_cpu_conversion external_teacher_inference=disabled"
echo "[field2plan-geometry-cache] cpus=$available_cpus workers=$workers split=$split max_samples=$max_samples"
echo "[field2plan-geometry-cache] source=$source_root datalist=$datalist output=$cache_root"

if [ "${FIELD2PLAN_TOPOLOGY_ONLY:-0}" = "1" ]; then
  exit 0
fi

args=(
  --source-root "$source_root"
  --datalist "$datalist"
  --split "$split"
  --output-dir "$cache_root"
  --workers "$workers"
  --max-samples "$max_samples"
)
if [ "${FIELD2PLAN_GEOMETRY_OVERWRITE:-0}" = "1" ]; then
  args+=(--overwrite)
fi
if [ "${FIELD2PLAN_GEOMETRY_VALIDATE_ONLY:-0}" = "1" ]; then
  args+=(--validate-only)
fi

python tools/field2plan/cache_geometry_da3.py "${args[@]}"

