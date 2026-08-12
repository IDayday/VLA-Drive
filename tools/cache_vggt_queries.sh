#!/usr/bin/env bash
# Offline-only VGGT teacher cache generation.

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_root/load_env.sh"

: "${VGGT_REPO:?Set VGGT_REPO in env.local.sh}"
: "${VGGT_CHECKPOINT:?Set VGGT_CHECKPOINT in env.local.sh}"
if [[ ! -d "$VGGT_REPO" ]]; then
  echo "Missing local VGGT repository: $VGGT_REPO (no download is attempted)" >&2
  exit 2
fi
if [[ ! -f "$VGGT_CHECKPOINT" ]]; then
  echo "Missing local VGGT checkpoint: $VGGT_CHECKPOINT (no download is attempted)" >&2
  exit 2
fi

split="${SPLIT:-train}"
datalist="${NAVSIM_DATALIST_PATH:-$DRIVEDREAMER_ROOT/${split}_meta.json}"
processes="${VGGT_CACHE_NUM_PROCESSES:-1}"
if [[ -n "${VGGT_CACHE_MAP_SIZE_GB:-}" ]]; then
  map_size_gb="$VGGT_CACHE_MAP_SIZE_GB"
elif (( processes >= 4 )); then
  # V2 stores 195x1024 bf16 features plus compact geometry targets. With 16
  # ranks, a 16 GiB map per writer leaves ample LMDB headroom without
  # reserving a 512 GiB sparse map per device on shared storage.
  map_size_gb=16
else
  map_size_gb=64
fi
args=(
  --datalist-path "$datalist"
  --data-root "$DATA_ROOT"
  --split "$split"
  --sensor-root "$NAVSIM_TRAINVAL_SENSOR_ROOT"
  --cache-root "$NAVSIM_VGGT_CACHE_ROOT"
  --vggt-repo "$VGGT_REPO"
  --vggt-checkpoint "$VGGT_CHECKPOINT"
  --batch-size "${VGGT_CACHE_BATCH_SIZE:-1}"
  --map-size-gb "$map_size_gb"
  --minimum-valid-ratio "${VGGT_CACHE_MIN_VALID_RATIO:-0.25}"
  --minimum-slot-variance "${VGGT_CACHE_MIN_SLOT_VARIANCE:-1e-6}"
)
if [[ -n "${VGGT_CACHE_MAX_SAMPLES:-}" ]]; then
  args+=(--max-samples "$VGGT_CACHE_MAX_SAMPLES")
fi
if [[ "${VGGT_CACHE_OVERWRITE:-0}" == "1" ]]; then
  args+=(--overwrite)
fi

set -x
torchrun --standalone --nnodes=1 --nproc-per-node="$processes" \
  tools/precompute_vggt_query_cache.py "${args[@]}"
