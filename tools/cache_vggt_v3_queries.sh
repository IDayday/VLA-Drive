#!/usr/bin/env bash
# Materialize gated layer-11-global codec latents as strict student targets.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_root/load_env.sh"

: "${VGGT_REPO:?Set VGGT_REPO in env.local.sh}"
: "${VGGT_CHECKPOINT:?Set VGGT_CHECKPOINT in env.local.sh}"
: "${VGGT_V3_CODEC:?Set VGGT_V3_CODEC in env.local.sh or env.sh}"
[[ -f "$VGGT_V3_CODEC" ]] || { echo "Missing gated V3 codec: $VGGT_V3_CODEC" >&2; exit 2; }

processes="${VGGT_CACHE_NUM_PROCESSES:-1}"
map_size_gb="${VGGT_CACHE_MAP_SIZE_GB:-16}"
args=(
  --datalist-path "$NAVSIM_DATALIST_PATH"
  --data-root "$DATA_ROOT"
  --split "${SPLIT:-train}"
  --sensor-root "$NAVSIM_TRAINVAL_SENSOR_ROOT"
  --cache-root "$NAVSIM_VGGT_V3_CACHE_ROOT"
  --vggt-repo "$VGGT_REPO"
  --vggt-checkpoint "$VGGT_CHECKPOINT"
  --native-codec "$VGGT_V3_CODEC"
  --batch-size "${VGGT_CACHE_BATCH_SIZE:-1}"
  --map-size-gb "$map_size_gb"
  --minimum-valid-ratio "${VGGT_CACHE_MIN_VALID_RATIO:-0.25}"
  --minimum-slot-variance "${VGGT_CACHE_MIN_SLOT_VARIANCE:-1e-6}"
)
[[ -z "${VGGT_CACHE_MAX_SAMPLES:-}" ]] || args+=(--max-samples "$VGGT_CACHE_MAX_SAMPLES")
[[ "${VGGT_CACHE_OVERWRITE:-0}" != "1" ]] || args+=(--overwrite)

set -x
torchrun --standalone --nnodes=1 --nproc-per-node="$processes" \
  "$DRIVEDREAMER_ROOT/tools/precompute_vggt_query_cache.py" "${args[@]}"
