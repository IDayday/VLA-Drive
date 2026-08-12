#!/usr/bin/env bash
# One-pass VGGT spatial-resolution probe; never writes or mutates feature caches.

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

timestamp="$(date +%Y%m%d_%H%M%S)"
output="${VGGT_RESOLUTION_PROBE_OUTPUT:-$NAVSIM_EXP_ROOT/vggt_resolution_probe/probe_${timestamp}.json}"
args=(
  --datalist-path "${NAVSIM_DATALIST_PATH:-$DRIVEDREAMER_ROOT/train_meta.json}"
  --data-root "$DATA_ROOT"
  --split "${SPLIT:-train}"
  --sensor-root "$NAVSIM_TRAINVAL_SENSOR_ROOT"
  --vggt-repo "$VGGT_REPO"
  --vggt-checkpoint "$VGGT_CHECKPOINT"
  --output "$output"
  --max-samples "${VGGT_RESOLUTION_PROBE_SAMPLES:-1024}"
  --batch-size "${VGGT_RESOLUTION_PROBE_BATCH_SIZE:-1}"
  --device "${VGGT_RESOLUTION_PROBE_DEVICE:-auto}"
)
if [[ "${VGGT_RESOLUTION_PROBE_SKIP_CHECKPOINT_HASH:-0}" == "1" ]]; then
  args+=(--skip-checkpoint-hash)
fi

set -x
python tools/probe_vggt_spatial_resolution.py "${args[@]}" "$@"
