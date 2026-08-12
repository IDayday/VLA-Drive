#!/usr/bin/env bash
# Read-only VGGT layer-11 geometry probe; never writes or mutates feature caches.

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
output="${VGGT_GEOMETRY_PROBE_OUTPUT:-$NAVSIM_EXP_ROOT/vggt_geometry_probe/probe_${timestamp}.json}"
args=(
  --datalist-path "${NAVSIM_DATALIST_PATH:-$DRIVEDREAMER_ROOT/train_meta.json}"
  --data-root "$DATA_ROOT"
  --sensor-root "$NAVSIM_TRAINVAL_SENSOR_ROOT"
  --vggt-repo "$VGGT_REPO"
  --vggt-checkpoint "$VGGT_CHECKPOINT"
  --output "$output"
  --split "${SPLIT:-train}"
  --train-samples "${VGGT_GEOMETRY_PROBE_TRAIN_SAMPLES:-96}"
  --val-samples "${VGGT_GEOMETRY_PROBE_VAL_SAMPLES:-32}"
  --grid-rows "${VGGT_GEOMETRY_PROBE_GRID_ROWS:-6}"
  --grid-cols "${VGGT_GEOMETRY_PROBE_GRID_COLS:-10}"
  --lidar-min-points "${VGGT_GEOMETRY_PROBE_LIDAR_MIN_POINTS:-3}"
  --ridge-alpha "${VGGT_GEOMETRY_PROBE_RIDGE_ALPHA:-10.0}"
  --seed "${VGGT_GEOMETRY_PROBE_SEED:-20260811}"
  --device "${VGGT_GEOMETRY_PROBE_DEVICE:-auto}"
)
if [[ "${VGGT_GEOMETRY_PROBE_SKIP_CHECKPOINT_HASH:-0}" == "1" ]]; then
  args+=(--skip-checkpoint-hash)
fi

cd "$project_root"
set -x
python tools/probe_vggt_geometry_signal.py "${args[@]}" "$@"
