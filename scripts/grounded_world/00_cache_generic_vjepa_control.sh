#!/usr/bin/env bash
set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
source "$project_root/env.sh"

: "${VJEPA2_REPO:?local V-JEPA repo is required}"
: "${VJEPA2_CHECKPOINT:?local V-JEPA checkpoint is required}"
: "${GROUNDEDWORLD_DYNAMICS_PRIOR_CACHE:?output cache path is required}"
export GROUNDEDWORLD_DATALIST_PATH="${GROUNDEDWORLD_DATALIST_PATH:-$project_root/train_meta.json}"

torchrun --standalone --nproc_per_node="${CACHE_GPUS:-16}" \
  tools/grounded_world/cache_current_prior_vjepa.py \
  --vjepa-repo "$VJEPA2_REPO" \
  --checkpoint "$VJEPA2_CHECKPOINT" \
  --datalist "$GROUNDEDWORLD_DATALIST_PATH" \
  --meta-root "$DATA_ROOT/meta/train" \
  --runtime-raw-root "$OPENSCENE_DATA_ROOT" \
  --trainval-sensor-root "$NAVSIM_TRAINVAL_SENSOR_ROOT" \
  --output-dir "$GROUNDEDWORLD_DYNAMICS_PRIOR_CACHE" \
  "$@"
