#!/usr/bin/env bash
# Cache current/history Driving-JEPA through an explicit local adapter factory.
set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
source "$project_root/env.sh"

: "${GROUNDEDWORLD_DRIVING_JEPA_ADAPTER_FACTORY:?module:function is required}"
: "${GROUNDEDWORLD_DRIVING_JEPA_REPO:?local Driving-JEPA repo is required}"
: "${GROUNDEDWORLD_DRIVING_JEPA_CHECKPOINT:?local Driving-JEPA checkpoint is required}"
: "${GROUNDEDWORLD_DYNAMICS_PRIOR_CACHE:?output cache path is required}"
export GROUNDEDWORLD_DATALIST_PATH="${GROUNDEDWORLD_DATALIST_PATH:-$project_root/train_meta.json}"

torchrun --standalone --nproc_per_node="${CACHE_GPUS:-16}" \
  tools/grounded_world/cache_current_prior_adapter.py \
  --adapter-factory "$GROUNDEDWORLD_DRIVING_JEPA_ADAPTER_FACTORY" \
  --teacher-repo "$GROUNDEDWORLD_DRIVING_JEPA_REPO" \
  --checkpoint "$GROUNDEDWORLD_DRIVING_JEPA_CHECKPOINT" \
  --datalist "$GROUNDEDWORLD_DATALIST_PATH" \
  --meta-root "$DATA_ROOT/meta/train" \
  --runtime-raw-root "$OPENSCENE_DATA_ROOT" \
  --trainval-sensor-root "$NAVSIM_TRAINVAL_SENSOR_ROOT" \
  --output-dir "$GROUNDEDWORLD_DYNAMICS_PRIOR_CACHE" \
  "$@"
