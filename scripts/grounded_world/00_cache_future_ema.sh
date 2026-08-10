#!/usr/bin/env bash
set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
source "$project_root/env.sh"

: "${GROUNDEDWORLD_STAGE1_CHECKPOINT:?Stage-I checkpoint is required}"
: "${GROUNDEDWORLD_FUTURE_TARGET_CACHE:?output cache path is required}"
export GROUNDEDWORLD_DATALIST_PATH="${GROUNDEDWORLD_DATALIST_PATH:-$project_root/train_meta.json}"

torchrun --standalone --nproc_per_node="${CACHE_GPUS:-16}" \
  tools/grounded_world/cache_future_student_ema.py \
  --config starVLA/config/training/cfg_groundedworld_stage1.yaml \
  --stage1-checkpoint "$GROUNDEDWORLD_STAGE1_CHECKPOINT" \
  --datalist "$GROUNDEDWORLD_DATALIST_PATH" \
  --meta-root "$DATA_ROOT/meta/train" \
  --runtime-raw-root "$OPENSCENE_DATA_ROOT" \
  --trainval-sensor-root "$NAVSIM_TRAINVAL_SENSOR_ROOT" \
  --output-dir "$GROUNDEDWORLD_FUTURE_TARGET_CACHE" \
  "$@"
