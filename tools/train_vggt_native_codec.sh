#!/usr/bin/env bash
# Teacher-only layer-11-global V3 codec and native VGGT downstream gate.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_root/load_env.sh"

: "${VGGT_REPO:?Set VGGT_REPO in env.local.sh}"
: "${VGGT_CHECKPOINT:?Set VGGT_CHECKPOINT in env.local.sh}"
: "${VGGT_V3_CODEC_ROOT:?Set VGGT_V3_CODEC_ROOT in env.local.sh or env.sh}"

for path in "$VGGT_REPO" "$DATA_ROOT" "$NAVSIM_TRAINVAL_SENSOR_ROOT"; do
  [[ -d "$path" ]] || { echo "Missing directory: $path" >&2; exit 2; }
done
for path in "$VGGT_CHECKPOINT" "$NAVSIM_DATALIST_PATH"; do
  [[ -f "$path" ]] || { echo "Missing file: $path" >&2; exit 2; }
done

processes="${VGGT_CODEC_NUM_PROCESSES:-${LOCAL_NUM_PROCESSES:-1}}"
args=(
  --datalist-path "$NAVSIM_DATALIST_PATH"
  --data-root "$DATA_ROOT"
  --sensor-root "$NAVSIM_TRAINVAL_SENSOR_ROOT"
  --vggt-repo "$VGGT_REPO"
  --vggt-checkpoint "$VGGT_CHECKPOINT"
  --output-dir "$VGGT_V3_CODEC_ROOT"
  --train-samples "${VGGT_CODEC_TRAIN_SAMPLES:-4096}"
  --validation-samples "${VGGT_CODEC_VALIDATION_SAMPLES:-512}"
  --batch-size "${VGGT_CODEC_BATCH_SIZE:-1}"
  --steps "${VGGT_CODEC_STEPS:-10000}"
  --learning-rate "${VGGT_CODEC_LEARNING_RATE:-2e-4}"
  --checkpoint-interval "${VGGT_CODEC_CHECKPOINT_INTERVAL:-250}"
)
if [[ "${VGGT_CODEC_OVERWRITE:-0}" == "1" ]]; then
  args+=(--overwrite)
elif [[ "${VGGT_CODEC_RESUME:-1}" == "1" ]]; then
  args+=(--resume)
fi

set -x
torchrun --standalone --nnodes=1 --nproc-per-node="$processes" \
  "$DRIVEDREAMER_ROOT/tools/train_vggt_native_codec.py" "${args[@]}"
