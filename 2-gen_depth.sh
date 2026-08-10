#!/usr/bin/env bash
# Step 2: Generate monocular depth maps for NAVSIM scenes using Depth-Anything-3.
# Requires conda activate img_process before running.
#
# Single GPU:
#   source env.sh
#   SPLIT=test CUDA_VISIBLE_DEVICES=0 bash 2-gen_depth.sh
#
# Multi GPU, one process per visible GPU. On a 16-card DLC node this is enough:
#   source env.sh
#   SPLIT=test bash 2-gen_depth.sh
#
# Explicit 16-card run:
#   source env.sh
#   SPLIT=test NUM_GPUS=16 bash 2-gen_depth.sh
#
# Useful path overrides for the current DLC layout:
#   OPENSCENE_DATA_ROOT=/mnt/workspace/Public_Space/navsim
#   NAVSIM_SENSOR_BLOBS_ROOT=/mnt/workspace/Public_Space/navsim/test_sensor_blobs
#   DA3_MODEL=/mnt/workspace/VLA_Group/LLM_weight/depth-anything/da3metric-large

set -euo pipefail

SPLIT="${SPLIT:-mini}"
DATA_ROOT="${DATA_ROOT:-navsim_dataset}"
if [ "$SPLIT" = "navtrain" ]; then
    SPLIT=train
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Make DATA_ROOT absolute before child processes cd into depth_process/.
if [[ "$DATA_ROOT" != /* ]]; then
    DATA_ROOT="$PROJECT_DIR/$DATA_ROOT"
fi

MAX_SAMPLES="${MAX_SAMPLES:-0}"
DEPTH_LOG_DIR="${DEPTH_LOG_DIR:-$PROJECT_DIR/logs}"
DEPTH_LOG_FILE="${DEPTH_LOG_FILE:-}"

export PYTHONPATH="$PROJECT_DIR/depth_process/Depth-Anything-3/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export DA3_LOG_LEVEL="${DA3_LOG_LEVEL:-WARN}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"
: "${DA3_MODEL:?Set DA3_MODEL in env.sh or pass DA3_MODEL=/path/to/da3metric-large}"

_detect_gpu_count() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        python - <<'PYCOUNT_VISIBLE'
import os
visible = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
print(len(visible) if visible else 1)
PYCOUNT_VISIBLE
    else
        python - <<'PYCOUNT_TORCH'
try:
    import torch
    print(torch.cuda.device_count() or 1)
except Exception:
    print(1)
PYCOUNT_TORCH
    fi
}

NUM_GPUS="${NUM_GPUS:-$(_detect_gpu_count)}"
WORLD_SIZE="${WORLD_SIZE:-$NUM_GPUS}"
RANK="${RANK:-0}"

# Parent launcher: start one child per visible GPU/rank, then wait.
if [[ "${DEPTH_SHARD_CHILD:-0}" != "1" && "$NUM_GPUS" -gt 1 ]]; then
    mkdir -p "$DEPTH_LOG_DIR"
    echo "[depth] split=$SPLIT data_root=$DATA_ROOT world_size=$NUM_GPUS logs=$DEPTH_LOG_DIR"
    for rank in $(seq 0 $((NUM_GPUS - 1))); do
        log_file="$DEPTH_LOG_DIR/depth_${SPLIT}_rank${rank}.log"
        echo "[depth] launch rank=$rank gpu=$rank log=$log_file"
        DEPTH_SHARD_CHILD=1 \
        WORLD_SIZE="$NUM_GPUS" \
        RANK="$rank" \
        NUM_GPUS=1 \
        CUDA_VISIBLE_DEVICES="$rank" \
        DEPTH_LOG_FILE="$log_file" \
            bash "$PROJECT_DIR/2-gen_depth.sh" &
    done
    wait
    echo "[depth] all ranks finished"
    exit 0
fi

cd "$PROJECT_DIR/depth_process"
args=(
    --split "$SPLIT"
    --data_root "$DATA_ROOT"
    --datalist "$PROJECT_DIR/${SPLIT}_meta.json"
    --meta_dir "$DATA_ROOT/meta/$SPLIT"
    --world_size "$WORLD_SIZE"
    --rank "$RANK"
    --max_samples "$MAX_SAMPLES"
)

if [[ -n "$DEPTH_LOG_FILE" ]]; then
    mkdir -p "$(dirname "$DEPTH_LOG_FILE")"
    echo "[depth] rank=$RANK/$WORLD_SIZE cuda=${CUDA_VISIBLE_DEVICES:-0} start $(date -Is)" >>"$DEPTH_LOG_FILE"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        python depth.py "${args[@]}" >>"$DEPTH_LOG_FILE" 2>&1
    echo "[depth] rank=$RANK/$WORLD_SIZE done $(date -Is)" >>"$DEPTH_LOG_FILE"
else
    echo "[depth] rank=$RANK/$WORLD_SIZE cuda=${CUDA_VISIBLE_DEVICES:-0}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        python depth.py "${args[@]}"
fi
