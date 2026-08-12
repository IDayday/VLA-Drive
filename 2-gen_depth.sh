#!/usr/bin/env bash
# Step 2: Generate monocular depth maps for all scenes using Depth-Anything-3.
# Requires conda activate img_process before running.
# Run: source env.sh && bash 2-gen_depth.sh

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

SPLIT="${SPLIT:-mini}"
DATA_ROOT="${DATA_ROOT:-navsim_dataset}"
if [ "$SPLIT" = "navtrain" ]; then
    SPLIT=train
fi
PROJECT_DIR="$DRIVEDREAMER_ROOT"
WORLD_SIZE="${WORLD_SIZE:-1}"
RANK="${RANK:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
DEPTH_LOG_FILE="${DEPTH_LOG_FILE:-}"

export PYTHONPATH="$PROJECT_DIR/depth_process/Depth-Anything-3/src:$PYTHONPATH"
export HF_HUB_OFFLINE=1
export DA3_LOG_LEVEL="${DA3_LOG_LEVEL:-WARN}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"
: "${DA3_MODEL:?Set DA3_MODEL in env.sh}"

cd depth_process
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
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        python depth.py "${args[@]}" >>"$DEPTH_LOG_FILE" 2>&1
else
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        python depth.py "${args[@]}"
fi
