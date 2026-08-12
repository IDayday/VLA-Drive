#!/usr/bin/env bash
# Step 3: Resumably generate the three NAVSIM view videos in parallel.
# Run: source env.sh && SPLIT=mini bash 3-gen_videos.sh

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

SPLIT="${SPLIT:-mini}"
DATA_ROOT="${DATA_ROOT:-navsim_dataset}"
VIDEO_WORKERS="${VIDEO_WORKERS:-16}"
VIDEO_ENCODER_PRESET="${VIDEO_ENCODER_PRESET:-medium}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
if [ "$SPLIT" = "navtrain" ]; then
    SPLIT=train
fi

python navsim_data_process/make_videos.py \
    --split "$SPLIT" \
    --data_root "$DATA_ROOT" \
    --datalist "${DRIVEDREAMER_ROOT}/${SPLIT}_meta.json" \
    --workers "$VIDEO_WORKERS" \
    --max_samples "$MAX_SAMPLES" \
    --encoder-preset "$VIDEO_ENCODER_PRESET"
