#!/usr/bin/env bash
# Step 0: Process raw NAVSIM data into the DriveDreamer-Policy format.
# Run: source env.sh && bash 0-process_data.sh

set -euo pipefail

SPLIT="${SPLIT:-test}"
DATA_ROOT="${DATA_ROOT:-navsim_dataset}"
if [ "$SPLIT" = "navtrain" ]; then
    SPLIT=train
fi

extra_args=()
if [ "${MAKE_VIDEO:-0}" = "1" ]; then
    extra_args+=(--make_video)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python navsim_data_process/make_data.py \
    --split "$SPLIT" \
    --data_root "$DATA_ROOT" \
    "${extra_args[@]}"
