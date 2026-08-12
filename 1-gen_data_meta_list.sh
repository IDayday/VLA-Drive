#!/usr/bin/env bash
# Step 1: Generate the data meta-list JSON used by the dataloader.
# Run: source env.sh && bash 1-gen_data_meta_list.sh

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

SPLIT="${SPLIT:-test}"
DATA_ROOT="${DATA_ROOT:-navsim_dataset}"
if [ "$SPLIT" = "navtrain" ]; then
    SPLIT=train
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python navsim_data_process/data_list.py \
    --split "$SPLIT" \
    --data_root "$DATA_ROOT"
