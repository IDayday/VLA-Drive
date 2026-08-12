#!/usr/bin/env bash
# Step 3: Print dataset statistics (number of frames, action distribution, etc.).
# Run: source env.sh && bash 3-stat_data.sh

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

SPLIT="${SPLIT:-mini}"
DATA_ROOT="${DATA_ROOT:-navsim_dataset}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python navsim_data_process/data_stat.py \
    --split "$SPLIT" \
    --data_root "$DATA_ROOT"
