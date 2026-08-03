#!/usr/bin/env bash
# Step 4: Run inference with a trained checkpoint and save per-token .npy predictions.
# Run: source env.sh && bash 4-infer.sh

set -euo pipefail

# ── Required env vars (set in env.sh) ─────────────────────────────────────────
: "${NAVSIM_EXP_ROOT:?Set NAVSIM_EXP_ROOT in env.sh}"
: "${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT in env.sh}"
: "${DRIVEDREAMER_ROOT:?Set DRIVEDREAMER_ROOT in env.sh}"
: "${DATA_ROOT:?Set DATA_ROOT in env.sh}"
: "${RELEASE_MODEL:?Set RELEASE_MODEL in env.sh}"

# ── Configuration ─────────────────────────────────────────────────────────────
SPLIT="${SPLIT:-mini}"  # mini | test | navhard_two_stage
MODEL_DIR="${MODEL_DIR:-${RELEASE_MODEL}}"
DATALIST="${DATALIST:-${DRIVEDREAMER_ROOT}/${SPLIT}_meta.json}"
OUT_DIR="${OUT_DIR:-${DRIVEDREAMER_ROOT}/navsim_planning_results}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-7}"
GPU="${GPU:-0}"
RANK="${RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-1}"
OVERWRITE="${OVERWRITE:-0}"

set -x
pwd

args=(
  --ckpt_dir "${MODEL_DIR}"
  --datalist_path "${DATALIST}"
  --data_root "${DATA_ROOT}"
  --out_dir "${OUT_DIR}"
  --split "${SPLIT}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --rank "${RANK}"
  --world_size "${WORLD_SIZE}"
  --smooth 0
)

if [[ "$OVERWRITE" == "1" ]]; then
  args+=(--overwrite)
fi

CUDA_VISIBLE_DEVICES="${GPU}" python infer.py "${args[@]}"
