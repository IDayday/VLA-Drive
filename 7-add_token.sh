#!/usr/bin/env bash
# Step 7: Add world & action special tokens to the base Qwen3-VL model.
#
# Extends the Qwen3-VL-2B-Instruct vocabulary with tokens for both
# world generation (<2d_world_*>) and action prediction (<robot_history_action_*>).
# This must be run ONCE before training. It saves a new model with the
# extended vocabulary to $TARGET_VLM, which you should then set as
# BASE_VLM in env.sh for all subsequent training runs.
#
# Run: source env.sh && bash 7-add_token.sh

set -euo pipefail

# ── Required env vars (set in env.sh) ─────────────────────────────────────────
: "${HF_HOME:?Set HF_HOME in env.sh}"

# ── Paths ─────────────────────────────────────────────────────────────────────
# Path to the original Qwen3-VL-2B-Instruct checkpoint (downloaded from HF)
: "${SOURCE_VLM:?Set SOURCE_VLM in env.sh}"

# Where to save the token-extended model (set this as BASE_VLM in env.sh afterwards)
: "${BASE_VLM:?Set BASE_VLM in env.sh}"
TARGET_VLM="${BASE_VLM}"

# Token list bundled with this repo
TOKEN_LIST="starVLA/model/modules/vlm/tools/add_qwen_special_tokens/world_tokens_all_64.txt"

set -x

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python starVLA/model/modules/vlm/tools/add_qwen_special_tokens/add_special_tokens_to_qwen.py \
  --model-id  "${SOURCE_VLM}" \
  --tokens-file "${TOKEN_LIST}" \
  --save-dir  "${TARGET_VLM}" \
  --init-strategy normal

echo ""
echo "✅ Done. Token-extended model saved to: ${TARGET_VLM}"
echo "   → Set BASE_VLM=${TARGET_VLM} in env.sh before running 8-train.sh"
