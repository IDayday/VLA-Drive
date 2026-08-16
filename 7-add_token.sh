#!/usr/bin/env bash
# Step 7: Add world, mine-agent, and action special tokens to the base Qwen3-VL model.
#
# Extends the Qwen3-VL-2B-Instruct vocabulary with tokens for both
# world generation (<2d_world_*>) , agent-query alignment (<mine_agent_*>) and action prediction (<robot_history_action_*>)
# This must be run ONCE before training. It saves a new model with the
# extended vocabulary to $TARGET_VLM, which you should then set as
# BASE_VLM in env.sh for all subsequent training runs.
#
# Run: source env.sh && bash 7-add_token.sh

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

# ── Required env vars (set in env.sh) ─────────────────────────────────────────
: "${HF_HOME:?Set HF_HOME in env.sh}"

# ── Paths ─────────────────────────────────────────────────────────────────────
# Path to the original Qwen3-VL-2B-Instruct checkpoint (downloaded from HF)
: "${SOURCE_VLM:?Set SOURCE_VLM in env.sh}"

# Where to save the token-extended model (set this as BASE_VLM in env.sh afterwards)
: "${BASE_VLM:?Set BASE_VLM in env.sh}"
TARGET_VLM="${TARGET_VLM:-${BASE_VLM}}"

# Token lists bundled with this repo
WORLD_TOKEN_LIST="starVLA/model/modules/vlm/tools/add_qwen_special_tokens/world_tokens_all_64.txt"
MINE_AGENT_TOKEN_LIST="starVLA/model/modules/vlm/tools/add_qwen_special_tokens/mine_agent_tokens_32.txt"
VGGT_TOKEN_LIST="starVLA/model/modules/vlm/tools/add_qwen_special_tokens/vggt_tokens_195.txt"
COMBINED_TOKEN_LIST="$(mktemp /tmp/qwen_special_tokens.XXXXXX)"
trap 'rm -f "${COMBINED_TOKEN_LIST}"' EXIT

: > "${COMBINED_TOKEN_LIST}"
cat "${WORLD_TOKEN_LIST}" >> "${COMBINED_TOKEN_LIST}"
printf "\n" >> "${COMBINED_TOKEN_LIST}"
cat "${MINE_AGENT_TOKEN_LIST}" >> "${COMBINED_TOKEN_LIST}"
printf "\n" >> "${COMBINED_TOKEN_LIST}"
cat "${VGGT_TOKEN_LIST}" >> "${COMBINED_TOKEN_LIST}"

set -x

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python starVLA/model/modules/vlm/tools/add_qwen_special_tokens/add_special_tokens_to_qwen.py \
  --model-id  "${SOURCE_VLM}" \
  --tokens-file "${COMBINED_TOKEN_LIST}" \
  --save-dir  "${TARGET_VLM}" \
  --init-strategy normal

echo ""
echo "✅ Done. Token-extended model saved to: ${TARGET_VLM}"
echo "   → Set BASE_VLM=${TARGET_VLM} in env.sh before running 8-train.sh"
