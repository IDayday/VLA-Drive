#!/usr/bin/env bash
# Build a separate Qwen checkpoint with world/action and 15 VGGT global tokens.

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

: "${VGGT_BASE_VLM:?Set VGGT_BASE_VLM in env.local.sh}"
source_vlm="${VGGT_SOURCE_VLM:-$BASE_VLM}"
if [[ ! -d "$source_vlm" ]]; then
  echo "Missing local VGGT source VLM: $source_vlm (no download is attempted)" >&2
  exit 2
fi

world_tokens="$DRIVEDREAMER_ROOT/starVLA/model/modules/vlm/tools/add_qwen_special_tokens/world_tokens_all_64.txt"
vggt_tokens="$DRIVEDREAMER_ROOT/starVLA/model/modules/vlm/tools/add_qwen_special_tokens/vggt_global_query_tokens_15.txt"
combined_tokens="$(mktemp /tmp/qwen_vggt_tokens.XXXXXX)"
trap 'rm -f "$combined_tokens"' EXIT

cp "$world_tokens" "$combined_tokens"
printf '\n' >> "$combined_tokens"
sed '/^[[:space:]]*$/d' "$vggt_tokens" >> "$combined_tokens"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python \
  starVLA/model/modules/vlm/tools/add_qwen_special_tokens/add_special_tokens_to_qwen.py \
  --model-id "$source_vlm" \
  --tokens-file "$combined_tokens" \
  --save-dir "$VGGT_BASE_VLM" \
  --init-strategy normal \
  --device "${VGGT_TOKEN_DEVICE:-cpu}" \
  --attn-implementation "${VLM_ATTN_IMPLEMENTATION:-sdpa}"

echo "VGGT-token VLM saved to: $VGGT_BASE_VLM"
