#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=load_env.sh
source "$SCRIPT_DIR/load_env.sh"

# V2-B is the clean source for this gate: its student received VGGT
# supervision, while access_enabled=false kept its DiT on the action-only path.
V3_SOURCE_RUN="${V3_SOURCE_RUN:-$NAVSIM_EXP_ROOT/vggt-v2-control-supervision-no-access-v2cache-20260812}"
V3_SOURCE_STEP="${V3_SOURCE_STEP:-80000}"
V3_BASE_VLM="${V3_BASE_VLM:-$DRIVEDREAMER_ROOT/weights/derived/Qwen3-VL-2B-VGGTAction}"
V3_VGGT_CACHE_ROOT="${V3_VGGT_CACHE_ROOT:-$DRIVEDREAMER_ROOT/navsim_feature_cache/vggt_query_train_v2_layer11_global_m195}"
V3_OUTPUT_DIR="${V3_OUTPUT_DIR:-$NAVSIM_EXP_ROOT/vggt-local-gates/v3-b80-centered-gate-seed20260813}"
V3_FEATURE_FILE="${V3_FEATURE_FILE:-$NAVSIM_EXP_ROOT/vggt-local-gates/v3-b80-features-384t-96v.pt}"
V3_DEVICE="${V3_DEVICE:-cuda:0}"

python "$DRIVEDREAMER_ROOT/tools/validate_vggt_v3_gate.py" \
  --run-dir "$V3_SOURCE_RUN" \
  --checkpoint-step "$V3_SOURCE_STEP" \
  --base-vlm "$V3_BASE_VLM" \
  --vggt-cache-root "$V3_VGGT_CACHE_ROOT" \
  --datalist-path "$NAVSIM_DATALIST_PATH" \
  --data-root "$DATA_ROOT" \
  --output-dir "$V3_OUTPUT_DIR" \
  --feature-file "$V3_FEATURE_FILE" \
  --device "$V3_DEVICE" \
  --train-samples 384 \
  --validation-samples 96 \
  --feature-batch-size 12 \
  --batch-size 8 \
  --validation-batch-size 8 \
  --workers 4 \
  --steps 1000 \
  --margin-fraction 0.05 \
  --margin-weight 10 \
  --log-interval 25
