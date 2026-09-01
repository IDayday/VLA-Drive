#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
seed="${1:-0}"
planreg_validate_seed "${seed}" "0"
planreg_launch e1_register_frozen "${seed}" \
  agent.vlm_config.vision_qv_lora_enabled=false \
  agent.vision_adaptation.mode=none \
  agent.world_model.enabled=false \
  agent.ema.enabled=false
