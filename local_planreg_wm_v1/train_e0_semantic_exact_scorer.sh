#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
seed="${1:-0}"
planreg_validate_seed "${seed}" "0,1,2"
planreg_launch e0_semantic_exact_scorer "${seed}" \
  agent.vlm_config.planning_registers_enabled=false \
  agent.vlm_config.vision_qv_lora_enabled=false \
  agent.vision_adaptation.mode=none \
  agent.scene_fusion.mode=semantic_only \
  agent.world_model.enabled=false \
  agent.ema.enabled=false
