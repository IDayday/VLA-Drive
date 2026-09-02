#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
seed="${1:-0}"
planreg_validate_seed "${seed}" "0,1,2"
planreg_launch bootstrap_registers "${seed}" \
  trainer.params.max_epochs=2 \
  agent.world_model.enabled=false \
  agent.ema.enabled=false \
  agent.lr_args.planning_adapter_lr=2.0e-4 \
  agent.lr_args.future_predictor_lr=2.0e-4 \
  agent.lr_args.fusion_lr=1.0e-4 \
  agent.lr_args.vision_qv_lora_lr=2.0e-5 \
  agent.lr_args.action_head_lr=2.0e-5 \
  agent.lr_args.scorer_lr=2.0e-5 \
  agent.lr_args.semantic_qformer_lr=5.0e-6 \
  agent.scheduler_args.warmup_ratio=0.05
