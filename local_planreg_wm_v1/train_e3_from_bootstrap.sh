#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
seed="${1:-0}"
planreg_validate_seed "${seed}" "0,1,2"
: "${BOOTSTRAP_CHECKPOINT:?Set BOOTSTRAP_CHECKPOINT to the matching 2-epoch bootstrap checkpoint}"
planreg_require_file_unless_dry_run "${BOOTSTRAP_CHECKPOINT}"
planreg_launch e3_from_bootstrap "${seed}" \
  "agent.checkpoint_path=${BOOTSTRAP_CHECKPOINT}" trainer.params.max_epochs=8 \
  agent.world_model.enabled=true agent.ema.enabled=true agent.world_model.future_mode=correct \
  agent.lr_args.planning_adapter_lr=1.0e-4 agent.lr_args.future_predictor_lr=1.0e-4 \
  agent.lr_args.fusion_lr=5.0e-5 agent.lr_args.vision_qv_lora_lr=2.0e-5 \
  agent.lr_args.action_head_lr=5.0e-5 agent.lr_args.scorer_lr=5.0e-5 \
  agent.lr_args.semantic_qformer_lr=5.0e-6 agent.scheduler_args.warmup_ratio=0.03
