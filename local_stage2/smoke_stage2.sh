#!/usr/bin/env bash

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export STAGE2_EXPERIMENT="${STAGE2_EXPERIMENT:-stage2_smoke_seed0}"
export STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-/mnt/project/DriveVLA-M0-stage2/runs/smoke/${STAGE2_EXPERIMENT}}"
export STAGE2_MAX_EPOCHS=1

exec "${script_dir}/train_stage2_full.sh" \
  trainer.params.limit_train_batches=1 \
  trainer.params.limit_val_batches=1 \
  trainer.params.num_sanity_val_steps=0 \
  +trainer.params.enable_model_summary=false \
  "$@"
