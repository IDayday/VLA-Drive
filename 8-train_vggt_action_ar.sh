#!/usr/bin/env bash
# Step 8 (train): Autoregressive VGGT latent COT + action training.
# It reuses the VGGT action training launcher and only switches the VGGT mode.
# Run: bash 8-train_vggt_action_ar.sh

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export VGGT_MODE="${VGGT_MODE:-autoregressive}"

if [ -z "${RUN_ID:-}" ]; then
  timestamp="$(date +"%m%d_%H")"
  bz="${PER_DEVICE_BATCH_SIZE:-2}"
  gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-1}"
  export RUN_ID="${timestamp}-vggt-ar-action-lr1e5-16g-bz_${bz}-ga_${gradient_accumulation_steps}-train"
fi

exec bash "${project_root}/8-train_vggt_action.sh"
