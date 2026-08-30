#!/usr/bin/env bash

# Stage-2 recipe closest to the public evidence:
#   * released eager-attention setting;
#   * standard Lightning train-mode semantics for the frozen VLM;
#   * released AdamW decay on every action-head tensor; and
#   * shuffle-then-pad global batches matching 16 GPUs x batch 1.
#
# The exact private launcher is not released.  Every ambiguous choice remains
# overridable so controlled ablations can identify rather than hide its effect.

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export STAGE2_FLASH_ATTENTION="${STAGE2_FLASH_ATTENTION:-false}"
export STAGE2_FROZEN_BACKBONE_MODE="${STAGE2_FROZEN_BACKBONE_MODE:-train}"
export STAGE2_DECAY_NORM_AND_BIAS="${STAGE2_DECAY_NORM_AND_BIAS:-true}"
export STAGE2_PREPAD_DATASET="${STAGE2_PREPAD_DATASET:-false}"
export STAGE2_OFFICIAL_SAMPLER="${STAGE2_OFFICIAL_SAMPLER:-true}"
export STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-1}"
export STAGE2_ACCUMULATE_GRAD_BATCHES="${STAGE2_ACCUMULATE_GRAD_BATCHES:-2}"
export STAGE2_EFFECTIVE_GLOBAL_BATCH_SIZE="${STAGE2_EFFECTIVE_GLOBAL_BATCH_SIZE:-16}"
export STAGE2_SEED="${STAGE2_SEED:-2}"
export STAGE2_BASE_LR="${STAGE2_BASE_LR:-1e-4}"
export STAGE2_BASE_BATCH_SIZE="${STAGE2_BASE_BATCH_SIZE:-16}"

exec "${script_dir}/train_stage2_full.sh" "$@"
