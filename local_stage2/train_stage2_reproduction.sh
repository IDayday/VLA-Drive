#!/usr/bin/env bash

# Stage-2 recipe closest to the public evidence:
#   * released eager-attention setting;
#   * standard Lightning train-mode semantics for the frozen VLM;
#   * released AdamW decay on every action-head tensor; and
#   * the recovered five-second long-trajectory target;
#   * the source warmup-cosine schedule selected by the current evidence; and
#   * shuffle-then-pad global batches matching 16 GPUs x batch 1.
#
# The exact private launcher is not released.  Every ambiguous choice remains
# overridable so controlled ablations can identify rather than hide its effect.

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
requested_feature_cache="${DRIVEVLA_NAVTRAIN_FEATURE_CACHE-}"
source "${script_dir}/common.sh"

if [[ -z "${requested_feature_cache}" ]]; then
  export DRIVEVLA_NAVTRAIN_FEATURE_CACHE="${DRIVEVLA_NAVTRAIN_LONG2_FEATURE_CACHE}"
fi

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
export STAGE2_SCHEDULER="${STAGE2_SCHEDULER:-source_cosine}"
export STAGE2_LONG_TRAJECTORY_ADDITIONAL_POSES="${STAGE2_LONG_TRAJECTORY_ADDITIONAL_POSES:-2}"
export STAGE2_REQUIRE_LIGHTNING_VERSION="${STAGE2_REQUIRE_LIGHTNING_VERSION:-2.2.1}"
export STAGE2_REQUIRE_TRANSFORMERS_VERSION="${STAGE2_REQUIRE_TRANSFORMERS_VERSION:-4.48.3}"

exec "${script_dir}/train_stage2_full.sh" "$@"
