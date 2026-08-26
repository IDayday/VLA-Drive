#!/usr/bin/env bash
# Complete visual-unfrozen ablation: Register64 -> offline DrivoR -> PDMS + EPDMS.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

# Fixed semantic arm. Stale caller/environment values cannot silently change
# the model, selector, topology, or effective Stage-G batch.
export REGISTER64_ENABLE_SUPRIM=0
export REGISTER64_ARM=off
export REGISTER64_GENERATOR_VARIANT=visual_unfrozen
export LOCAL_NUM_PROCESSES=16
export NUM_MACHINES=1
export MACHINE_RANK=0
export NUM_PROCESSES=16
export REGISTER64_MAIN_PROCESS_PORT="${REGISTER64_MAIN_PROCESS_PORT:-29781}"
export REGISTER64_RUN_ID="${REGISTER64_RUN_ID:-register64-drivor-off-visual-unfrozen-$(date +'%Y%m%d_%H%M%S')}"

# Fine-tuned visual features must be recomputed from raw images.
unset REGISTER64_TRAIN_FEATURE_CACHE_ROOT
unset NAVSIM_FEATURE_CACHE_ROOT
unset NAVSIM_AGENT_DINO_CACHE_ROOT
unset NAVSIM_VGGT_CACHE_ROOT
export NAVSIM_USE_FEATURE_CACHE=0

exec bash "$project_root/train_register64_drivor_pipeline_dlc.sh" "$@"
