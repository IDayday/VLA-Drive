#!/usr/bin/env bash
# Complete matched arm B: arm A + DriveSuprim dynamic Top-32 refinement.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"
export REGISTER64_ENABLE_SUPRIM=1
export REGISTER64_ARM=on
export REGISTER64_GENERATOR_VARIANT=frozen
export LOCAL_NUM_PROCESSES="${LOCAL_NUM_PROCESSES:-16}"
export NUM_MACHINES="${NUM_MACHINES:-1}"
export MACHINE_RANK="${MACHINE_RANK:-0}"
export NUM_PROCESSES="${NUM_PROCESSES:-$((LOCAL_NUM_PROCESSES * NUM_MACHINES))}"
export REGISTER64_MAIN_PROCESS_PORT="${REGISTER64_MAIN_PROCESS_PORT:-29761}"
export REGISTER64_RUN_ID="${REGISTER64_RUN_ID:-register64-drivor-suprim-on-$(date +'%Y%m%d_%H%M%S')}"

exec bash "$project_root/train_register64_drivor_pipeline_dlc.sh" "$@"
