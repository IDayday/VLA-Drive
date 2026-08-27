#!/usr/bin/env bash
# One-command formal 16-PPU Register64 CLOVER-PDMS training and evaluation.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"
pseudo_default="$project_root/navsim_exp/assets/clover_stage1_pseudo_experts/CLOVER/dataset_decoupled_v2_clean.pkl"
case "${CLOVER_PSEUDO_EXPERT_PKL:-}" in
  ""|/absolute/path/to/pseudo_experts.pkl|/absolute/path/to/official/pseudo_experts.pkl|/absolute/path/to/official-clover-pseudo-experts.pkl|/PATH/TO/pseudo_expert.pkl)
    export CLOVER_PSEUDO_EXPERT_PKL="$pseudo_default"
    ;;
esac
export LOCAL_NUM_PROCESSES="${LOCAL_NUM_PROCESSES:-16}"
export NUM_PROCESSES="${NUM_PROCESSES:-$LOCAL_NUM_PROCESSES}"
export NUM_MACHINES="${NUM_MACHINES:-1}"
export MACHINE_RANK="${MACHINE_RANK:-0}"
export CLOVER_MAIN_PROCESS_PORT="${CLOVER_MAIN_PROCESS_PORT:-29831}"
export CLOVER_NUM_CYCLES="${CLOVER_NUM_CYCLES:-30}"
export CLOVER_RUN_ID="${CLOVER_RUN_ID:-register64-clover-pdms-$(date +'%Y%m%d_%H%M%S')}"

exec bash "$project_root/train_register64_clover_pdms_pipeline_dlc.sh" "$@"
