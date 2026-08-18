#!/usr/bin/env bash
# Formal V3 student/planner training. VGGT itself is never imported here.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

export NAVSIM_VGGT_CACHE_ROOT="$NAVSIM_VGGT_V3_CACHE_ROOT"
export VGGT_EXPERIMENT_OVERLAY="$DRIVEDREAMER_ROOT/starVLA/config/training/vggt_query_v3.yaml"
export VGGT_PLANNER_VERSION=3
export RUN_ID="${RUN_ID:-vggt-query-v3-layer11-global-codec-m195-${PAI_JOB_ID:-$(date +'%Y%m%d_%H%M%S')}}"

exec bash "$DRIVEDREAMER_ROOT/8-train_vggt_action.sh"
