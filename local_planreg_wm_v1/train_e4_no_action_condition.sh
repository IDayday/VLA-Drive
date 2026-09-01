#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
seed="${1:-0}"
planreg_validate_seed "${seed}" "0"
planreg_launch e4_no_action_condition "${seed}" \
  agent.world_model.future_mode=no_action_condition \
  agent.world_model.predictor_only=false
