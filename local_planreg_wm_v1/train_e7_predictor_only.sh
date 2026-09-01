#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
seed="${1:-0}"
planreg_validate_seed "${seed}" "0"
planreg_launch e7_predictor_only "${seed}" \
  agent.world_model.future_mode=correct \
  agent.world_model.predictor_only=true
