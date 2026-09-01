#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
seed="${1:-0}"
planreg_validate_seed "${seed}" "0,1,2"
planreg_launch e2_register_qvlora "${seed}" \
  agent.world_model.enabled=false \
  agent.ema.enabled=false
