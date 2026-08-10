#!/usr/bin/env bash
set -Eeuo pipefail
export GROUNDEDWORLD_STAGE=stage2
export GROUNDEDWORLD_FUTURE_ENABLED=1
export GROUNDEDWORLD_WORLD_ACCESS=0
export GROUNDEDWORLD_CONSEQUENCE_ENABLED=0
exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/train_stage_common.sh"
