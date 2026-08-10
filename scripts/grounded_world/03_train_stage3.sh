#!/usr/bin/env bash
set -Eeuo pipefail
export GROUNDEDWORLD_STAGE=stage3
exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/train_stage_common.sh"
