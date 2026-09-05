#!/usr/bin/env bash
set -euo pipefail
export PLANREG_PROTOCOL_VERSION=task_future_lite
export PLANREG_PEER_HOST="${PLANREG_PEER_HOST:-training-rl-zt4}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/formal_launch_common.sh"
formal_launch driving_vqa "${1:-0}"
