#!/usr/bin/env bash
set -euo pipefail
export PLANREG_PROTOCOL_VERSION=task_future_lite
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/formal_launch_common.sh"
formal_launch base "${1:-0}"
