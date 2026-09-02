#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/formal_launch_common.sh"
formal_launch base "${1:-0}"
