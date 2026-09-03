#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/benchmark_formal_common.sh"
formal_benchmark_layout 16x8 16 8 2 8
