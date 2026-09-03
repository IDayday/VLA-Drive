#!/usr/bin/env bash
set -euo pipefail
export PLANREG_BENCHMARK_GRADIENT_CHECKPOINTING=false
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/benchmark_formal_common.sh"
formal_benchmark_layout 16x6 16 6 2 8
