#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/benchmark_formal_common.sh"
export PLANREG_BENCHMARK_SCORE_PARTITIONS="${PLANREG_BENCHMARK_SCORE_PARTITIONS:-2}"
formal_benchmark_layout 16x8 16 8 2 8
