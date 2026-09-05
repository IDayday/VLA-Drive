#!/usr/bin/env bash
set -euo pipefail
export PLANREG_PROTOCOL_VERSION=v1p1
# Keep eager until split-SDPA forward/backward parity has been run in this environment.
export PLANREG_BENCHMARK_ATTENTION_BACKEND="${PLANREG_BENCHMARK_ATTENTION_BACKEND:-eager}"
export PLANREG_BENCHMARK_RUN_ROOT="${PLANREG_BENCHMARK_RUN_ROOT:-/mnt/project/DriveVLA-M0-formal-runs/v1p1_throughput}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/benchmark_formal_common.sh"
export PLANREG_BENCHMARK_REPORT_ROOT="${PLANREG_BENCHMARK_REPORT_ROOT:-${PLANREG_REPO_ROOT}/reports/planreg_wm_v1p1/throughput}"
case "${1:-}" in
  8x8) formal_benchmark_layout 8x8 8 8 1 8 ;;
  16x4) formal_benchmark_layout 16x4 16 4 2 4 ;;
  16x8) formal_benchmark_layout 16x8 16 8 2 4 ;;
  *) echo 'Usage: benchmark_v1p1_layout.sh 8x8|16x4|16x8' >&2; exit 2 ;;
esac
