#!/usr/bin/env bash
set -euo pipefail
export PLANREG_PROTOCOL_VERSION=task_future_lite
export PLANREG_BENCHMARK_ATTENTION_BACKEND=eager
export PLANREG_BENCHMARK_GRADIENT_CHECKPOINTING=true
# Whole 64-candidate groups stay intact in one sidecar task per scene.
export PLANREG_BENCHMARK_SCORE_PARTITIONS=1
export PLANREG_BENCHMARK_NUM_WORKERS="${PLANREG_BENCHMARK_NUM_WORKERS:-8}"
export PLANREG_BENCHMARK_RUN_ROOT="${PLANREG_BENCHMARK_RUN_ROOT:?Set a new benchmark output root}"
export PLANREG_BENCHMARK_REPORT_ROOT="${PLANREG_BENCHMARK_REPORT_ROOT:?Set a new throughput evidence root}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/benchmark_formal_common.sh"
formal_benchmark_layout 16x4 16 4 2 4
