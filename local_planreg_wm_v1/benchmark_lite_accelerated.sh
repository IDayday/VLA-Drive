#!/usr/bin/env bash
# Same model/loss, explicit resource-only alternatives. No automatic formal run.
set -euo pipefail
layout="${1:?Use 16x4, 16x8, 32x2 or 32x4}"
export PLANREG_PROTOCOL_VERSION=task_future_lite
export PLANREG_BENCHMARK_ATTENTION_BACKEND=split_sdpa
export PLANREG_BENCHMARK_GRADIENT_CHECKPOINTING=false
export PLANREG_BENCHMARK_SCORE_PARTITIONS=1
export PLANREG_BENCHMARK_NUM_WORKERS="${PLANREG_BENCHMARK_NUM_WORKERS:-8}"
: "${PLANREG_ATTENTION_PARITY:?Require a real-model backend parity report}"
[[ "$(jq -r .status "${PLANREG_ATTENTION_PARITY}")" == PASS ]] || exit 2
[[ "$(jq '.blocks | length' "${PLANREG_ATTENTION_PARITY}")" == 24 ]] || exit 2
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/benchmark_formal_common.sh"
case "${layout}" in
  16x4) formal_benchmark_layout 16x4 16 4 2 4 ;;
  16x8) formal_benchmark_layout 16x8 16 8 2 8 ;;
  32x2) formal_benchmark_layout 32x2 32 2 4 2 ;;
  32x4) formal_benchmark_layout 32x4 32 4 4 4 ;;
  *) echo "Unsupported Lite benchmark layout: ${layout}" >&2; exit 2 ;;
esac
