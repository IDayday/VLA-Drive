#!/usr/bin/env bash
set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

: "${RESULT_MATRIX:?JSON matrix of actual summary paths is required}"
: "${REFERENCE_ARM:?reference arm, normally b0_pure_vlm_dit, is required}"
: "${RESULT_REPORT_DIR:?output report directory is required}"

python tools/grounded_world/aggregate_navsim_results.py \
  --matrix "$RESULT_MATRIX" \
  --reference-arm "$REFERENCE_ARM" \
  --output-dir "$RESULT_REPORT_DIR" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES:-10000}" \
  --seed "${BOOTSTRAP_SEED:-20260810}"
