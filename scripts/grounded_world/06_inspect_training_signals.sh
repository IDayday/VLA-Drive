#!/usr/bin/env bash
set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

: "${RUN_DIR:?GroundedWorld run directory is required}"
: "${GROUNDEDWORLD_AUDIT_STAGE:?prior, predictive, or planning is required}"

python tools/grounded_world/summarize_training_signals.py \
  --run-dir "$RUN_DIR" \
  --stage "$GROUNDEDWORLD_AUDIT_STAGE" \
  --window "${GROUNDEDWORLD_AUDIT_WINDOW:-20}" \
  --output "${GROUNDEDWORLD_AUDIT_OUTPUT:-$RUN_DIR/training_signal_audit.md}"
