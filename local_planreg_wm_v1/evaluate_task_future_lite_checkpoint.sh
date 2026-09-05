#!/usr/bin/env bash
set -euo pipefail
export PLANREG_PROTOCOL_VERSION=task_future_lite
: "${PLANREG_FORMAL_EVAL_ROOT:?Set a NEW immutable evaluation output root}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/evaluate_formal_checkpoint.sh" "$@"
