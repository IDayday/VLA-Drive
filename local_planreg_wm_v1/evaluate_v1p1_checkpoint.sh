#!/usr/bin/env bash
set -euo pipefail
export PLANREG_PROTOCOL_VERSION=v1p1
: "${PLANREG_FORMAL_EVAL_ROOT:?Set a new immutable evaluation output root}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/evaluate_formal_checkpoint.sh" "$@"
