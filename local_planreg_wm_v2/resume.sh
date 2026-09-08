#!/usr/bin/env bash
set -euo pipefail
: "${RESUME_CHECKPOINT:?explicit same-run V2 checkpoint required}"
exec bash "$(dirname "$0")/train.sh" "$@"
