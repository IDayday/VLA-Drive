#!/usr/bin/env bash
# Stable absolute entrypoint for a non-interactive one-node/16-PPU DLC job.

set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${project_root}/scripts/field2plan/10_eval_all_ckpts_16gpu.sh" "$@"
