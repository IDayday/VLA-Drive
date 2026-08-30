#!/usr/bin/env bash

# Detached launcher for the controlled Stage-2 reproduction defaults.

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export STAGE2_TRAIN_ENTRYPOINT="${script_dir}/train_stage2_reproduction.sh"
export STAGE2_EXPERIMENT="${STAGE2_EXPERIMENT:-stage2_reproduction_seed2}"
export STAGE2_EVAL_EXPERIMENT="${STAGE2_EVAL_EXPERIMENT:-${STAGE2_EXPERIMENT}_navtest}"

exec "${script_dir}/launch_stage2_full.sh" "$@"
