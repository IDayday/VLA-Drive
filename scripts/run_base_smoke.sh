#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-drivevla_base_no_memory_smoke}"
export MAX_SCENES="${MAX_SCENES:-1}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export NUM_WORKERS="${NUM_WORKERS:-1}"
export NUM_GPUS="${NUM_GPUS:-1}"

exec "${SCRIPT_DIR}/run_base_pdms.sh" worker=sequential "$@"
