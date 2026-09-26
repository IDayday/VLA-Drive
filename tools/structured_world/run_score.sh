#!/usr/bin/env bash
set -euo pipefail
: "${WORLD_ROOT:?}" "${WORLD_ARTIFACTS:?}" "${WORLD_PYTHON:?}" "${PDM_DEVKIT:?}" "${PDM_CACHE_METADATA:?}"
run=${1:?run id}
cd "$WORLD_ROOT"
export PYTHONPATH="$PDM_DEVKIT:$PDM_DEVKIT/nuplan-devkit"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "$WORLD_PYTHON" tools/structured_world/score_pdms.py --devkit "$PDM_DEVKIT" --cache-metadata "$PDM_CACHE_METADATA" --manifest "$WORLD_ARTIFACTS/dev_tokens.json" --predictions "$WORLD_ARTIFACTS/${run}_dev/predictions" --output "$WORLD_ARTIFACTS/${run}_pdms" --workers "${PDM_WORKERS:-16}"
