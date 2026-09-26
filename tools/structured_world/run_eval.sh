#!/usr/bin/env bash
set -euo pipefail
: "${WORLD_ROOT:?}" "${WORLD_ARTIFACTS:?}" "${WORLD_PYTHON:?}" "${BASE_CHECKPOINT:?}" "${BASE_VLM:?}" "${NUPLAN_DEVKIT:?}"
variant=${1:?}; run_id=${2:?}; output=${3:-${run_id}_dev}
cd "$WORLD_ROOT"
export PYTHONPATH="$WORLD_ROOT:$WORLD_ROOT/navsim:$NUPLAN_DEVKIT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
extra=()
if [[ "$variant" != A0 ]]; then extra+=(--delta "$WORLD_ARTIFACTS/$run_id/checkpoint.pt"); fi
exec "$WORLD_PYTHON" -u tools/structured_world/evaluate.py --checkpoint "$BASE_CHECKPOINT" --vlm "$BASE_VLM" \
 --data-root "$WORLD_ARTIFACTS/dataset_v1" --manifest "$WORLD_ARTIFACTS/dev_tokens.json" \
 --target-cache "${WORLD_DEV_TARGET_CACHE:-$WORLD_ARTIFACTS/targets_v6_dev_full}" --config "configs/structured_world_v1/$variant.yaml" \
 --output "$WORLD_ARTIFACTS/$output" "${extra[@]}"
