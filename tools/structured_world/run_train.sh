#!/usr/bin/env bash
set -euo pipefail
# Paths supplied by the caller; no workstation-specific defaults.
: "${WORLD_ROOT:?}" "${WORLD_ARTIFACTS:?}" "${WORLD_PYTHON:?}" "${BASE_CHECKPOINT:?}" "${BASE_VLM:?}" "${NUPLAN_DEVKIT:?}"
variant=${1:?variant}; run_id=${2:?run id}; steps=${3:-1000}; seed=${4:-42}
cd "$WORLD_ROOT"
export PYTHONPATH="$WORLD_ROOT:$WORLD_ROOT/navsim:$NUPLAN_DEVKIT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
extra=()
if [[ "$variant" == D || "$variant" == E || "$variant" == E_adapter ]]; then
  extra+=(--provider-checkpoint "$WORLD_ARTIFACTS/repair1_provider/checkpoint.pt")
fi
exec "$WORLD_PYTHON" -u tools/structured_world/train.py \
 --checkpoint "$BASE_CHECKPOINT" --vlm "$BASE_VLM" \
 --data-root "$WORLD_ARTIFACTS/dataset_v1" --manifest "$WORLD_ARTIFACTS/train_tokens.json" \
 --target-cache "$WORLD_ARTIFACTS/targets_v4_train8192" --config "configs/structured_world_v1/$variant.yaml" \
 --output "$WORLD_ARTIFACTS/$run_id" --ledger "$WORLD_ARTIFACTS/budget_ledger.json" --run-id "$run_id" \
 --base-sha256 "${BASE_CHECKPOINT_SHA256:?}" --steps "$steps" --limit 8192 --seed "$seed" --save-every 100 "${extra[@]}"
