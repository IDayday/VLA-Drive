#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${BASE_INPUTS:?Set BASE_INPUTS to the prepared TRAINING_INPUTS.json}"
: "${BASE_OUTPUT:?Set BASE_OUTPUT to a NEW IL-only experiment output}"
: "${RESOURCE_POLICY:?Set RESOURCE_POLICY to the authorized server/GPU policy JSON}"
exec "${PYTHON:-python}" -m torch.distributed.run --standalone --nproc-per-node="${NPROC_PER_NODE:-8}" \
  -m iqe.base_training train --inputs "$BASE_INPUTS" --output "$BASE_OUTPUT" \
  --resource-policy "$RESOURCE_POLICY" "$@"
