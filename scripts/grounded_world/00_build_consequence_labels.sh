#!/usr/bin/env bash
set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
source "$project_root/env.sh"

: "${NAVSIM_METRIC_CACHE_ROOT:?NAVSIM metric cache is required}"
: "${GROUNDEDWORLD_CONSEQUENCE_CACHE:?output cache path is required}"
export GROUNDEDWORLD_CONSEQUENCE_PROVIDER_FACTORY="${GROUNDEDWORLD_CONSEQUENCE_PROVIDER_FACTORY:-starVLA.model.modules.grounded_world.navsim_consequence_provider:build_navsim_nonreactive_provider}"
export GROUNDEDWORLD_DATALIST_PATH="${GROUNDEDWORLD_DATALIST_PATH:-$project_root/train_meta.json}"

python tools/grounded_world/build_consequence_labels.py \
  --provider-factory "$GROUNDEDWORLD_CONSEQUENCE_PROVIDER_FACTORY" \
  --metric-cache-root "$NAVSIM_METRIC_CACHE_ROOT" \
  --datalist "$GROUNDEDWORLD_DATALIST_PATH" \
  --meta-root "$DATA_ROOT/meta/train" \
  --output-dir "$GROUNDEDWORLD_CONSEQUENCE_CACHE" \
  "$@"
