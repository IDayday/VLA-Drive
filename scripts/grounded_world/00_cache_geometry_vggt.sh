#!/usr/bin/env bash
# Reuse the existing strict local-only VGGT cache implementation.
set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
source "$project_root/env.sh"
export GROUNDEDWORLD_GEOMETRY_CACHE="${GROUNDEDWORLD_GEOMETRY_CACHE:-$DRIVEDREAMER_SHARED_ROOT/groundedworld_cache/vggt}"
export GROUNDEDWORLD_DATALIST_PATH="${GROUNDEDWORLD_DATALIST_PATH:-$project_root/train_meta.json}"
export FIELD2PLAN_VGGT_CACHE="$GROUNDEDWORLD_GEOMETRY_CACHE"
export FIELD2PLAN_DATALIST_PATH="$GROUNDEDWORLD_DATALIST_PATH"
exec bash "$project_root/scripts/field2plan/06_cache_geometry_vggt.sh" "$@"
