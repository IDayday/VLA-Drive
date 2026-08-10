#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export FIELD2PLAN_PHASE3_EXPERIMENT=p3_geo_dyn_temporal_shuffle
exec bash "$script_dir/14_run_phase3_experiment.sh"
