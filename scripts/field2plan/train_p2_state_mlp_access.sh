#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export FIELD2PLAN_EXPERIMENT=p2_state_mlp_access
exec bash "$script_dir/07_run_phase2_experiment.sh"
