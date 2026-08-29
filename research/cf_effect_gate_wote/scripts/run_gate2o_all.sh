#!/usr/bin/env bash

set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/_oracle_effect_common.sh"
gate2o_parse_args "$@"
gate2o_require_common
gate2o_export_context

"$script_dir/run_gate2o_preflight.sh"
"$script_dir/run_gate2o_relabel.sh"
"$script_dir/run_gate2o_cache_features.sh"
"$script_dir/run_gate2o_cache_effects.sh"
"$script_dir/run_gate2o_train.sh"
"$script_dir/run_gate2o_evaluate.sh"
printf '%s\n' '[gate2o] STOP: Oracle Effect Gate is the terminal scope of this run.'

