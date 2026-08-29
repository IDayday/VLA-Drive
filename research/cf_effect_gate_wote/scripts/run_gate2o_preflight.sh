#!/usr/bin/env bash

set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/_oracle_effect_common.sh"
gate2o_parse_args "$@"
gate2o_require_common
mkdir -p "$gate2o_audit_root"

gate2o_run_python -m research.cf_effect_gate_wote.src.oracle_effect_data build-splits \
  --split-dir "$gate2o_split_dir" --output-dir "$gate2o_split_dir" \
  --manifest "$gate2o_audit_root/split_manifest.json"
gate2o_run_python -m research.cf_effect_gate_wote.src.oracle_effect_data preflight \
  --repo-root "$cf_gate_project_root" --wote-root "$gate2o_wote_root" \
  --checkpoint "$gate2o_checkpoint" --candidate-bank "$gate2o_anchors" \
  --evaluator-contract "$gate2o_contract" --data-root "$gate2o_data_root" \
  --map-root "$gate2o_map_root" --output "$gate2o_audit_root/ASSET_MANIFEST.json"
gate2o_run pytest "$cf_gate_project_root/research/cf_effect_gate_wote/tests" -q
printf '%s\n' '[gate2o] preflight PASS'
