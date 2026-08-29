#!/usr/bin/env bash

set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/_oracle_effect_common.sh"
gate2o_parse_args "$@"
gate2o_require_common

relabel_one() {
  local tokens="$1" output="$2" count="$3" metric_cache="$4"
  if [[ ! -f "$output/manifest.json" ]]; then
    gate2o_run_python -m research.cf_effect_gate_wote.src.independent_relabel \
      run-six-factor --wote-root "$gate2o_wote_root" \
      --metric-cache-root "$metric_cache" --tokens "$tokens" \
      --anchors "$gate2o_anchors" --evaluator-contract "$gate2o_contract" \
      --output "$output" --expected-scenes "$count" --shard-scenes 16
  fi
}

determinism_metric_cache="$gate2o_output_root/metric-cache-determinism16"
if [[ ! -d "$determinism_metric_cache/metadata" ]]; then
  gate2o_run_python -m research.cf_effect_gate_wote.src.cache_metric_subset \
    --wote-root "$gate2o_wote_root" --data-root "$gate2o_data_root" \
    --map-root "$gate2o_map_root" --tokens "$gate2o_determinism_tokens" \
    --output "$determinism_metric_cache"
fi
relabel_one "$gate2o_determinism_tokens" "$gate2o_output_root/relabel-determinism-run1" 16 "$determinism_metric_cache"
relabel_one "$gate2o_determinism_tokens" "$gate2o_output_root/relabel-determinism-run2" 16 "$determinism_metric_cache"
gate2o_run_python -m research.cf_effect_gate_wote.src.oracle_effect_data compare-determinism \
  --first "$gate2o_output_root/relabel-determinism-run1" \
  --second "$gate2o_output_root/relabel-determinism-run2" --kind labels \
  --output "$gate2o_audit_root/relabel_determinism_audit.json"
if [[ ! -d "$gate2o_metric_cache/metadata" ]]; then
  gate2o_run_python -m research.cf_effect_gate_wote.src.cache_metric_subset \
    --wote-root "$gate2o_wote_root" --data-root "$gate2o_data_root" \
    --map-root "$gate2o_map_root" --tokens "$gate2o_train_tokens" \
    "$gate2o_val_tokens" "$gate2o_test_tokens" --output "$gate2o_metric_cache"
fi
relabel_one "$gate2o_train_tokens" "$gate2o_output_root/labels-train" 1024 "$gate2o_metric_cache"
relabel_one "$gate2o_val_tokens" "$gate2o_output_root/labels-val" 256 "$gate2o_metric_cache"
relabel_one "$gate2o_test_tokens" "$gate2o_output_root/labels-test" 512 "$gate2o_metric_cache"
printf '%s\n' '[gate2o] independent six-factor relabel complete'
