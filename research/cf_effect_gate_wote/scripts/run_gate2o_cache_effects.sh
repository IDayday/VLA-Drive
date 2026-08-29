#!/usr/bin/env bash

set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/_oracle_effect_common.sh"
gate2o_parse_args "$@"
gate2o_require_common

effect_one() {
  local frozen="$1" output="$2"
  if [[ ! -f "$output/manifest.json" ]]; then
    gate2o_run_python -m research.cf_effect_gate_wote.src.train_probe cache-effects \
      --frozen-cache "$frozen" --output "$output" --wote-root "$gate2o_wote_root" \
      --data-root "$gate2o_data_root" --metric-cache "$gate2o_metric_cache" \
      --actor-slots 16 --interval-seconds 0.5 \
      --workers "${GATE2O_EFFECT_WORKERS:-16}"
  fi
}

effect_one "$gate2o_output_root/features-determinism16" "$gate2o_output_root/effects-determinism-run1"
effect_one "$gate2o_output_root/features-determinism16" "$gate2o_output_root/effects-determinism-run2"
if [[ ! -f "$gate2o_audit_root/effect_cache_determinism_audit.json" ]]; then
  gate2o_run_python -m research.cf_effect_gate_wote.src.oracle_effect_data compare-determinism \
    --first "$gate2o_output_root/effects-determinism-run1" \
    --second "$gate2o_output_root/effects-determinism-run2" --kind effects \
    --output "$gate2o_audit_root/effect_cache_determinism_audit.json"
fi
effect_one "$gate2o_output_root/features-train" "$gate2o_output_root/effects-train"
effect_one "$gate2o_output_root/features-val" "$gate2o_output_root/effects-val"
effect_one "$gate2o_output_root/features-test" "$gate2o_output_root/effects-test"
printf '%s\n' '[gate2o] replay-grounded primitive and diagnostic effect caches complete'
