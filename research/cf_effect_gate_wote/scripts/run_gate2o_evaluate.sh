#!/usr/bin/env bash

set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/_oracle_effect_common.sh"
gate2o_parse_args "$@"
gate2o_require_common

if [[ ! -f "$gate2o_output_root/evaluation/evaluation_manifest.json" ]]; then
  gate2o_run_python -m research.cf_effect_gate_wote.src.oracle_effect_evaluation \
    evaluate-suite --training "$gate2o_output_root/training" \
    --test-cache "$gate2o_output_root/features-test" \
    --test-effects "$gate2o_output_root/effects-test" \
    --test-labels "$gate2o_output_root/labels-test" --device "$gate2o_device" \
    --output "$gate2o_output_root/evaluation"
fi
gate2o_run_python -m research.cf_effect_gate_wote.src.oracle_effect_report \
  --repo-root "$cf_gate_project_root" --evaluation "$gate2o_output_root/evaluation" \
  --hyperparameters "$gate2o_audit_root/GLOBAL_HYPERPARAMETER_SELECTION.json" \
  --split-manifest "$gate2o_audit_root/split_manifest.json" \
  --relabel-audit "$gate2o_audit_root/relabel_determinism_audit.json" \
  --effect-audit "$gate2o_audit_root/effect_cache_determinism_audit.json" \
  --asset-manifest "$gate2o_audit_root/ASSET_MANIFEST.json" \
  --train-cache "$gate2o_output_root/features-train" \
  --val-cache "$gate2o_output_root/features-val" \
  --test-cache "$gate2o_output_root/features-test" \
  --train-effects "$gate2o_output_root/effects-train" \
  --val-effects "$gate2o_output_root/effects-val" \
  --test-effects "$gate2o_output_root/effects-test" \
  --train-labels "$gate2o_output_root/labels-train" \
  --val-labels "$gate2o_output_root/labels-val" \
  --test-labels "$gate2o_output_root/labels-test" --output "$gate2o_report_dir"
printf '%s\n' '[gate2o] Oracle Effect Gate report complete; forward/inverse/refinement remain NOT_RUN'
