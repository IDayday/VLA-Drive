#!/usr/bin/env bash

set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/_oracle_effect_common.sh"
gate2o_parse_args "$@"
gate2o_require_common

common=(--train-cache "$gate2o_output_root/features-train"
  --val-cache "$gate2o_output_root/features-val"
  --train-effects "$gate2o_output_root/effects-train"
  --val-effects "$gate2o_output_root/effects-val"
  --train-labels "$gate2o_output_root/labels-train"
  --val-labels "$gate2o_output_root/labels-val" --device "$gate2o_device")

if [[ ! -f "$gate2o_output_root/overfit-smoke.json" ]]; then
  gate2o_run_python -m research.cf_effect_gate_wote.src.oracle_effect_evaluation \
    overfit-smoke --train-cache "$gate2o_output_root/features-train" \
    --train-effects "$gate2o_output_root/effects-train" \
    --train-labels "$gate2o_output_root/labels-train" --device "$gate2o_device" \
    --steps 30 --output "$gate2o_output_root/overfit-smoke.json"
fi
if [[ ! -f "$gate2o_audit_root/GLOBAL_HYPERPARAMETER_SELECTION.json" ]]; then
  gate2o_run_python -m research.cf_effect_gate_wote.src.oracle_effect_evaluation \
    pilot "${common[@]}" --config "$gate2o_config" \
    --output "$gate2o_audit_root/GLOBAL_HYPERPARAMETER_SELECTION.json"
fi
if [[ ! -f "$gate2o_output_root/training/training_manifest.json" ]]; then
  gate2o_run_python -m research.cf_effect_gate_wote.src.oracle_effect_evaluation \
    train-main "${common[@]}" --config "$gate2o_config" \
    --hyperparameters "$gate2o_audit_root/GLOBAL_HYPERPARAMETER_SELECTION.json" \
    --output "$gate2o_output_root/training"
fi
printf '%s\n' '[gate2o] A-L matched-capacity training complete'
