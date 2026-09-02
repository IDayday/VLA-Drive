#!/usr/bin/env bash
set -Eeuo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"

stage="${1:-all}"
source_run="${ORACLE_SOURCE_RUN:-/mnt/workspace/project/VLA-Drive-cf-effect-oracle/experiments/cf_effect_wote_oracle_effect/oracle-effect-v2-20260829}"
direct_root="${DIRECT_CHECKPOINT_ROOT:-${repo_root}/experiments/cf_effect_wote_direct_rehab/direct-rehab-v1-20260901/confirmation-v1}"
output_root="${MATCHED_ORACLE_OUTPUT_ROOT:-${repo_root}/experiments/cf_effect_wote_oracle_effect_rehab/gate2o-matched-hybrid-v3-20260902}"
report_root="${MATCHED_ORACLE_REPORT_ROOT:-${repo_root}/reports/cf_effect_wote_oracle_effect_rehab}"
config="${repo_root}/research/cf_effect_gate_wote/configs/oracle_effect_rehab_v3.yaml"

common=(
  --train-cache "${source_run}/features-train"
  --val-cache "${source_run}/features-val"
  --train-effects "${source_run}/effects-train"
  --val-effects "${source_run}/effects-val"
  --train-labels "${source_run}/labels-train"
  --val-labels "${source_run}/labels-val"
  --probe-backbone matched_hybrid_v3
  --direct-checkpoint-root "${direct_root}"
  --device cuda
)

require_assets() {
  local path
  for path in \
    "${source_run}/features-train/manifest.json" \
    "${source_run}/features-val/manifest.json" \
    "${source_run}/features-test/manifest.json" \
    "${source_run}/effects-train/manifest.json" \
    "${source_run}/effects-val/manifest.json" \
    "${source_run}/effects-test/manifest.json" \
    "${source_run}/labels-train/manifest.json" \
    "${source_run}/labels-val/manifest.json" \
    "${source_run}/labels-test/manifest.json" \
    "${direct_root}/hybrid_current-seed0.pt" \
    "${direct_root}/hybrid_current-seed1.pt" \
    "${direct_root}/hybrid_current-seed2.pt"; do
    [[ -f "${path}" ]] || { echo "missing required asset: ${path}" >&2; return 2; }
  done
}

run_smoke() {
  local destination="${output_root}/overfit-smoke.json"
  [[ ! -e "${destination}" ]] || { echo "refusing existing smoke: ${destination}" >&2; return 2; }
  mkdir -p "${output_root}"
  python -m research.cf_effect_gate_wote.src.oracle_effect_evaluation \
    overfit-smoke \
    --train-cache "${source_run}/features-train" \
    --train-effects "${source_run}/effects-train" \
    --train-labels "${source_run}/labels-train" \
    --probe-backbone matched_hybrid_v3 \
    --direct-checkpoint-root "${direct_root}" \
    --device cuda --steps 30 --output "${destination}"
}

run_pilot() {
  local destination="${output_root}/GLOBAL_HYPERPARAMETER_SELECTION.json"
  [[ ! -e "${destination}" ]] || { echo "refusing existing pilot: ${destination}" >&2; return 2; }
  mkdir -p "${output_root}"
  python -m research.cf_effect_gate_wote.src.oracle_effect_evaluation \
    pilot "${common[@]}" --config "${config}" --output "${destination}"
}

run_train() {
  [[ -f "${output_root}/GLOBAL_HYPERPARAMETER_SELECTION.json" ]] || {
    echo "pilot result missing" >&2
    return 2
  }
  if [[ -f "${output_root}/training/training_manifest.json" ]]; then
    echo "matched Oracle training already complete"
    return 0
  fi
  # train-main validates every existing checkpoint against model/seed/global
  # hyperparameters and resumes only the missing trials.  It never overwrites.
  python -m research.cf_effect_gate_wote.src.oracle_effect_evaluation \
    train-main "${common[@]}" --config "${config}" \
    --hyperparameters "${output_root}/GLOBAL_HYPERPARAMETER_SELECTION.json" \
    --output "${output_root}/training"
}

run_evaluate() {
  [[ -f "${output_root}/training/training_manifest.json" ]] || {
    echo "training manifest missing" >&2
    return 2
  }
  [[ ! -e "${output_root}/evaluation" ]] || {
    echo "refusing existing evaluation output" >&2
    return 2
  }
  python -m research.cf_effect_gate_wote.src.oracle_effect_evaluation \
    evaluate-suite --training "${output_root}/training" \
    --test-cache "${source_run}/features-test" \
    --test-effects "${source_run}/effects-test" \
    --test-labels "${source_run}/labels-test" \
    --device cuda --output "${output_root}/evaluation"
}

run_report() {
  local old_report="${repo_root}/reports/cf_effect_wote_oracle_effect"
  [[ -d "${output_root}/evaluation" ]] || {
    echo "evaluation output missing" >&2
    return 2
  }
  [[ ! -e "${report_root}" ]] || {
    echo "refusing existing matched report output" >&2
    return 2
  }
  python -m research.cf_effect_gate_wote.src.oracle_effect_report \
    --repo-root "${repo_root}" \
    --evaluation "${output_root}/evaluation" \
    --hyperparameters "${output_root}/GLOBAL_HYPERPARAMETER_SELECTION.json" \
    --split-manifest "${old_report}/split_manifest.json" \
    --relabel-audit "${old_report}/relabel_determinism_audit.json" \
    --effect-audit "${old_report}/effect_cache_determinism_audit.json" \
    --asset-manifest "${old_report}/ASSET_MANIFEST.json" \
    --train-cache "${source_run}/features-train" \
    --train-effects "${source_run}/effects-train" \
    --train-labels "${source_run}/labels-train" \
    --val-cache "${source_run}/features-val" \
    --val-effects "${source_run}/effects-val" \
    --val-labels "${source_run}/labels-val" \
    --test-cache "${source_run}/features-test" \
    --test-effects "${source_run}/effects-test" \
    --test-labels "${source_run}/labels-test" \
    --probe-backbone matched_hybrid_v3 \
    --output "${report_root}"
}

require_assets
case "${stage}" in
  preflight) echo "matched Oracle assets: PASS" ;;
  smoke) run_smoke ;;
  pilot) run_pilot ;;
  train) run_train ;;
  evaluate) run_evaluate ;;
  report) run_report ;;
  all)
    run_smoke
    run_pilot
    run_train
    run_evaluate
    run_report
    ;;
  *) echo "usage: $0 {preflight|smoke|pilot|train|evaluate|report|all}" >&2; exit 2 ;;
esac
