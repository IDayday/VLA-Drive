#!/usr/bin/env bash

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/_common.sh"

output_root="${CF_GATE_EXPERIMENT_ROOT:-$cf_gate_project_root/experiments/cf_effect_gate_wote}"
report_dir="${CF_GATE_REPORT_DIR:-$cf_gate_project_root/reports/cf_effect_gate_wote}"
gate01_run_id="gate0-smoke"
gate2_run_id="gate2-main"
gate3_run_id="gate3-main"
run_id="gate4-main"
config="$cf_gate_project_root/research/cf_effect_gate_wote/configs/gate_main.yaml"
device="${CF_GATE_DEVICE:-cuda}"
dry_run=0
preflight_only=0

usage() {
  cat <<'EOF'
Usage: run_gate4_inverse.sh [options]

  --output-root PATH
  --report-dir PATH
  --gate01-run-id ID
  --gate2-run-id ID
  --gate3-run-id ID
  --run-id ID
  --config PATH
  --device DEVICE
  --preflight-only
  --dry-run
  -h, --help

G4 is permitted only after G3 PASS. It trains ego-only, environment-only, and
full-effect inverse probes on score-free geometry-FPS candidate sets; evaluates
retrieval/delta shuffles; freezes the probes; tunes additive/rejection gates on
validation only; and evaluates the fixed choices once on test.
EOF
}

while (($#)); do
  case "$1" in
    --output-root|--report-dir|--gate01-run-id|--gate2-run-id|--gate3-run-id|--run-id|--config|--device)
      [[ $# -ge 2 ]] || { printf 'missing value for %s\n' "$1" >&2; exit 2; }
      case "$1" in
        --output-root) output_root="$2" ;;
        --report-dir) report_dir="$2" ;;
        --gate01-run-id) gate01_run_id="$2" ;;
        --gate2-run-id) gate2_run_id="$2" ;;
        --gate3-run-id) gate3_run_id="$2" ;;
        --run-id) run_id="$2" ;;
        --config) config="$2" ;;
        --device) device="$2" ;;
      esac
      shift 2
      ;;
    --preflight-only) preflight_only=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

[[ "$run_id" =~ ^[A-Za-z0-9._-]+$ ]] || { printf 'unsafe run id: %s\n' "$run_id" >&2; exit 2; }

gate01_root="$output_root/$gate01_run_id"
gate2_root="$output_root/$gate2_run_id"
gate3_root="$output_root/$gate3_run_id"
run_root="$output_root/$run_id"
g3_summary="$gate3_root/evaluation/g3_summary.json"
train_cache="$gate2_root/frozen-train"
val_cache="$gate2_root/frozen-val"
test_cache="$gate01_root/g1-test-cache"
train_effects="$gate2_root/effects-train"
val_effects="$gate2_root/effects-val"
test_effects="$gate2_root/effects-test"
predicted_effects="$gate3_root/predicted-effects"
g3_scorers="$gate3_root/scorers"
training_root="$run_root/training"
identifiability_root="$run_root/identifiability"
planning_root="$run_root/planning"
evaluation_root="$run_root/evaluation"

train_inverse=(
  -m research.cf_effect_gate_wote.src.inverse_experiments train-inverse
  --config "$config" --train-cache "$train_cache" --val-cache "$val_cache"
  --train-effects "$train_effects" --val-effects "$val_effects"
  --output "$training_root" --device "$device"
)
evaluate_identifiability=(
  -m research.cf_effect_gate_wote.src.inverse_experiments evaluate-identifiability
  --config "$config" --training-root "$training_root"
  --test-cache "$test_cache" --test-effects "$test_effects"
  --output "$identifiability_root" --device "$device"
)
planning_gate=(
  -m research.cf_effect_gate_wote.src.inverse_experiments planning-gate
  --config "$config" --inverse-training-root "$training_root"
  --g3-scorer-root "$g3_scorers" --predicted-effect-root "$predicted_effects"
  --val-cache "$val_cache" --test-cache "$test_cache"
  --output "$planning_root" --device "$device"
)
finalize=(
  -m research.cf_effect_gate_wote.src.inverse_experiments finalize-g4
  --config "$config" --identifiability-root "$identifiability_root"
  --planning-root "$planning_root" --output "$evaluation_root"
)

printf '[gate4] project_root=%s\n' "$cf_gate_project_root"
printf '[gate4] run_root=%s\n' "$run_root"
printf '[gate4] device=%s retrieval_K=16\n' "$device"

if ((dry_run)); then
  cf_gate_print_command python "${train_inverse[@]}"
  cf_gate_print_command python "${evaluate_identifiability[@]}"
  cf_gate_print_command python "${planning_gate[@]}"
  cf_gate_print_command python "${finalize[@]}"
  exit 0
fi

cf_gate_require_file "$config" 'Gate config' || exit 3
cf_gate_require_file "$g3_summary" 'G3 summary' || exit 3
for cache in "$train_cache" "$val_cache" "$test_cache" "$train_effects" "$val_effects" "$test_effects"; do
  cf_gate_require_file "$cache/manifest.json" 'finalized cache manifest' || exit 3
done
cf_gate_require_file "$g3_scorers/training_manifest.json" 'G3 scorer manifest' || exit 3
cf_gate_require_file "$predicted_effects/effect_prediction_metrics.csv" 'predicted effect manifest' || exit 3
cf_gate_run_python -c \
  'import json,sys; assert json.load(open(sys.argv[1])).get("gate_g3_pass") is True, "G3 is not PASS"' \
  "$g3_summary"

if ((preflight_only)); then
  cf_gate_run_python -c \
    'from pathlib import Path; from research.cf_effect_gate_wote.src.feature_store import FeatureShardReader; [FeatureShardReader(Path(p)) for p in __import__("sys").argv[1:]]' \
    "$train_cache" "$val_cache" "$test_cache" "$train_effects" "$val_effects" "$test_effects"
  printf '[gate4] preflight passed; no outputs were created\n'
  exit 0
fi

if [[ -e "$run_root" ]]; then
  printf '[gate4] refusing existing run root: %s\n' "$run_root" >&2
  exit 3
fi

cf_gate_run_python "${train_inverse[@]}"
cf_gate_run_python "${evaluate_identifiability[@]}"
cf_gate_run_python "${planning_gate[@]}"
cf_gate_run_python "${finalize[@]}"

mkdir -p "$report_dir"
for report_name in inverse_metrics.csv scene_level_g4.parquet; do
  if [[ -e "$report_dir/$report_name" ]]; then
    printf '[gate4] refusing existing report: %s\n' "$report_dir/$report_name" >&2
    exit 3
  fi
  cp "$evaluation_root/$report_name" "$report_dir/$report_name"
done

gate_pass="$(cf_gate_run_python -c \
  'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["gate_g4_pass"])).lower())' \
  "$evaluation_root/g4_summary.json")"
if [[ "$gate_pass" != "true" ]]; then
  printf '[gate4] FAIL: inverse remains diagnostic only\n' >&2
  exit 4
fi
printf '[gate4] PASS\n'
