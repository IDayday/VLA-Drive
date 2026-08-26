#!/usr/bin/env bash

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/_common.sh"

output_root="${CF_GATE_EXPERIMENT_ROOT:-$cf_gate_project_root/experiments/cf_effect_gate_wote}"
report_dir="${CF_GATE_REPORT_DIR:-$cf_gate_project_root/reports/cf_effect_gate_wote}"
gate01_run_id="gate0-smoke"
gate2_run_id="gate2-main"
run_id="gate3-main"
config="$cf_gate_project_root/research/cf_effect_gate_wote/configs/gate_main.yaml"
device="${CF_GATE_DEVICE:-cuda}"
dry_run=0
preflight_only=0

usage() {
  cat <<'EOF'
Usage: run_gate3_effect_prediction.sh [options]

  --output-root PATH
  --report-dir PATH
  --gate01-run-id ID
  --gate2-run-id ID
  --run-id ID
  --config PATH
  --device DEVICE
  --preflight-only
  --dry-run
  -h, --help

G3 is permitted only after G2 PASS. The forward model reads frozen current BEV,
current ego status, and candidate trajectory; its checkpoint records this input
schema. It predicts structured caches for every split/seed, then evaluates a
matched PredictedEffectScorer alongside Direct, Oracle, WoTE full-future, and
WoTE environment-only controls.
EOF
}

while (($#)); do
  case "$1" in
    --output-root|--report-dir|--gate01-run-id|--gate2-run-id|--run-id|--config|--device)
      [[ $# -ge 2 ]] || { printf 'missing value for %s\n' "$1" >&2; exit 2; }
      case "$1" in
        --output-root) output_root="$2" ;;
        --report-dir) report_dir="$2" ;;
        --gate01-run-id) gate01_run_id="$2" ;;
        --gate2-run-id) gate2_run_id="$2" ;;
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
run_root="$output_root/$run_id"
g2_summary="$gate2_root/evaluation/g2_summary.json"
train_cache="$gate2_root/frozen-train"
val_cache="$gate2_root/frozen-val"
test_cache="$gate01_root/g1-test-cache"
train_effects="$gate2_root/effects-train"
val_effects="$gate2_root/effects-val"
test_effects="$gate2_root/effects-test"
g2_training="$gate2_root/probe-training"
predictor_root="$run_root/predictor"
predicted_effect_root="$run_root/predicted-effects"
scorer_root="$run_root/scorers"
evaluation_root="$run_root/evaluation"

train_predictor=(
  -m research.cf_effect_gate_wote.src.effect_prediction train-predictor
  --config "$config" --train-cache "$train_cache" --val-cache "$val_cache"
  --train-effects "$train_effects" --val-effects "$val_effects"
  --output "$predictor_root" --device "$device"
)
predict_caches=(
  -m research.cf_effect_gate_wote.src.effect_prediction predict-caches
  --config "$config" --predictor-root "$predictor_root"
  --train-cache "$train_cache" --train-effects "$train_effects"
  --val-cache "$val_cache" --val-effects "$val_effects"
  --test-cache "$test_cache" --test-effects "$test_effects"
  --output "$predicted_effect_root" --device "$device"
)
train_scorers=(
  -m research.cf_effect_gate_wote.src.effect_prediction train-scorers
  --config "$config" --g2-training-root "$g2_training"
  --predicted-effect-root "$predicted_effect_root"
  --train-cache "$train_cache" --val-cache "$val_cache"
  --train-effects "$train_effects" --val-effects "$val_effects"
  --output "$scorer_root" --device "$device"
)
evaluate=(
  -m research.cf_effect_gate_wote.src.effect_prediction evaluate-g3
  --config "$config" --training-root "$scorer_root" --predictor-root "$predictor_root"
  --predicted-effect-root "$predicted_effect_root"
  --test-cache "$test_cache" --test-effects "$test_effects"
  --output "$evaluation_root" --device "$device"
)

printf '[gate3] project_root=%s\n' "$cf_gate_project_root"
printf '[gate3] run_root=%s\n' "$run_root"
printf '[gate3] device=%s predictor_limit=10000000\n' "$device"

if ((dry_run)); then
  cf_gate_print_command python "${train_predictor[@]}"
  cf_gate_print_command python "${predict_caches[@]}"
  cf_gate_print_command python "${train_scorers[@]}"
  cf_gate_print_command python "${evaluate[@]}"
  exit 0
fi

cf_gate_require_file "$config" 'Gate config' || exit 3
cf_gate_require_file "$g2_summary" 'G2 summary' || exit 3
for cache in "$train_cache" "$val_cache" "$test_cache" "$train_effects" "$val_effects" "$test_effects"; do
  cf_gate_require_file "$cache/manifest.json" "finalized cache manifest" || exit 3
done
cf_gate_require_file "$g2_training/training_manifest.json" 'G2 training manifest' || exit 3
cf_gate_run_python -c \
  'import json,sys; assert json.load(open(sys.argv[1])).get("gate_g2_pass") is True, "G2 is not PASS"' \
  "$g2_summary"

if ((preflight_only)); then
  cf_gate_run_python -c \
    'from pathlib import Path; from research.cf_effect_gate_wote.src.feature_store import FeatureShardReader; [FeatureShardReader(Path(p)) for p in __import__("sys").argv[1:]]' \
    "$train_cache" "$val_cache" "$test_cache" "$train_effects" "$val_effects" "$test_effects"
  printf '[gate3] preflight passed; no outputs were created\n'
  exit 0
fi

if [[ -e "$run_root" ]]; then
  printf '[gate3] refusing existing run root: %s\n' "$run_root" >&2
  exit 3
fi

cf_gate_run_python "${train_predictor[@]}"
cf_gate_run_python "${predict_caches[@]}"
cf_gate_run_python "${train_scorers[@]}"
cf_gate_run_python "${evaluate[@]}"

mkdir -p "$report_dir"
for report_name in effect_prediction_metrics.csv probe_metrics_g3.csv scene_level_g3.parquet; do
  if [[ -e "$report_dir/$report_name" ]]; then
    printf '[gate3] refusing existing report: %s\n' "$report_dir/$report_name" >&2
    exit 3
  fi
  cp "$evaluation_root/$report_name" "$report_dir/$report_name"
done

gate_pass="$(cf_gate_run_python -c \
  'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["gate_g3_pass"])).lower())' \
  "$evaluation_root/g3_summary.json")"
if [[ "$gate_pass" != "true" ]]; then
  printf '[gate3] FAIL: EFFECT_TARGET_VALID_BUT_PREDICTION_BOTTLENECK\n' >&2
  exit 4
fi
printf '[gate3] PASS\n'
