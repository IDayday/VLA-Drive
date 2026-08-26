#!/usr/bin/env bash

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/_common.sh"

wote_root="${CF_GATE_WOTE_ROOT:-$cf_gate_project_root/../third_party/WoTE}"
release_root="${CF_GATE_WOTE_RELEASE_ROOT:-$cf_gate_project_root/../third_party/WoTE_release/wote}"
data_root="${CF_GATE_NAVSIM_DATA_ROOT:-}"
metric_cache_root="${CF_GATE_NAVTRAIN_METRIC_CACHE_ROOT:-}"
output_root="${CF_GATE_EXPERIMENT_ROOT:-$cf_gate_project_root/experiments/cf_effect_gate_wote}"
report_dir="${CF_GATE_REPORT_DIR:-$cf_gate_project_root/reports/cf_effect_gate_wote}"
gate01_run_id="gate0-smoke"
run_id="gate2-main"
config="$cf_gate_project_root/research/cf_effect_gate_wote/configs/gate_main.yaml"
device="${CF_GATE_DEVICE:-cuda}"
dry_run=0
preflight_only=0

usage() {
  cat <<'EOF'
Usage: run_gate2_replay_effect.sh [options]

  --wote-root PATH
  --release-root PATH
  --data-root PATH
  --metric-cache-root PATH
  --output-root PATH
  --report-dir PATH
  --gate01-run-id ID          Run containing passing G0/G1 and G1 test cache.
  --run-id ID
  --config PATH
  --device DEVICE
  --preflight-only
  --dry-run
  -h, --help

The launcher refuses to run unless G1 passed. It caches the frozen train/val
representations, builds immutable logged-replay effects for train/val/test,
trains the four exactly matched probes for every configured seed, and evaluates
all 256 candidates on the fixed test split. Existing run/report outputs are
never overwritten.
EOF
}

while (($#)); do
  case "$1" in
    --wote-root|--release-root|--data-root|--metric-cache-root|--output-root|--report-dir|--gate01-run-id|--run-id|--config|--device)
      [[ $# -ge 2 ]] || { printf 'missing value for %s\n' "$1" >&2; exit 2; }
      case "$1" in
        --wote-root) wote_root="$2" ;;
        --release-root) release_root="$2" ;;
        --data-root) data_root="$2" ;;
        --metric-cache-root) metric_cache_root="$2" ;;
        --output-root) output_root="$2" ;;
        --report-dir) report_dir="$2" ;;
        --gate01-run-id) gate01_run_id="$2" ;;
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

[[ -n "$data_root" ]] || { printf -- '--data-root is required\n' >&2; exit 2; }
[[ -n "$metric_cache_root" ]] || { printf -- '--metric-cache-root is required\n' >&2; exit 2; }
[[ "$run_id" =~ ^[A-Za-z0-9._-]+$ ]] || { printf 'unsafe run id: %s\n' "$run_id" >&2; exit 2; }

split_dir="$cf_gate_project_root/research/cf_effect_gate_wote/configs/splits"
train_tokens="$split_dir/train_tokens.txt"
val_tokens="$split_dir/val_tokens.txt"
test_tokens="$split_dir/test_tokens.txt"
gate01_root="$output_root/$gate01_run_id"
g1_summary="$gate01_root/g1_candidate_oracle_summary.json"
test_cache="$gate01_root/g1-test-cache"
run_root="$output_root/$run_id"
train_cache="$run_root/frozen-train"
val_cache="$run_root/frozen-val"
train_effects="$run_root/effects-train"
val_effects="$run_root/effects-val"
test_effects="$run_root/effects-test"
training_root="$run_root/probe-training"
evaluation_root="$run_root/evaluation"

cache_train=(
  -m research.cf_effect_gate_wote.src.cache_wote_features cache
  --wote-root "$wote_root" --release-root "$release_root" --data-root "$data_root"
  --tokens "$train_tokens" --output "$train_cache" --run-id "$run_id-train"
  --split train --device "$device" --shard-scenes 64
)
cache_val=(
  -m research.cf_effect_gate_wote.src.cache_wote_features cache
  --wote-root "$wote_root" --release-root "$release_root" --data-root "$data_root"
  --tokens "$val_tokens" --output "$val_cache" --run-id "$run_id-val"
  --split val --device "$device" --shard-scenes 64
)
effect_base=(
  -m research.cf_effect_gate_wote.src.train_probe cache-effects
  --wote-root "$wote_root" --data-root "$data_root" --metric-cache "$metric_cache_root"
  --actor-slots 16 --interval-seconds 0.5 --ego-length-m 4.87 --ego-width-m 2.27
  --clearance-m 6 --tca-seconds 3 --tca-distance-m 10
  --conflict-zone-clearance-m 1 --sensitivity-clearance-m 4 6 8
)
train_command=(
  -m research.cf_effect_gate_wote.src.train_probe train-suite
  --config "$config" --train-cache "$train_cache" --val-cache "$val_cache"
  --train-effects "$train_effects" --val-effects "$val_effects"
  --output "$training_root" --device "$device"
  --models trajectory_only direct_current shared_logged_future oracle_replay_effect \
    wote_full_future wote_environment_only
)
evaluate_command=(
  -m research.cf_effect_gate_wote.src.evaluate_probe
  --config "$config" --training-root "$training_root" --test-cache "$test_cache"
  --test-effects "$test_effects" --output-dir "$evaluation_root" --device "$device"
)

printf '[gate2] project_root=%s\n' "$cf_gate_project_root"
printf '[gate2] run_root=%s\n' "$run_root"
printf '[gate2] device=%s candidates=256\n' "$device"

if ((dry_run)); then
  cf_gate_print_command python "${cache_train[@]}" --dry-run
  cf_gate_print_command python "${cache_val[@]}" --dry-run
  cf_gate_print_command python "${effect_base[@]}" --frozen-cache "$train_cache" --output "$train_effects"
  cf_gate_print_command python "${effect_base[@]}" --frozen-cache "$val_cache" --output "$val_effects"
  cf_gate_print_command python "${effect_base[@]}" --frozen-cache "$test_cache" --output "$test_effects"
  cf_gate_print_command python "${train_command[@]}"
  cf_gate_print_command python "${evaluate_command[@]}"
  exit 0
fi

cf_gate_require_file "$g1_summary" 'G1 summary' || exit 3
cf_gate_require_file "$config" 'Gate config' || exit 3
cf_gate_require_file "$train_tokens" 'fixed train split' || exit 3
cf_gate_require_file "$val_tokens" 'fixed val split' || exit 3
cf_gate_require_file "$test_tokens" 'fixed test split' || exit 3
cf_gate_require_dir "$test_cache" 'G1 frozen test cache' || exit 3
cf_gate_require_dir "$metric_cache_root/metadata" 'navtrain metric cache metadata' || exit 3
cf_gate_run_python -c \
  'import json,sys; assert json.load(open(sys.argv[1])).get("gate_g1_pass") is True, "G1 is not PASS"' \
  "$g1_summary"

if ((preflight_only)); then
  cf_gate_run_python "${cache_train[@]}" --preflight-only
  cf_gate_run_python "${cache_val[@]}" --preflight-only
  cf_gate_run_python -c \
    'from pathlib import Path; from research.cf_effect_gate_wote.src.feature_store import FeatureShardReader; [FeatureShardReader(Path(p)) for p in __import__("sys").argv[1:]]' \
    "$test_cache"
  printf '[gate2] preflight passed; no outputs were created\n'
  exit 0
fi

if [[ -e "$run_root" ]]; then
  printf '[gate2] refusing existing run root: %s\n' "$run_root" >&2
  exit 3
fi
mkdir -p "$run_root"

cf_gate_run_python "${cache_train[@]}"
cf_gate_run_python "${cache_val[@]}"
cf_gate_run_python "${effect_base[@]}" --frozen-cache "$train_cache" --output "$train_effects"
cf_gate_run_python "${effect_base[@]}" --frozen-cache "$val_cache" --output "$val_effects"
cf_gate_run_python "${effect_base[@]}" --frozen-cache "$test_cache" --output "$test_effects"
cf_gate_run_python "${train_command[@]}"
cf_gate_run_python "${evaluate_command[@]}"

mkdir -p "$report_dir"
for report_name in probe_metrics.csv scene_level_g2.parquet; do
  staged_name="$report_name"
  if [[ "$report_name" == "probe_metrics.csv" ]]; then
    staged_name="probe_metrics_g2.csv"
  fi
  if [[ -e "$report_dir/$staged_name" ]]; then
    printf '[gate2] refusing existing report: %s\n' "$report_dir/$staged_name" >&2
    exit 3
  fi
  cp "$evaluation_root/$report_name" "$report_dir/$staged_name"
done

gate_pass="$(cf_gate_run_python -c \
  'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["gate_g2_pass"])).lower())' \
  "$evaluation_root/g2_summary.json")"
if [[ "$gate_pass" != "true" ]]; then
  printf '[gate2] FAIL: replay-grounded candidate effect adds no planning information\n' >&2
  exit 4
fi
printf '[gate2] PASS\n'
