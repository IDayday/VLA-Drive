#!/usr/bin/env bash

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/_common.sh"

WOTE_DEFAULT="$cf_gate_project_root/../third_party/WoTE"
wote_root="${CF_GATE_WOTE_ROOT:-$WOTE_DEFAULT}"
release_root="${CF_GATE_WOTE_RELEASE_ROOT:-$cf_gate_project_root/../third_party/WoTE_release/wote}"
data_root="${CF_GATE_NAVSIM_DATA_ROOT:-}"
output_root="${CF_GATE_EXPERIMENT_ROOT:-$cf_gate_project_root/experiments/cf_effect_gate_wote}"
report_dir="${CF_GATE_REPORT_DIR:-$cf_gate_project_root/reports/cf_effect_gate_wote}"
run_id="gate0-smoke"
device="${CF_GATE_DEVICE:-cuda}"
dry_run=0
preflight_only=0

usage() {
  cat <<'EOF'
Usage: run_gate1_candidate_oracle.sh [options]

  --wote-root PATH
  --release-root PATH
  --data-root PATH
  --output-root PATH
  --report-dir PATH
  --gate0-run-id ID
  --device DEVICE
  --preflight-only
  --dry-run
  -h, --help

G1 consumes only a passing G0 cache and the fixed test split. It does not alter
the candidate set or select candidates using ground-truth scores.
EOF
}

while (($#)); do
  case "$1" in
    --wote-root|--release-root|--data-root|--output-root|--report-dir|--gate0-run-id|--device)
      [[ $# -ge 2 ]] || { printf 'missing value for %s\n' "$1" >&2; exit 2; }
      case "$1" in
        --wote-root) wote_root="$2" ;;
        --release-root) release_root="$2" ;;
        --data-root) data_root="$2" ;;
        --output-root) output_root="$2" ;;
        --report-dir) report_dir="$2" ;;
        --gate0-run-id) run_id="$2" ;;
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

run_root="$output_root/$run_id"
cache_root="$run_root/g1-test-cache"
g0_summary="$run_root/g0_smoke_summary.json"
selected_json="$run_root/g1_selected_indices.json"
score_path="$release_root/extra_data/planning_vb/formatted_pdm_score_256.npy"
test_tokens="$cf_gate_project_root/research/cf_effect_gate_wote/configs/splits/test_tokens.txt"
oracle_csv="$run_root/candidate_oracle_summary.csv"
oracle_summary="$run_root/g1_candidate_oracle_summary.json"

[[ -n "$data_root" ]] || { printf -- '--data-root is required\n' >&2; exit 2; }

cache_command=(
  -m research.cf_effect_gate_wote.src.cache_wote_features cache
  --wote-root "$wote_root"
  --release-root "$release_root"
  --data-root "$data_root"
  --tokens "$test_tokens"
  --output "$cache_root"
  --run-id "$run_id-g1-test"
  --split test
  --device "$device"
  --shard-scenes 64
)

if ((dry_run)); then
  cf_gate_print_command python "${cache_command[@]}" --dry-run
  cf_gate_print_command python -m research.cf_effect_gate_wote.src.candidate_alignment selected-from-cache \
    --cache-root "$cache_root" --output "$selected_json"
  cf_gate_print_command python -m research.cf_effect_gate_wote.src.candidate_alignment oracle \
    --score-path "$score_path" --selected-json "$selected_json" --tokens "$test_tokens" \
    --output-csv "$oracle_csv" --output-summary "$oracle_summary"
  exit 0
fi

cf_gate_require_file "$g0_summary" 'G0 summary' || exit 3
cf_gate_require_file "$score_path" 'candidate score table' || exit 3
cf_gate_require_file "$test_tokens" 'fixed test token split' || exit 3
cf_gate_run_python -c \
  'import json,sys; p=json.load(open(sys.argv[1])); assert p.get("gate_g0_pass") is True, "G0 is not PASS"' \
  "$g0_summary"

if ((preflight_only)); then
  cf_gate_run_python "${cache_command[@]}" --preflight-only
  printf '[gate1] preflight passed; no outputs were created\n'
  exit 0
fi

cf_gate_run_python "${cache_command[@]}"
cf_gate_run_python -m research.cf_effect_gate_wote.src.candidate_alignment selected-from-cache \
  --cache-root "$cache_root" --output "$selected_json"
cf_gate_run_python -m research.cf_effect_gate_wote.src.candidate_alignment oracle \
  --score-path "$score_path" \
  --selected-json "$selected_json" \
  --tokens "$test_tokens" \
  --output-csv "$oracle_csv" \
  --output-summary "$oracle_summary"

mkdir -p "$report_dir"
if [[ -e "$report_dir/candidate_oracle_summary.csv" ]]; then
  printf '[gate1] refusing existing report: %s\n' "$report_dir/candidate_oracle_summary.csv" >&2
  exit 3
fi
cp "$oracle_csv" "$report_dir/candidate_oracle_summary.csv"

gate_pass="$(cf_gate_run_python -c \
  'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["gate_g1_pass"])).lower())' \
  "$oracle_summary")"
if [[ "$gate_pass" != "true" ]]; then
  printf '[gate1] FAIL: candidate coverage insufficient\n' >&2
  exit 4
fi
printf '[gate1] PASS\n'
