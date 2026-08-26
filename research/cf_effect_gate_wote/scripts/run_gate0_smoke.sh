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
run_id="gate0-smoke"
device="${CF_GATE_DEVICE:-cuda}"
dry_run=0
preflight_only=0

usage() {
  cat <<'EOF'
Usage: run_gate0_smoke.sh [options]

Required paths may be supplied by CLI or matching CF_GATE_* variables.
  --wote-root PATH
  --release-root PATH
  --data-root PATH
  --metric-cache-root PATH   Targeted or full navtrain metric cache.
  --output-root PATH
  --report-dir PATH
  --run-id ID
  --device DEVICE
  --preflight-only           Read-only asset/token/cache validation.
  --dry-run                  Print commands without imports or writes.
  -h, --help

The formal run evaluates the first 200 fixed test tokens twice, compares cache
hashes, verifies patched/unpatched outputs at 1e-6, and recomputes 20x256
candidates to audit 20x10 sampled labels. Existing outputs are refused.
EOF
}

while (($#)); do
  case "$1" in
    --wote-root|--release-root|--data-root|--metric-cache-root|--output-root|--report-dir|--run-id|--device)
      [[ $# -ge 2 ]] || { printf 'missing value for %s\n' "$1" >&2; exit 2; }
      option="$1"
      value="$2"
      case "$option" in
        --wote-root) wote_root="$value" ;;
        --release-root) release_root="$value" ;;
        --data-root) data_root="$value" ;;
        --metric-cache-root) metric_cache_root="$value" ;;
        --output-root) output_root="$value" ;;
        --report-dir) report_dir="$value" ;;
        --run-id) run_id="$value" ;;
        --device) device="$value" ;;
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

test_tokens="$cf_gate_project_root/research/cf_effect_gate_wote/configs/splits/test_tokens.txt"
anchor_path="$release_root/extra_data/planning_vb/trajectory_anchors_256.npy"
score_path="$release_root/extra_data/planning_vb/formatted_pdm_score_256.npy"
run_root="$output_root/$run_id"
cache_first="$run_root/cache-first"
cache_second="$run_root/cache-second"

setup_command=(
  bash "$script_dir/setup_wote_gate.sh"
  --wote-root "$wote_root"
  --release-root "$release_root"
  --data-root "$data_root"
  --report-dir "$report_dir"
)
cache_base=(
  -m research.cf_effect_gate_wote.src.cache_wote_features cache
  --wote-root "$wote_root"
  --release-root "$release_root"
  --data-root "$data_root"
  --tokens "$test_tokens"
  --run-id "$run_id-smoke"
  --split smoke
  --device "$device"
  --shard-scenes 16
  --limit 200
)
alignment_command=(
  -m research.cf_effect_gate_wote.src.candidate_alignment audit
  --wote-root "$wote_root"
  --anchor-path "$anchor_path"
  --score-path "$score_path"
  --metric-cache-root "$metric_cache_root"
  --tokens "$test_tokens"
  --output-csv "$run_root/g0_candidate_alignment.csv"
  --output-summary "$run_root/g0_candidate_alignment_summary.json"
  --proposal-num-poses 40
)

printf '[gate0] project_root=%s\n' "$cf_gate_project_root"
printf '[gate0] run_root=%s\n' "$run_root"
printf '[gate0] device=%s scenes=200 candidates=256\n' "$device"

if ((dry_run)); then
  cf_gate_print_command "${setup_command[@]}" --dry-run
  cf_gate_print_command python "${cache_base[@]}" --output "$cache_first" --dry-run
  cf_gate_print_command python "${cache_base[@]}" --output "$cache_second" --dry-run
  cf_gate_print_command python "${alignment_command[@]}"
  cf_gate_print_command python -m research.cf_effect_gate_wote.src.cache_wote_features summarize-g0 \
    --cache-first "$cache_first" --cache-second "$cache_second" \
    --alignment-summary "$run_root/g0_candidate_alignment_summary.json" \
    --output-dir "$run_root"
  exit 0
fi

cf_gate_require_file "$test_tokens" 'fixed test token split' || exit 3
cf_gate_require_dir "$metric_cache_root/metadata" 'navtrain metric cache metadata' || exit 3

if ((preflight_only)); then
  "${setup_command[@]}" --preflight-only
  cf_gate_run_python "${cache_base[@]}" --output "$cache_first" --preflight-only
  cf_gate_run_python -m research.cf_effect_gate_wote.src.candidate_alignment source-audit \
    --wote-root "$wote_root"
  printf '[gate0] preflight passed; no outputs were created\n'
  exit 0
fi

"${setup_command[@]}" --write-manifest
if [[ -e "$run_root" ]]; then
  printf '[gate0] refusing existing run root: %s\n' "$run_root" >&2
  exit 3
fi
mkdir -p "$run_root"

cf_gate_run_python -m research.cf_effect_gate_wote.src.candidate_alignment source-audit \
  --wote-root "$wote_root" --output "$run_root/g0_source_alignment.json"
cf_gate_run_python "${cache_base[@]}" --output "$cache_first"
cf_gate_run_python "${cache_base[@]}" --output "$cache_second"
alignment_status=0
if cf_gate_run_python "${alignment_command[@]}"; then
  alignment_status=0
else
  alignment_status=$?
  if [[ "$alignment_status" -ne 4 ]]; then
    exit "$alignment_status"
  fi
fi

summary_status=0
if cf_gate_run_python -m research.cf_effect_gate_wote.src.cache_wote_features summarize-g0 \
  --cache-first "$cache_first" \
  --cache-second "$cache_second" \
  --alignment-summary "$run_root/g0_candidate_alignment_summary.json" \
  --output-dir "$run_root"; then
  summary_status=0
else
  summary_status=$?
  cf_gate_require_file "$run_root/g0_smoke_summary.json" 'G0 failure summary' || exit "$summary_status"
fi

if [[ -e "$report_dir/candidate_alignment.csv" ]]; then
  printf '[gate0] refusing existing report: %s\n' "$report_dir/candidate_alignment.csv" >&2
  exit 3
fi
mkdir -p "$report_dir"
cp "$run_root/g0_candidate_alignment.csv" "$report_dir/candidate_alignment.csv"
if [[ "$alignment_status" -eq 4 || "$summary_status" -ne 0 ]]; then
  printf '[gate0] FAIL: candidate-label alignment unresolved; dependent Gates are NOT_RUN\n' >&2
  exit 4
fi
printf '[gate0] PASS\n'
