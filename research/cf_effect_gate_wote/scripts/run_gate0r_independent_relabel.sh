#!/usr/bin/env bash

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/_common.sh"

wote_root="${CF_RELABEL_WOTE_ROOT:-$cf_gate_project_root/../third_party/WoTE}"
release_root="${CF_RELABEL_WOTE_RELEASE_ROOT:-$cf_gate_project_root/../third_party/WoTE_release/wote}"
metric_cache_root="${CF_RELABEL_METRIC_CACHE_ROOT:-}"
output_root="${CF_RELABEL_EXPERIMENT_ROOT:-$cf_gate_project_root/experiments/cf_effect_wote_relabel}"
report_dir="${CF_RELABEL_REPORT_DIR:-$cf_gate_project_root/reports/cf_effect_wote_relabel}"
run_id="g0r-independent-relabel-200"
dry_run=0

while (($#)); do
  case "$1" in
    --wote-root|--release-root|--metric-cache-root|--output-root|--report-dir|--run-id)
      [[ $# -ge 2 ]] || { printf 'missing value for %s\n' "$1" >&2; exit 2; }
      case "$1" in
        --wote-root) wote_root="$2" ;;
        --release-root) release_root="$2" ;;
        --metric-cache-root) metric_cache_root="$2" ;;
        --output-root) output_root="$2" ;;
        --report-dir) report_dir="$2" ;;
        --run-id) run_id="$2" ;;
      esac
      shift 2
      ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help)
      printf '%s\n' 'Usage: run_gate0r_independent_relabel.sh --metric-cache-root PATH [options]'
      exit 0
      ;;
    *) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

[[ -n "$metric_cache_root" ]] || { printf '%s\n' '--metric-cache-root is required' >&2; exit 2; }
[[ "$run_id" =~ ^[A-Za-z0-9._-]+$ ]] || { printf 'unsafe run id: %s\n' "$run_id" >&2; exit 2; }

tokens="$cf_gate_project_root/research/cf_effect_gate_wote/configs/splits/relabel_headroom_tokens.txt"
anchors="$release_root/extra_data/planning_vb/trajectory_anchors_256.npy"
checkpoint="$release_root/epoch=29-step=19950.ckpt"
published="$release_root/extra_data/planning_vb/formatted_pdm_score_256.npy"
run_root="$output_root/$run_id"
run1="$run_root/relabel-run1"
run2="$run_root/relabel-run2"
contract="$report_dir/EVALUATOR_CONTRACT.json"

contract_command=(python -m research.cf_effect_gate_wote.src.independent_relabel write-contract
  --wote-root "$wote_root" --checkpoint "$checkpoint" --anchors "$anchors" --output "$contract")
run_base=(python -m research.cf_effect_gate_wote.src.independent_relabel run
  --wote-root "$wote_root" --metric-cache-root "$metric_cache_root" --tokens "$tokens"
  --anchors "$anchors" --evaluator-contract "$contract" --expected-scenes 200 --shard-scenes 16)

if ((dry_run)); then
  cf_gate_print_command "${contract_command[@]}"
  cf_gate_print_command "${run_base[@]}" --output "$run1"
  cf_gate_print_command "${run_base[@]}" --output "$run2"
  cf_gate_print_command python -m research.cf_effect_gate_wote.src.relabel_consistency compare \
    --run1 "$run1" --run2 "$run2" --output-csv "$report_dir/relabel_consistency.csv" \
    --output-summary "$report_dir/relabel_consistency_summary.json"
  cf_gate_print_command python -m research.cf_effect_gate_wote.src.relabel_consistency published-audit \
    --independent-labels "$run1" --published-scores "$published" \
    --output-csv "$report_dir/published_vs_independent_audit.csv" \
    --output-summary "$report_dir/published_vs_independent_summary.json"
  exit 0
fi

cf_gate_require_dir "$wote_root/.git" 'pinned external WoTE checkout' || exit 3
cf_gate_require_dir "$metric_cache_root/metadata" 'fixed-token metric cache metadata' || exit 3
cf_gate_require_file "$tokens" 'fixed 200-scene token file' || exit 3
cf_gate_require_file "$anchors" 'released base anchors' || exit 3
cf_gate_require_file "$checkpoint" 'released checkpoint' || exit 3
[[ ! -e "$run_root" ]] || { printf 'refusing existing run root: %s\n' "$run_root" >&2; exit 3; }
[[ ! -e "$report_dir" ]] || { printf 'refusing existing report root: %s\n' "$report_dir" >&2; exit 3; }
mkdir -p "$run_root" "$report_dir"

PYTHONPATH="$cf_gate_pythonpath" "${contract_command[@]}"
set +e
PYTHONPATH="$cf_gate_pythonpath" "${run_base[@]}" --output "$run1"
run1_status=$?
set -e
if [[ "$run1_status" -ne 0 ]]; then
  if [[ "$run1_status" -eq 4 && -f "$run1/failure.json" ]]; then
    PYTHONPATH="$cf_gate_pythonpath" python -m research.cf_effect_gate_wote.src.oracle_effect_verdict \
      build-g0-failure-reports --report-dir "$report_dir" --failure "$run1/failure.json" \
      --tokens "$tokens" --checkpoint "$checkpoint" --anchors "$anchors" \
      --evaluator-contract "$contract"
    printf '%s\n' '[G0-R] FAIL; run2, audit, G1-R and G2-O are NOT_RUN' >&2
  fi
  exit "$run1_status"
fi

PYTHONPATH="$cf_gate_pythonpath" "${run_base[@]}" --output "$run2"
PYTHONPATH="$cf_gate_pythonpath" python -m research.cf_effect_gate_wote.src.relabel_consistency compare \
  --run1 "$run1" --run2 "$run2" --output-csv "$report_dir/relabel_consistency.csv" \
  --output-summary "$report_dir/relabel_consistency_summary.json"
PYTHONPATH="$cf_gate_pythonpath" python -m research.cf_effect_gate_wote.src.relabel_consistency published-audit \
  --independent-labels "$run1" --published-scores "$published" \
  --output-csv "$report_dir/published_vs_independent_audit.csv" \
  --output-summary "$report_dir/published_vs_independent_summary.json"
printf '%s\n' '[G0-R] PASS'

