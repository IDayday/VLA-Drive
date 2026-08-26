#!/usr/bin/env bash

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/_common.sh"

experiment_root="${CF_GATE_EXPERIMENT_ROOT:-$cf_gate_project_root/experiments/cf_effect_gate_wote}"
report_dir="${CF_GATE_REPORT_DIR:-$cf_gate_project_root/reports/cf_effect_gate_wote}"
gate01_run_id="gate0-smoke"
gate2_run_id="gate2-main"
gate3_run_id="gate3-main"
gate4_run_id="gate4-main"
dry_run=0
preflight_only=0

usage() {
  cat <<'EOF'
Usage: build_report.sh [options]

  --experiment-root PATH
  --report-dir PATH
  --gate01-run-id ID
  --gate2-run-id ID
  --gate3-run-id ID
  --gate4-run-id ID
  --preflight-only
  --dry-run
  -h, --help

The report is assembled only from existing Gate artifacts. Missing dependent
Gates are rendered as NOT_RUN. Existing final report files are refused.
EOF
}

while (($#)); do
  case "$1" in
    --experiment-root|--report-dir|--gate01-run-id|--gate2-run-id|--gate3-run-id|--gate4-run-id)
      [[ $# -ge 2 ]] || { printf 'missing value for %s\n' "$1" >&2; exit 2; }
      case "$1" in
        --experiment-root) experiment_root="$2" ;;
        --report-dir) report_dir="$2" ;;
        --gate01-run-id) gate01_run_id="$2" ;;
        --gate2-run-id) gate2_run_id="$2" ;;
        --gate3-run-id) gate3_run_id="$2" ;;
        --gate4-run-id) gate4_run_id="$2" ;;
      esac
      shift 2
      ;;
    --preflight-only) preflight_only=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

command=(
  -m research.cf_effect_gate_wote.src.verdict
  --project-root "$cf_gate_project_root"
  --experiment-root "$experiment_root"
  --report-dir "$report_dir"
  --gate01-run-id "$gate01_run_id"
  --gate2-run-id "$gate2_run_id"
  --gate3-run-id "$gate3_run_id"
  --gate4-run-id "$gate4_run_id"
)

printf '[report] project_root=%s\n' "$cf_gate_project_root"
printf '[report] experiment_root=%s\n' "$experiment_root"
printf '[report] report_dir=%s\n' "$report_dir"

if ((dry_run)); then
  cf_gate_print_command python "${command[@]}"
  exit 0
fi

if ((preflight_only)); then
  cf_gate_require_dir "$experiment_root" 'experiment root' || exit 3
  for final_name in GATE_REPORT.md VERDICT.json REPRODUCTION.md probe_metrics.csv scene_level_results.parquet failure_cases.csv ablation_summary.csv; do
    if [[ -e "$report_dir/$final_name" ]]; then
      printf '[report] final target already exists: %s\n' "$report_dir/$final_name" >&2
      exit 3
    fi
  done
  printf '[report] preflight passed; missing Gate artifacts will remain NOT_RUN\n'
  exit 0
fi

cf_gate_run_python "${command[@]}"
printf '[report] complete\n'
