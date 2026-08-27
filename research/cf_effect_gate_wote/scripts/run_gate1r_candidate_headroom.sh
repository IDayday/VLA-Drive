#!/usr/bin/env bash

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/_common.sh"

report_dir="${CF_RELABEL_REPORT_DIR:-$cf_gate_project_root/reports/cf_effect_wote_relabel}"
labels="${CF_RELABEL_HEADROOM_LABELS:-}"
feature_cache="${CF_RELABEL_HEADROOM_FEATURE_CACHE:-}"
output_root="${CF_RELABEL_EXPERIMENT_ROOT:-$cf_gate_project_root/experiments/cf_effect_wote_relabel}"
dry_run=0

while (($#)); do
  case "$1" in
    --report-dir|--labels|--feature-cache|--output-root)
      [[ $# -ge 2 ]] || { printf 'missing value for %s\n' "$1" >&2; exit 2; }
      case "$1" in
        --report-dir) report_dir="$2" ;;
        --labels) labels="$2" ;;
        --feature-cache) feature_cache="$2" ;;
        --output-root) output_root="$2" ;;
      esac
      shift 2
      ;;
    --dry-run) dry_run=1; shift ;;
    *) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

consistency="$report_dir/relabel_consistency_summary.json"
selected="$output_root/g1r-headroom/wote_base_selected.json"
scene_level="$report_dir/candidate_headroom_scene_level.parquet"
summary="$output_root/g1r-headroom/candidate_headroom_summary.json"

if ((dry_run)); then
  cf_gate_print_command python -m research.cf_effect_gate_wote.src.candidate_alignment selected-from-cache \
    --cache-root "$feature_cache" --output "$selected"
  cf_gate_print_command python -m research.cf_effect_gate_wote.src.oracle_effect_verdict headroom \
    --labels "$labels" --selected-indices "$selected" --output-parquet "$scene_level" \
    --output-summary "$summary"
  exit 0
fi

cf_gate_require_file "$consistency" 'G0-R consistency summary' || exit 3
PYTHONPATH="$cf_gate_pythonpath" python - "$consistency" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("status") != "PASS":
    raise SystemExit("G1-R is NOT_RUN because G0-R did not pass")
PY
[[ -n "$labels" && -n "$feature_cache" ]] || { printf '%s\n' '--labels and --feature-cache are required' >&2; exit 2; }
mkdir -p "$(dirname -- "$selected")"
cf_gate_run_python -m research.cf_effect_gate_wote.src.candidate_alignment selected-from-cache \
  --cache-root "$feature_cache" --output "$selected"
cf_gate_run_python -m research.cf_effect_gate_wote.src.oracle_effect_verdict headroom \
  --labels "$labels" --selected-indices "$selected" --output-parquet "$scene_level" \
  --output-summary "$summary"

