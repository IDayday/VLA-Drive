#!/usr/bin/env bash

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/_common.sh"

report_dir="${CF_RELABEL_REPORT_DIR:-$cf_gate_project_root/reports/cf_effect_wote_relabel}"
split_dir="$cf_gate_project_root/research/cf_effect_gate_wote/configs/splits"
headroom_summary="${CF_RELABEL_G1_SUMMARY:-$cf_gate_project_root/experiments/cf_effect_wote_relabel/g1r-headroom/candidate_headroom_summary.json}"
dry_run=0

while (($#)); do
  case "$1" in
    --report-dir) report_dir="$2"; shift 2 ;;
    --headroom-summary) headroom_summary="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    *) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

split_command=(python -m research.cf_effect_gate_wote.src.oracle_effect_verdict build-probe-split
  --split-dir "$split_dir" --headroom-tokens "$split_dir/relabel_headroom_tokens.txt"
  --output-dir "$split_dir")
if ((dry_run)); then
  cf_gate_print_command "${split_command[@]}"
  printf '%s\n' 'G2-O then uses independent relabeling, --label-source none caches, primitive effects, and seeds 0/1/2.'
  exit 0
fi

cf_gate_require_file "$headroom_summary" 'G1-R headroom summary' || exit 3
PYTHONPATH="$cf_gate_pythonpath" python - "$headroom_summary" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("gate_g1r") != "PASS":
    raise SystemExit("G2-O is NOT_RUN because G1-R did not pass")
PY
cf_gate_run_python -m research.cf_effect_gate_wote.src.oracle_effect_verdict build-probe-split \
  --split-dir "$split_dir" --headroom-tokens "$split_dir/relabel_headroom_tokens.txt" \
  --output-dir "$split_dir"
printf '%s\n' 'Probe split created. Run the relabel/cache/effect/probe stages recorded in oracle_effect_probe.yaml.'

