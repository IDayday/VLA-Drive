#!/usr/bin/env bash

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/_six_factor_common.sh"

metric_cache_root="${CF_SIX_FACTOR_METRIC_CACHE_ROOT:-}"
while (($#)); do
  case "$1" in
    --metric-cache-root) metric_cache_root="$2"; shift 2 ;;
    --wote-root) six_factor_wote_root="$2"; shift 2 ;;
    --release-root) six_factor_release_root="$2"; shift 2 ;;
    --output-root) six_factor_output_root="$2"; shift 2 ;;
    --report-dir) six_factor_report_dir="$2"; shift 2 ;;
    *) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "$metric_cache_root" ]] || { printf '%s\n' '--metric-cache-root is required' >&2; exit 2; }

six_factor_anchors="$six_factor_release_root/extra_data/planning_vb/trajectory_anchors_256.npy"
six_factor_contract="$six_factor_report_dir/EVALUATOR_CONTRACT.json"
tokens="$cf_gate_project_root/research/cf_effect_gate_wote/configs/splits/relabel_determinism10_tokens.txt"
single_summary="$six_factor_output_root/g0r2a-single/summary.json"
run1="$six_factor_output_root/g0r2b-run1"
run2="$six_factor_output_root/g0r2b-run2"

cf_gate_run_python - "$single_summary" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("gate") != "SINGLE_SCENE_SIX_FACTOR_PASS":
    raise SystemExit("G0-R2b is NOT_RUN because G0-R2a did not pass")
PY
[[ ! -e "$run1" && ! -e "$run2" ]] || { printf '%s\n' 'refusing existing G0-R2b run' >&2; exit 3; }
for run in "$run1" "$run2"; do
  cf_gate_run_python -m research.cf_effect_gate_wote.src.independent_relabel run-six-factor \
    --wote-root "$six_factor_wote_root" --metric-cache-root "$metric_cache_root" \
    --tokens "$tokens" --anchors "$six_factor_anchors" \
    --evaluator-contract "$six_factor_contract" --output "$run" \
    --expected-scenes 10 --shard-scenes 10
done
cf_gate_run_python -m research.cf_effect_gate_wote.src.six_factor_verdict compare \
  --run1 "$run1" --run2 "$run2" --expected-scenes 10 \
  --pass-status TEN_SCENE_DETERMINISM_PASS \
  --output-csv "$six_factor_report_dir/g0r2b_determinism.csv" \
  --output-summary "$six_factor_report_dir/g0r2b_summary.json"
