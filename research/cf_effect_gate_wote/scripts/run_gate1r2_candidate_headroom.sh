#!/usr/bin/env bash

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/_six_factor_common.sh"

data_root="${CF_SIX_FACTOR_DATA_ROOT:-}"
device="${CF_SIX_FACTOR_DEVICE:-cuda}"
while (($#)); do
  case "$1" in
    --data-root) data_root="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --wote-root) six_factor_wote_root="$2"; shift 2 ;;
    --release-root) six_factor_release_root="$2"; shift 2 ;;
    --output-root) six_factor_output_root="$2"; shift 2 ;;
    --report-dir) six_factor_report_dir="$2"; shift 2 ;;
    *) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "$data_root" ]] || { printf '%s\n' '--data-root is required' >&2; exit 2; }

tokens="$cf_gate_project_root/research/cf_effect_gate_wote/configs/splits/relabel_headroom_tokens.txt"
labels="$six_factor_output_root/g0r2c-run1"
consistency="$six_factor_report_dir/relabel_consistency_200_summary.json"
feature_cache="$six_factor_output_root/g1r2-feature-cache"
summary="$six_factor_report_dir/candidate_headroom_summary.json"

cf_gate_run_python - "$consistency" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("status") != "SIX_FACTOR_RELABEL_PASS":
    raise SystemExit("G1-R2 is NOT_RUN because G0-R2c did not pass")
PY
[[ ! -e "$feature_cache" ]] || { printf 'refusing existing feature cache: %s\n' "$feature_cache" >&2; exit 3; }
cf_gate_run_python -m research.cf_effect_gate_wote.src.cache_wote_features cache \
  --wote-root "$six_factor_wote_root" --release-root "$six_factor_release_root" \
  --data-root "$data_root" --tokens "$tokens" --output "$feature_cache" \
  --run-id g1r2-base-anchor-headroom-200 --split headroom --device "$device" \
  --shard-scenes 16 --label-source none
cf_gate_run_python -m research.cf_effect_gate_wote.src.six_factor_verdict headroom \
  --labels "$labels" --feature-cache "$feature_cache" \
  --output-parquet "$six_factor_report_dir/candidate_headroom_scene_level.parquet" \
  --output-summary "$summary" --output-ddc "$six_factor_report_dir/ddc_diagnostics.csv"
