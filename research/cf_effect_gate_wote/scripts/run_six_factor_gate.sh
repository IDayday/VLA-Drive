#!/usr/bin/env bash

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/_six_factor_common.sh"

data_root="${CF_SIX_FACTOR_DATA_ROOT:-}"
map_root="${CF_SIX_FACTOR_MAP_ROOT:-${NUPLAN_MAPS_ROOT:-}}"
metric_cache_root="${CF_SIX_FACTOR_METRIC_CACHE_ROOT:-}"
device="${CF_SIX_FACTOR_DEVICE:-cuda}"
while (($#)); do
  case "$1" in
    --data-root) data_root="$2"; shift 2 ;;
    --map-root) map_root="$2"; shift 2 ;;
    --metric-cache-root) metric_cache_root="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --wote-root) six_factor_wote_root="$2"; shift 2 ;;
    --release-root) six_factor_release_root="$2"; shift 2 ;;
    --output-root) six_factor_output_root="$2"; shift 2 ;;
    --report-dir) six_factor_report_dir="$2"; shift 2 ;;
    *) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "$data_root" ]] || { printf '%s\n' '--data-root is required' >&2; exit 2; }
[[ -n "$map_root" ]] || { printf '%s\n' '--map-root is required' >&2; exit 2; }

six_factor_checkpoint="$six_factor_release_root/epoch=29-step=19950.ckpt"
six_factor_anchors="$six_factor_release_root/extra_data/planning_vb/trajectory_anchors_256.npy"
six_factor_published="$six_factor_release_root/extra_data/planning_vb/formatted_pdm_score_256.npy"
six_factor_contract="$six_factor_report_dir/EVALUATOR_CONTRACT.json"
tokens="$cf_gate_project_root/research/cf_effect_gate_wote/configs/splits/relabel_headroom_tokens.txt"

[[ ! -e "$six_factor_output_root" ]] || { printf 'refusing existing output root: %s\n' "$six_factor_output_root" >&2; exit 3; }
[[ ! -e "$six_factor_report_dir" ]] || { printf 'refusing existing report root: %s\n' "$six_factor_report_dir" >&2; exit 3; }
six_factor_verify_assets
cf_gate_require_dir "$data_root/navsim_logs/trainval" 'NAVSIM trainval logs' || exit 3
cf_gate_require_dir "$data_root/sensor_blobs/trainval" 'NAVSIM trainval sensors' || exit 3
cf_gate_require_dir "$map_root" 'nuPlan map root' || exit 3
six_factor_require_sha "$tokens" "$six_factor_expected_tokens_sha" 'fixed 200-token split'

pytest "$cf_gate_project_root/research/cf_effect_gate_wote/tests" -q
mkdir -p "$six_factor_output_root"
if [[ -z "$metric_cache_root" ]]; then
  metric_cache_root="$six_factor_output_root/metric-cache-headroom-200"
  cf_gate_run_python -m research.cf_effect_gate_wote.src.cache_metric_subset \
    --wote-root "$six_factor_wote_root" --data-root "$data_root" \
    --map-root "$map_root" --tokens "$tokens" --output "$metric_cache_root"
else
  cf_gate_require_dir "$metric_cache_root/metadata" 'provided metric cache metadata' || exit 3
fi

common=(--metric-cache-root "$metric_cache_root" --wote-root "$six_factor_wote_root"
  --release-root "$six_factor_release_root" --output-root "$six_factor_output_root"
  --report-dir "$six_factor_report_dir")
"$script_dir/run_gate0r2a_single_scene.sh" "${common[@]}"
"$script_dir/run_gate0r2b_determinism10.sh" "${common[@]}"
"$script_dir/run_gate0r2c_determinism200.sh" "${common[@]}"

cf_gate_run_python -m research.cf_effect_gate_wote.src.six_factor_verdict published-audit \
  --labels "$six_factor_output_root/g0r2c-run1" --published "$six_factor_published" \
  --output-csv "$six_factor_report_dir/published_vs_six_factor_audit.csv" \
  --output-summary "$six_factor_report_dir/published_vs_six_factor_summary.json"

"$script_dir/run_gate1r2_candidate_headroom.sh" \
  --data-root "$data_root" --device "$device" --wote-root "$six_factor_wote_root" \
  --release-root "$six_factor_release_root" --output-root "$six_factor_output_root" \
  --report-dir "$six_factor_report_dir"

cf_gate_run_python -m research.cf_effect_gate_wote.src.six_factor_verdict build-reports \
  --report-dir "$six_factor_report_dir" \
  --single-summary "$six_factor_output_root/g0r2a-single/summary.json" \
  --ten-summary "$six_factor_report_dir/g0r2b_summary.json" \
  --two-hundred-summary "$six_factor_report_dir/relabel_consistency_200_summary.json" \
  --headroom-summary "$six_factor_report_dir/candidate_headroom_summary.json" \
  --tokens "$tokens"

printf '%s\n' '[six-factor-gate] stopped after G1-R2 as required; effect/forward/inverse tasks are NOT_RUN'
