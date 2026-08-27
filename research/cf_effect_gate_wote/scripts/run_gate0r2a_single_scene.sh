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

six_factor_checkpoint="$six_factor_release_root/epoch=29-step=19950.ckpt"
six_factor_anchors="$six_factor_release_root/extra_data/planning_vb/trajectory_anchors_256.npy"
six_factor_contract="$six_factor_report_dir/EVALUATOR_CONTRACT.json"
tokens="$cf_gate_project_root/research/cf_effect_gate_wote/configs/splits/relabel_single_scene_token.txt"
run_root="$six_factor_output_root/g0r2a-single"

six_factor_verify_assets
cf_gate_require_dir "$metric_cache_root/metadata" 'metric cache metadata' || exit 3
[[ ! -e "$run_root" ]] || { printf 'refusing existing run root: %s\n' "$run_root" >&2; exit 3; }
mkdir -p "$run_root" "$six_factor_report_dir"
six_factor_write_contract_if_absent

cf_gate_run_python -m research.cf_effect_gate_wote.src.independent_relabel run-six-factor \
  --wote-root "$six_factor_wote_root" --metric-cache-root "$metric_cache_root" \
  --tokens "$tokens" --anchors "$six_factor_anchors" \
  --evaluator-contract "$six_factor_contract" --output "$run_root/labels" \
  --expected-scenes 1 --shard-scenes 1
cf_gate_run_python -m research.cf_effect_gate_wote.src.six_factor_verdict single-scene \
  --labels "$run_root/labels" --output-csv "$run_root/six_factor_validation.csv" \
  --output-contrast "$run_root/old_five_vs_six_factor.csv" \
  --output-summary "$run_root/summary.json"
cf_gate_run_python -m research.cf_effect_gate_wote.src.six_factor_verdict single-scene \
  --labels "$run_root/labels" --output-csv "$six_factor_report_dir/g0r2a_single_scene.csv" \
  --output-contrast "$six_factor_report_dir/g0r2a_old_five_vs_six_factor.csv" \
  --output-summary "$six_factor_report_dir/g0r2a_summary.json"
