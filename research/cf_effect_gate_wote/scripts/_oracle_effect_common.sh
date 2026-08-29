#!/usr/bin/env bash

set -Eeuo pipefail

gate2o_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$gate2o_script_dir/_common.sh"

gate2o_base_commit="e68e3e635b0a13eda51438a83ef4df86400d0dce"
gate2o_run_id="${GATE2O_RUN_ID:-oracle-effect-v2-20260829}"
gate2o_data_root="${GATE2O_DATA_ROOT:-}"
gate2o_map_root="${GATE2O_MAP_ROOT:-${NUPLAN_MAPS_ROOT:-}}"
gate2o_device="${GATE2O_DEVICE:-cuda}"
gate2o_wote_root="${GATE2O_WOTE_ROOT:-$cf_gate_project_root/../third_party/WoTE}"
gate2o_release_root="${GATE2O_RELEASE_ROOT:-$cf_gate_project_root/../third_party/WoTE_release/wote}"
gate2o_output_root="${GATE2O_OUTPUT_ROOT:-}"
gate2o_report_dir="${GATE2O_REPORT_DIR:-$cf_gate_project_root/reports/cf_effect_wote_oracle_effect}"
gate2o_metric_cache="${GATE2O_METRIC_CACHE:-}"

gate2o_refresh_paths() {
  if [[ -z "$gate2o_output_root" ]]; then
    gate2o_output_root="$cf_gate_project_root/experiments/cf_effect_wote_oracle_effect/$gate2o_run_id"
  fi
  if [[ -z "$gate2o_metric_cache" ]]; then
    gate2o_metric_cache="$gate2o_output_root/metric-cache-1792"
  fi
  gate2o_checkpoint="$gate2o_release_root/epoch=29-step=19950.ckpt"
  gate2o_anchors="$gate2o_release_root/extra_data/planning_vb/trajectory_anchors_256.npy"
  gate2o_contract="$cf_gate_project_root/reports/cf_effect_wote_six_factor/EVALUATOR_CONTRACT.json"
  gate2o_split_dir="$cf_gate_project_root/research/cf_effect_gate_wote/configs/splits"
  gate2o_config="$cf_gate_project_root/research/cf_effect_gate_wote/configs/oracle_effect_probe_v2.yaml"
  gate2o_train_tokens="$gate2o_split_dir/oracle_effect_v2_train_tokens.txt"
  gate2o_val_tokens="$gate2o_split_dir/oracle_effect_v2_val_tokens.txt"
  gate2o_test_tokens="$gate2o_split_dir/oracle_effect_v2_test_tokens.txt"
  gate2o_determinism_tokens="$gate2o_split_dir/oracle_effect_v2_determinism16_tokens.txt"
  gate2o_commands="$gate2o_output_root/COMMANDS.sh"
  gate2o_audit_root="$gate2o_output_root/audit"
}

gate2o_parse_args() {
  while (($#)); do
    case "$1" in
      --run-id) gate2o_run_id="$2"; shift 2 ;;
      --data-root) gate2o_data_root="$2"; shift 2 ;;
      --map-root) gate2o_map_root="$2"; shift 2 ;;
      --device) gate2o_device="$2"; shift 2 ;;
      --wote-root) gate2o_wote_root="$2"; shift 2 ;;
      --release-root) gate2o_release_root="$2"; shift 2 ;;
      --output-root) gate2o_output_root="$2"; shift 2 ;;
      --report-dir) gate2o_report_dir="$2"; shift 2 ;;
      --metric-cache-root) gate2o_metric_cache="$2"; shift 2 ;;
      *) printf '[gate2o] unknown option: %s\n' "$1" >&2; return 2 ;;
    esac
  done
  gate2o_refresh_paths
}

gate2o_require_common() {
  [[ -n "$gate2o_data_root" ]] || { printf '%s\n' '[gate2o] --data-root is required' >&2; return 2; }
  [[ -n "$gate2o_map_root" ]] || { printf '%s\n' '[gate2o] --map-root is required' >&2; return 2; }
  cf_gate_require_dir "$gate2o_data_root/navsim_logs/trainval" 'NAVSIM trainval logs'
  cf_gate_require_dir "$gate2o_data_root/sensor_blobs/trainval" 'NAVSIM trainval sensors'
  cf_gate_require_dir "$gate2o_map_root" 'nuPlan map root'
  cf_gate_require_dir "$gate2o_wote_root/.git" 'frozen WoTE checkout'
  cf_gate_require_file "$gate2o_checkpoint" 'frozen WoTE checkpoint'
  cf_gate_require_file "$gate2o_anchors" 'fixed 256-anchor bank'
  cf_gate_require_file "$gate2o_contract" 'independent six-factor evaluator contract'
  git -C "$cf_gate_project_root" merge-base --is-ancestor "$gate2o_base_commit" HEAD

  # The deployed NAVSIM dataclasses capture NUPLAN_MAPS_ROOT at module-import
  # time.  Validating --map-root is therefore insufficient: every Python
  # subprocess that imports Scene must inherit the resolved path.
  export NUPLAN_MAPS_ROOT="$gate2o_map_root"
}

gate2o_run() {
  mkdir -p "$gate2o_output_root"
  if [[ ! -e "$gate2o_commands" ]]; then
    printf '#!/usr/bin/env bash\nset -Eeuo pipefail\n' >"$gate2o_commands"
  fi
  {
    printf 'COMMAND'
    printf ' %q' "$@"
    printf '\n'
  } >>"$gate2o_commands"
  printf '[gate2o]'
  printf ' %q' "$@"
  printf '\n'
  PYTHONPATH="$cf_gate_pythonpath" "$@"
}

gate2o_run_python() {
  gate2o_run python "$@"
}

gate2o_export_context() {
  export GATE2O_RUN_ID="$gate2o_run_id"
  export GATE2O_DATA_ROOT="$gate2o_data_root"
  export GATE2O_MAP_ROOT="$gate2o_map_root"
  export GATE2O_DEVICE="$gate2o_device"
  export GATE2O_WOTE_ROOT="$gate2o_wote_root"
  export GATE2O_RELEASE_ROOT="$gate2o_release_root"
  export GATE2O_OUTPUT_ROOT="$gate2o_output_root"
  export GATE2O_REPORT_DIR="$gate2o_report_dir"
  export GATE2O_METRIC_CACHE="$gate2o_metric_cache"
}
