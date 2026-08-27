#!/usr/bin/env bash

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/_common.sh"

release_root="${CF_RELABEL_WOTE_RELEASE_ROOT:-$cf_gate_project_root/../third_party/WoTE_release/wote}"
output_root="${CF_RELABEL_EXPERIMENT_ROOT:-$cf_gate_project_root/experiments/cf_effect_wote_relabel}"
report_dir="${CF_RELABEL_REPORT_DIR:-$cf_gate_project_root/reports/cf_effect_wote_relabel}"
failure="${1:-$output_root/g0r-independent-relabel-200/relabel-run1/failure.json}"
tokens="$cf_gate_project_root/research/cf_effect_gate_wote/configs/splits/relabel_headroom_tokens.txt"
checkpoint="$release_root/epoch=29-step=19950.ckpt"
anchors="$release_root/extra_data/planning_vb/trajectory_anchors_256.npy"
contract="$report_dir/EVALUATOR_CONTRACT.json"

cf_gate_require_file "$failure" 'G0-R failure evidence' || exit 3
cf_gate_run_python -m research.cf_effect_gate_wote.src.oracle_effect_verdict \
  build-g0-failure-reports --report-dir "$report_dir" --failure "$failure" \
  --tokens "$tokens" --checkpoint "$checkpoint" --anchors "$anchors" \
  --evaluator-contract "$contract"
