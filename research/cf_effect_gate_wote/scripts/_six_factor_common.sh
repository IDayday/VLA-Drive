#!/usr/bin/env bash

set -Eeuo pipefail

six_factor_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$six_factor_script_dir/_common.sh"

six_factor_wote_root="${CF_SIX_FACTOR_WOTE_ROOT:-$cf_gate_project_root/../third_party/WoTE}"
six_factor_release_root="${CF_SIX_FACTOR_WOTE_RELEASE_ROOT:-$cf_gate_project_root/../third_party/WoTE_release/wote}"
six_factor_output_root="${CF_SIX_FACTOR_EXPERIMENT_ROOT:-$cf_gate_project_root/experiments/cf_effect_wote_six_factor}"
six_factor_report_dir="${CF_SIX_FACTOR_REPORT_DIR:-$cf_gate_project_root/reports/cf_effect_wote_six_factor}"

six_factor_checkpoint="$six_factor_release_root/epoch=29-step=19950.ckpt"
six_factor_anchors="$six_factor_release_root/extra_data/planning_vb/trajectory_anchors_256.npy"
six_factor_published="$six_factor_release_root/extra_data/planning_vb/formatted_pdm_score_256.npy"
six_factor_contract="$six_factor_report_dir/EVALUATOR_CONTRACT.json"

six_factor_expected_wote_commit="298957c128a91d41a1c6075bd0bb6e7e845e093f"
six_factor_expected_checkpoint_sha="f5e73261cc55220d681bdfe2ce306a2f8e8cd555b10be51034e9b20e2967e53b"
six_factor_expected_anchor_sha="44f64a763473c3a80482aaa3f78669445f56af40a1c00741a351c6c0650e758b"
six_factor_expected_tokens_sha="d33eae408b8d5bfba7bebd5d83d47755ba659fead2d11f2332f00f1a905da011"

six_factor_require_sha() {
  local path="$1"
  local expected="$2"
  local label="$3"
  cf_gate_require_file "$path" "$label" || return 1
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    printf '[six-factor-gate] %s SHA256 mismatch: expected %s, got %s\n' \
      "$label" "$expected" "$actual" >&2
    return 1
  fi
}

six_factor_verify_assets() {
  cf_gate_require_dir "$six_factor_wote_root/.git" 'external WoTE checkout' || return 1
  local actual_commit
  actual_commit="$(git -C "$six_factor_wote_root" rev-parse HEAD)"
  if [[ "$actual_commit" != "$six_factor_expected_wote_commit" ]]; then
    printf '[six-factor-gate] WoTE commit mismatch: expected %s, got %s\n' \
      "$six_factor_expected_wote_commit" "$actual_commit" >&2
    return 1
  fi
  six_factor_require_sha "$six_factor_checkpoint" \
    "$six_factor_expected_checkpoint_sha" checkpoint || return 1
  six_factor_require_sha "$six_factor_anchors" \
    "$six_factor_expected_anchor_sha" candidate-bank || return 1
}

six_factor_write_contract_if_absent() {
  if [[ ! -e "$six_factor_contract" ]]; then
    cf_gate_run_python -m research.cf_effect_gate_wote.src.independent_relabel \
      write-six-factor-contract --wote-root "$six_factor_wote_root" \
      --checkpoint "$six_factor_checkpoint" --anchors "$six_factor_anchors" \
      --output "$six_factor_contract"
  fi
}
