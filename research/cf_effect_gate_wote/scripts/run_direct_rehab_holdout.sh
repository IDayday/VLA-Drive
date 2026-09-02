#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

stage="${1:-preflight}"
case "$stage" in
  preflight|metric-cache|feature-smoke|features|labels|evaluate-final|all) ;;
  *)
    printf 'usage: %s {preflight|metric-cache|feature-smoke|features|labels|evaluate-final|all}\n' "$0" >&2
    exit 2
    ;;
esac

run_root="${DIRECT_REHAB_RUN_ROOT:-$repo_root/experiments/cf_effect_wote_direct_rehab/direct-rehab-v1-20260901}"
oracle_run="${ORACLE_RUN:-/mnt/workspace/project/VLA-Drive-cf-effect-oracle/experiments/cf_effect_wote_oracle_effect/oracle-effect-v2-20260829}"
data_root="${NAVSIM_DATA_VIEW:-$oracle_run/data-view}"
map_root="${NUPLAN_MAPS_ROOT:-/mnt/data_and_weight/Public_Space/navsim/maps}"
wote_root="${WOTE_ROOT:-/mnt/workspace/project/third_party/WoTE}"
release_root="${WOTE_RELEASE_ROOT:-/mnt/workspace/project/third_party/WoTE_release/wote}"
tokens="$repo_root/research/cf_effect_gate_wote/configs/splits/direct_rehab_holdout_512.txt"
access_policy="$repo_root/reports/cf_effect_wote_direct_rehab/ACCESS_POLICY.json"
access_log="$run_root/audit/access_log.jsonl"
contract="$repo_root/reports/cf_effect_wote_six_factor/EVALUATOR_CONTRACT.json"
anchors="$release_root/extra_data/planning_vb/trajectory_anchors_256.npy"
checkpoint="$release_root/epoch=29-step=19950.ckpt"
policy="$run_root/safe-ensemble-v1/dev_final_policy_a3_v2.json"
metric_cache="$run_root/metric-cache-holdout-512-v2"
feature_cache="$run_root/features-holdout-512-v2"
feature_provenance="$run_root/features-holdout-512-v2.provenance.json"
feature_smoke_cache="$run_root/features-holdout-smoke1-v3"
feature_smoke_provenance="$run_root/features-holdout-smoke1-v3.provenance.json"
label_store="$run_root/labels-holdout-512-v2"
final_output="$run_root/final-holdout-evaluation-v1.json"
commands_log="$run_root/COMMANDS_HOLDOUT.sh"
checkpoints=(
  "$run_root/confirmation-v1/hybrid_current-seed0.pt"
  "$run_root/confirmation-v1/hybrid_current-seed1.pt"
  "$run_root/confirmation-v1/hybrid_current-seed2.pt"
)

expected_checkpoint_sha="f5e73261cc55220d681bdfe2ce306a2f8e8cd555b10be51034e9b20e2967e53b"
expected_anchor_sha="44f64a763473c3a80482aaa3f78669445f56af40a1c00741a351c6c0650e758b"
expected_contract_sha="e1e376c9fc4c7e6020d0e18e5c2e061e2a7c53d91bb1c38da751139f4c69a98b"
expected_wote_commit="298957c128a91d41a1c6075bd0bb6e7e845e093f"
expected_policy_sha="bed6c2a8471fc1cf9af90f466a3741b0edfa37522a8ccf0c967d1c58d41c2309"
expected_probe_shas=(
  "03a3468165a0eae39ec1dbb5f86f7b3d85c69364732d000721c89624054a6b20"
  "7d0097063dd78d7365a908beb501a06ecc4d53964a2a9f28c861fd282db2d7cc"
  "b39285ac88aa13a13a8253e16f15f1034113af8bf81b8369b3412f4824e91a19"
)

require_file() {
  [[ -f "$1" ]] || { printf 'missing required file: %s\n' "$1" >&2; exit 3; }
}

require_dir() {
  [[ -d "$1" ]] || { printf 'missing required directory: %s\n' "$1" >&2; exit 3; }
}

verify_sha() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    printf 'SHA256 mismatch: %s expected=%s actual=%s\n' "$path" "$expected" "$actual" >&2
    exit 4
  }
}

refuse_existing() {
  [[ ! -e "$1" ]] || { printf 'refusing existing output: %s\n' "$1" >&2; exit 5; }
}

record_and_run() {
  mkdir -p "$(dirname "$commands_log")"
  if [[ ! -e "$commands_log" ]]; then
    printf '#!/usr/bin/env bash\nset -Eeuo pipefail\n' >"$commands_log"
  fi
  {
    printf 'COMMAND'
    printf ' %q' "$@"
    printf '\n'
  } >>"$commands_log"
  printf '[direct-rehab-holdout]'
  printf ' %q' "$@"
  printf '\n'
  PYTHONPATH="$repo_root" "$@"
}

preflight() {
  require_dir "$data_root/navsim_logs/trainval"
  require_dir "$data_root/sensor_blobs/trainval"
  require_dir "$map_root"
  require_dir "$wote_root/.git"
  require_file "$tokens"
  require_file "$access_policy"
  require_file "$contract"
  require_file "$anchors"
  require_file "$checkpoint"
  require_file "$policy"
  for checkpoint_index in "${!checkpoints[@]}"; do
    require_file "${checkpoints[$checkpoint_index]}"
    verify_sha \
      "${checkpoints[$checkpoint_index]}" \
      "${expected_probe_shas[$checkpoint_index]}"
  done
  verify_sha "$checkpoint" "$expected_checkpoint_sha"
  verify_sha "$anchors" "$expected_anchor_sha"
  verify_sha "$contract" "$expected_contract_sha"
  verify_sha "$policy" "$expected_policy_sha"
  local wote_commit
  wote_commit="$(git -C "$wote_root" rev-parse HEAD)"
  [[ "$wote_commit" == "$expected_wote_commit" ]] || {
    printf 'WoTE commit mismatch: expected=%s actual=%s\n' "$expected_wote_commit" "$wote_commit" >&2
    exit 4
  }
  [[ "$(wc -l <"$tokens")" -eq 512 ]] || {
    printf 'holdout token file must contain exactly 512 lines\n' >&2
    exit 4
  }
  record_and_run python -m research.cf_effect_gate_wote.src.direct_rehab_contracts \
    audit-access \
    --access-policy "$access_policy" \
    --log "$access_log"
}

run_metric_cache() {
  refuse_existing "$metric_cache"
  record_and_run python -m research.cf_effect_gate_wote.src.cache_metric_subset \
    --wote-root "$wote_root" \
    --data-root "$data_root" \
    --map-root "$map_root" \
    --tokens "$tokens" \
    --output "$metric_cache" \
    --access-policy "$access_policy" \
    --access-log "$access_log" \
    --access-phase asset_generation
}

run_features() {
  refuse_existing "$feature_cache"
  refuse_existing "$feature_provenance"
  record_and_run python -m research.cf_effect_gate_wote.src.direct_current_cache cache \
    --wote-root "$wote_root" \
    --release-root "$release_root" \
    --data-root "$data_root" \
    --tokens "$tokens" \
    --output "$feature_cache" \
    --run-id direct-rehab-v1-holdout-v2 \
    --split holdout \
    --device cuda \
    --shard-scenes 16 \
    --include-selector-reference \
    --provenance-output "$feature_provenance" \
    --access-policy "$access_policy" \
    --access-log "$access_log" \
    --access-phase asset_generation
}

run_feature_smoke() {
  refuse_existing "$feature_smoke_cache"
  refuse_existing "$feature_smoke_provenance"
  record_and_run python -m research.cf_effect_gate_wote.src.direct_current_cache cache \
    --wote-root "$wote_root" \
    --release-root "$release_root" \
    --data-root "$data_root" \
    --tokens "$tokens" \
    --output "$feature_smoke_cache" \
    --run-id direct-rehab-v1-holdout-smoke1-v3 \
    --split holdout \
    --device cuda \
    --shard-scenes 1 \
    --limit 1 \
    --include-selector-reference \
    --provenance-output "$feature_smoke_provenance" \
    --access-policy "$access_policy" \
    --access-log "$access_log" \
    --access-phase asset_generation
}

run_labels() {
  require_dir "$metric_cache"
  refuse_existing "$label_store"
  record_and_run python -m research.cf_effect_gate_wote.src.independent_relabel \
    run-six-factor \
    --wote-root "$wote_root" \
    --metric-cache-root "$metric_cache" \
    --tokens "$tokens" \
    --anchors "$anchors" \
    --evaluator-contract "$contract" \
    --output "$label_store" \
    --expected-scenes 512 \
    --shard-scenes 16 \
    --access-policy "$access_policy" \
    --access-log "$access_log" \
    --access-phase final_evaluation
}

run_final_evaluation() {
  require_dir "$feature_cache"
  require_dir "$label_store"
  refuse_existing "$final_output"
  record_and_run python -m research.cf_effect_gate_wote.src.direct_rehab_ensemble \
    evaluate-final \
    --feature-root "$feature_cache" \
    --label-root "$label_store" \
    --tokens "$tokens" \
    --checkpoints "${checkpoints[@]}" \
    --policy "$policy" \
    --phase final_evaluation \
    --batch-scenes 4 \
    --candidate-chunk 64 \
    --device cuda \
    --access-policy "$access_policy" \
    --access-log "$access_log" \
    --output "$final_output"
}

export NUPLAN_MAPS_ROOT="$map_root"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

case "$stage" in
  preflight) preflight ;;
  metric-cache) preflight; run_metric_cache ;;
  feature-smoke) preflight; run_feature_smoke ;;
  features) preflight; run_features ;;
  labels) preflight; run_labels ;;
  evaluate-final) preflight; run_final_evaluation ;;
  all)
    preflight
    run_metric_cache
    run_features
    run_labels
    run_final_evaluation
    ;;
esac
