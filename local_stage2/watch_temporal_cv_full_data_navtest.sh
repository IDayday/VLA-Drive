#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/local_stage2/common.sh"

cv_root="${TEMPORAL_CV_ROOT:-${DRIVEVLA_STAGE2_RUN_ROOT}/scorer_pdms93/temporal_consequence_cv5_dedicated_v1}"
full_root="${TEMPORAL_FULL_ROOT:-${DRIVEVLA_STAGE2_RUN_ROOT}/scorer_pdms93/temporal_consequence_full_data_v1}"
policy_root="${TEMPORAL_POLICY_ROOT:-${DRIVEVLA_STAGE2_RUN_ROOT}/scorer_pdms93/temporal_consequence_cv5_common_policy_v1}"
fold_navtest_root="${TEMPORAL_FOLD_NAVTEST_ROOT:-${DRIVEVLA_STAGE2_RUN_ROOT}/scorer_pdms93/temporal_consequence_navtest_full_v1}"
final_navtest_root="${TEMPORAL_FINAL_NAVTEST_ROOT:-${DRIVEVLA_STAGE2_RUN_ROOT}/scorer_pdms93/temporal_consequence_common_and_full_navtest_v1}"

feature_cache="${TEMPORAL_NAVTEST_FEATURE_CACHE:-${DRIVEVLA_STAGE2_RUN_ROOT}/ke_candidate_audit/public_base_navtest_scorer_features_full_fp32_v2/proposal_predictions.pkl}"
feature_manifest="${TEMPORAL_NAVTEST_FEATURE_MANIFEST:-${DRIVEVLA_STAGE2_RUN_ROOT}/ke_candidate_audit/public_base_navtest_scorer_features_full_fp32_v2/proposal_cache_manifest.json}"
candidate_matrix="${TEMPORAL_NAVTEST_CANDIDATE_MATRIX:-${DRIVEVLA_STAGE2_RUN_ROOT}/ke_candidate_audit/public_base_navtest_all_candidate_factors_fp32_v1/candidate_scores.npz}"
candidate_summary="${TEMPORAL_NAVTEST_CANDIDATE_SUMMARY:-${DRIVEVLA_STAGE2_RUN_ROOT}/ke_candidate_audit/public_base_navtest_all_candidate_factors_fp32_v1/summary.json}"

mkdir -p "${cv_root}/logs" "${full_root}/logs" "${policy_root}" "${final_navtest_root}"

while [[ "$(find "${cv_root}" -mindepth 2 -maxdepth 2 -name training_results.json | wc -l)" -lt 5 ]]; do
  printf 'WAIT_CV %s results=%s/5\n' \
    "$(date -u +%FT%TZ)" \
    "$(find "${cv_root}" -mindepth 2 -maxdepth 2 -name training_results.json | wc -l)"
  sleep 30
done

cv_summary="${cv_root}/cv_summary_final.json"
"${DRIVEVLA_PYTHON}" "${repo_root}/local_stage2/summarize_temporal_consequence_cv.py" \
  --run-root "${cv_root}" \
  --output "${cv_summary}" \
  --report "${cv_root}/cv_summary_final.md"

artifact_args=()
for fold in 0 1 2 3 4; do
  artifact_args+=(
    --artifact "${cv_root}/fold_${fold}_seed2/best_temporal_consequence_scorer.pt"
  )
done
"${DRIVEVLA_PYTHON}" "${repo_root}/local_stage2/materialize_temporal_cv_policy.py" \
  --cv-summary "${cv_summary}" \
  --output-root "${policy_root}" \
  "${artifact_args[@]}"

launch_full_data() {
  local gpu="$1"
  local seed="$2"
  local output="${full_root}/seed_${seed}"
  local log="${full_root}/logs/seed_${seed}.log"
  if [[ -f "${output}/training_results.json" ]]; then
    printf 'SKIP_FULL %s seed=%s complete\n' "$(date -u +%FT%TZ)" "${seed}"
    return
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" \
    bash "${repo_root}/local_stage2/train_temporal_consequence_scorer.sh" \
      --output-dir "${output}" \
      --train-all \
      --cv-summary "${cv_summary}" \
      --seed "${seed}" \
      --epochs 12 \
      --batch-size 128 \
      --eval-batch-size 256 \
      --device cuda >"${log}" 2>&1
}

launch_full_data 3 0 &
pid0=$!
launch_full_data 6 1 &
pid1=$!
launch_full_data 7 2 &
pid2=$!
wait "${pid0}" "${pid1}" "${pid2}"

# The already-running fold campaign owns GPU 5 first.  Waiting for its atomic
# campaign summary prevents overlap and proves every validation-positive fold
# artifact was tested before the common-policy/full-data campaign starts.
while [[ ! -f "${fold_navtest_root}/campaign_summary.json" ]]; do
  printf 'WAIT_FOLD_NAVTEST %s\n' "$(date -u +%FT%TZ)"
  sleep 30
done

evaluation_args=()
for fold in 0 1 2 3 4; do
  evaluation_args+=(
    --artifact "common_policy_fold_${fold}=${policy_root}/fold_${fold}_seed2/common_policy_temporal_consequence_scorer.pt"
  )
done
for seed in 0 1 2; do
  evaluation_args+=(
    --artifact "full_data_seed_${seed}=${full_root}/seed_${seed}/best_temporal_consequence_scorer.pt"
  )
done

CUDA_VISIBLE_DEVICES=5 "${DRIVEVLA_PYTHON}" \
  "${repo_root}/local_stage2/evaluate_cached_navtest_scorers.py" \
  --feature-cache "${feature_cache}" \
  --feature-manifest "${feature_manifest}" \
  --candidate-matrix "${candidate_matrix}" \
  --candidate-summary "${candidate_summary}" \
  --output-dir "${final_navtest_root}" \
  --device cuda \
  --batch-size 128 \
  --bootstrap-iterations 10000 \
  "${evaluation_args[@]}"

printf 'TEMPORAL_CAMPAIGN_COMPLETE %s output=%s\n' \
  "$(date -u +%FT%TZ)" "${final_navtest_root}"
