#!/usr/bin/env bash

# Promote and strictly evaluate every held-out-log-positive rl-zt4 wave-3
# artifact on the immutable No-VQA FP32 Navtest proposal bank.

set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
python_bin="${DRIVEVLA_PYTHON:-/mnt/project/DriveVLA-M0-env/bin/python}"
run_root="${NO_VQA_WAVE3_RUN_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_scene_token_wave3_v1}"
calibrated_root="${NO_VQA_WAVE3_CALIBRATED_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_scene_token_wave3_calibrated_v1}"
log_root="${NO_VQA_WAVE3_POST_LOG_ROOT:-/root/scorer_pdms93_logs/no_vqa_e35_scene_token_wave3_post_v1}"
report_root="${repo_root}/reports/m0_independent_scorer_representation/no_vqa_e35_scene_token_wave3_v1"
promotion_manifest="${report_root}/VALIDATION_PROMOTION.json"
package_root="${NO_VQA_WAVE3_PACKAGE_ROOT:-/root/scorer_pdms93_artifacts/no_vqa_e35_scene_token_wave3_v1}"
navtest_root="${NO_VQA_WAVE3_NAVTEST_ROOT:-/root/scorer_pdms93_navtest/no_vqa_e35_scene_token_wave3_v1}"
parity_root="${report_root}/online_cache_parity"
online_root="${NO_VQA_WAVE3_ONLINE_ROOT:-/root/scorer_pdms93_online/no_vqa_e35_scene_token_wave3_v1}"
base_checkpoint="/mnt/project/DriveVLA-M0-no-vqa/runs/training/no_vqa_full_ft_seed0_e36/lightning_logs/version_0/checkpoints/best-epoch=35-step=232416.ckpt"
resolved_config="/mnt/project/DriveVLA-M0-no-vqa/runs/training/no_vqa_full_ft_seed0_e36/code/hydra/config.yaml"
navtest_features="/mnt/project/DriveVLA-M0-stage2/runs/ke_candidate_audit/no_vqa_e35_navtest_scorer_features_fp32_v1/proposal_predictions.pkl"
navtest_scores="/root/scorer_pdms93_cache/no_vqa_e35_navtest_candidate_scores_fp32_v1"
wait_seconds="${NO_VQA_WAVE3_POST_POLL_SECONDS:-30}"

gpu_ids=(1 3 5 6 7)
names=(
  factorized_top16_cv_hybrid_safety5_seed2
  factorized_top16_cv_hybrid_safety1_seed2
  factorized_top8_cv_hybrid_safety5_seed2
  combined_top16_hybrid_safety5_seed2
  candidate_only_top16_factor_safety5_seed2
)

while [[ ! -f "${run_root}/.wave3_complete" || ! -f "${navtest_scores}/summary.json" || ! -f "${navtest_scores}/candidate_scores.npz" ]]; do
  echo "NO_VQA_WAVE3_NAVTEST waiting_for_training_and_candidate_matrix utc=$(date -u +%FT%TZ)"
  sleep "${wait_seconds}"
done

for name in "${names[@]}"; do
  [[ -f "${run_root}/${name}/training_summary.json" ]] || { echo "missing raw run ${name}" >&2; exit 2; }
  [[ -f "${calibrated_root}/${name}__calibrated/training_summary.json" ]] || { echo "missing calibrated run ${name}" >&2; exit 2; }
done
for path in "${log_root}" "${report_root}" "${package_root}" "${navtest_root}" "${parity_root}" "${online_root}"; do
  if [[ -e "${path}" ]]; then
    echo "wave-3 post output already exists: ${path}" >&2
    exit 2
  fi
done
mkdir -p "${log_root}" "${report_root}" "${package_root}" "${navtest_root}" "${parity_root}" "${online_root}"

export PYTHONPATH="${repo_root}:${repo_root}/nuplan-devkit${PYTHONPATH:+:${PYTHONPATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export DRIVEVLA_EVAL_PRECISION=32

validate_audit() {
  local audit_dir="$1"
  local skill_validator="/root/.codex/skills/navsim-scorer-evaluation/scripts/validate_audit.sh"
  if [[ -x "${skill_validator}" ]]; then
    DRIVEVLA_REPO_ROOT="${repo_root}" DRIVEVLA_PYTHON="${python_bin}" \
      "${skill_validator}" "${audit_dir}"
  else
    "${python_bin}" "${repo_root}/local_stage2/validate_navtest_proposal_audit.py" \
      --audit-dir "${audit_dir}" \
      --expected-scenes 12146 \
      --expected-candidates 64
  fi
}

validate_audit "${navtest_scores}" \
  >"${log_root}/validate_matching_base_bank.log" 2>&1

promotion_args=()
for name in "${names[@]}"; do
  promotion_args+=(--residual-run "${run_root}/${name}")
  promotion_args+=(--residual-run "${calibrated_root}/${name}__calibrated")
done
"${python_bin}" "${repo_root}/local_stage2/build_m0_native_promotion_manifest.py" \
  "${promotion_args[@]}" \
  --minimum-ci-lower 0 \
  --output "${promotion_manifest}" \
  >"${log_root}/validation_promotion.log" 2>&1

mapfile -t promoted < <(
  "${python_bin}" - "${promotion_manifest}" <<'PY'
import json
import sys
for row in json.load(open(sys.argv[1])).get("promoted", []):
    print(f"{row['name']}\t{row['artifact']}")
PY
)

declare -A package_for_name
for line in "${promoted[@]}"; do
  IFS=$'\t' read -r name artifact <<< "${line}"
  package="${package_root}/${name}.pt"
  "${python_bin}" "${repo_root}/local_stage2/package_m0_native_private_scorer.py" \
    --ranker-artifact "${artifact}" \
    --base-checkpoint "${base_checkpoint}" \
    --shortlist-size 64 \
    --output "${package}" \
    >"${log_root}/package_${name}.log" 2>&1
  package_for_name["${name}"]="${package}"
done

eval_pids=()
eval_names=()
job_index=0
for line in "${promoted[@]}"; do
  IFS=$'\t' read -r name _artifact <<< "${line}"
  gpu="${gpu_ids[$((job_index % ${#gpu_ids[@]}))]}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    exec "${python_bin}" "${repo_root}/local_stage2/evaluate_m0_private_residual_navtest_cache.py" \
      --artifact "${package_for_name[${name}]}" \
      --feature-cache "${navtest_features}" \
      --candidate-matrix "${navtest_scores}/candidate_scores.npz" \
      --public-audit-dir "${navtest_scores}" \
      --output-dir "${navtest_root}/${name}" \
      --batch-size 64 \
      --bootstrap-replicates 10000 \
      --seed 20260932 \
      --device cuda
  ) >"${log_root}/navtest_${name}.log" 2>&1 &
  eval_pids+=("$!")
  eval_names+=("${name}")
  job_index=$((job_index + 1))
done

failure=0
for index in "${!eval_pids[@]}"; do
  name="${eval_names[${index}]}"
  if wait "${eval_pids[${index}]}" && \
    validate_audit "${navtest_root}/${name}" \
      >"${log_root}/validate_navtest_${name}.log" 2>&1; then
    echo "NO_VQA_WAVE3_NAVTEST_COMPLETE name=${name}"
  else
    echo "NO_VQA_WAVE3_NAVTEST_FAILED name=${name}" >&2
    failure=1
  fi
done
(( failure == 0 )) || exit 1

parity_pids=()
parity_names=()
job_index=0
for line in "${promoted[@]}"; do
  IFS=$'\t' read -r name _artifact <<< "${line}"
  gpu="${gpu_ids[$((job_index % ${#gpu_ids[@]}))]}"
  experiment="${name}_online_fp32_smoke"
  online_dir="${online_root}/ke_candidate_audit/${experiment}"
  (
    export MASTER_PORT="$((29900 + job_index))"
    export DRIVEVLA_REPO_ROOT="${repo_root}"
    export DRIVEVLA_PYTHON="${python_bin}"
    export DRIVEVLA_STAGE2_RUN_ROOT="${online_root}"
    export DRIVEVLA_SCORE_WORKERS=2
    export DRIVEVLA_EVAL_PRECISION=32
    bash "${repo_root}/local_stage2/run_navtest_proposal_audit.sh" \
      "${package_for_name[${name}]}" \
      local_stage2.m0_native_private_scorer_agent.M0NativePrivateScorerAgent \
      "${experiment}" \
      "${gpu}" \
      +proposal_audit_limit_scenes=4 \
      +proposal_audit_skip_cpu_scoring=true \
      "+proposal_audit_resolved_agent_config=${resolved_config}" \
      worker.threads_per_node=2
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${python_bin}" "${repo_root}/local_stage2/validate_m0_native_online_cache_parity.py" \
      --online-predictions "${online_dir}/proposal_predictions.pkl" \
      --public-feature-cache "${navtest_features}" \
      --artifact "${package_for_name[${name}]}" \
      --output "${parity_root}/${name}.json" \
      --device cuda \
      --atol 1e-6
  ) >"${log_root}/online_parity_${name}.log" 2>&1 &
  parity_pids+=("$!")
  parity_names+=("${name}")
  job_index=$((job_index + 1))
done

for index in "${!parity_pids[@]}"; do
  name="${parity_names[${index}]}"
  if wait "${parity_pids[${index}]}"; then
    echo "NO_VQA_WAVE3_ONLINE_PARITY_COMPLETE name=${name}"
  else
    echo "NO_VQA_WAVE3_ONLINE_PARITY_FAILED name=${name}" >&2
    failure=1
  fi
done
(( failure == 0 )) || exit 1

"${python_bin}" "${repo_root}/local_stage2/summarize_no_vqa_scene_token_campaign.py" \
  --promotion-manifest "${promotion_manifest}" \
  --baseline-audit "${navtest_scores}" \
  --navtest-root "${navtest_root}" \
  --parity-root "${parity_root}" \
  --output-json "${report_root}/CAMPAIGN_RESULTS.json" \
  --output-csv "${report_root}/CAMPAIGN_RESULTS.csv" \
  --output-md "${report_root}/CAMPAIGN_RESULTS.md" \
  >"${log_root}/campaign_summary.log" 2>&1

echo "NO_VQA_WAVE3_NAVTEST_CAMPAIGN_COMPLETE report=${report_root}/CAMPAIGN_RESULTS.md"
