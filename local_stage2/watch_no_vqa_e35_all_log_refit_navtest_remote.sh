#!/usr/bin/env bash

# Strictly evaluate every predeclared all-log refit on the immutable matching
# No-VQA FP32 Navtest proposal/candidate bank and verify online/cache parity.

set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
python_bin="${DRIVEVLA_PYTHON:-/mnt/project/DriveVLA-M0-env/bin/python}"
run_root="${NO_VQA_REFIT_RUN_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_all_log_refit_wave3_v1}"
log_root="${NO_VQA_REFIT_POST_LOG_ROOT:-/root/scorer_pdms93_logs/no_vqa_e35_all_log_refit_wave3_post_v1}"
report_root="${repo_root}/reports/m0_independent_scorer_representation/no_vqa_e35_all_log_refit_wave3_v1"
report_root="${NO_VQA_REFIT_REPORT_ROOT:-${report_root}}"
manifest="${report_root}/REFIT_PROMOTION.json"
package_root="${NO_VQA_REFIT_PACKAGE_ROOT:-/root/scorer_pdms93_artifacts/no_vqa_e35_all_log_refit_wave3_v1}"
navtest_root="${NO_VQA_REFIT_NAVTEST_ROOT:-/root/scorer_pdms93_navtest/no_vqa_e35_all_log_refit_wave3_v1}"
parity_root="${report_root}/online_cache_parity"
online_root="${NO_VQA_REFIT_ONLINE_ROOT:-/root/scorer_pdms93_online/no_vqa_e35_all_log_refit_wave3_v1}"
base_checkpoint="/mnt/project/DriveVLA-M0-no-vqa/runs/training/no_vqa_full_ft_seed0_e36/lightning_logs/version_0/checkpoints/best-epoch=35-step=232416.ckpt"
resolved_config="/mnt/project/DriveVLA-M0-no-vqa/runs/training/no_vqa_full_ft_seed0_e36/code/hydra/config.yaml"
navtest_features="/mnt/project/DriveVLA-M0-stage2/runs/ke_candidate_audit/no_vqa_e35_navtest_scorer_features_fp32_v1/proposal_predictions.pkl"
navtest_scores="/root/scorer_pdms93_cache/no_vqa_e35_navtest_candidate_scores_fp32_v1"
resume_post="${NO_VQA_REFIT_POST_RESUME:-0}"
gpu_csv="${NO_VQA_REFIT_GPU_IDS:-1,3,5}"
IFS=',' read -r -a gpus <<< "${gpu_csv}"
if [[ -n "${NO_VQA_REFIT_NAMES_CSV:-}" ]]; then
  IFS=',' read -r -a names <<< "${NO_VQA_REFIT_NAMES_CSV}"
else
  names=(
    candidate_only_top16_factor_safety5_all_logs
    combined_top16_hybrid_safety5_all_logs
    factorized_top16_cv_hybrid_safety5_all_logs
  )
fi
(( ${#names[@]} > 0 )) || { echo "NO_VQA_REFIT_NAMES_CSV is empty" >&2; exit 2; }
(( ${#gpus[@]} >= ${#names[@]} )) || {
  echo "NO_VQA_REFIT_GPU_IDS must provide at least one GPU per refit" >&2
  exit 2
}

until [[ -f "${run_root}/.refit_complete" ]]; do
  echo "M0_ALL_LOG_REFIT_NAVTEST waiting_for_refit utc=$(date -u +%FT%TZ)"
  sleep 30
done
if [[ "${resume_post}" != "0" && "${resume_post}" != "1" ]]; then
  echo "NO_VQA_REFIT_POST_RESUME must be 0 or 1" >&2
  exit 2
fi
if [[ "${resume_post}" == "0" ]]; then
  for path in "${log_root}" "${report_root}" "${package_root}" "${navtest_root}" "${online_root}"; do
    [[ ! -e "${path}" ]] || { echo "refusing existing refit post output: ${path}" >&2; exit 2; }
  done
fi
mkdir -p "${log_root}" "${report_root}" "${package_root}" "${navtest_root}" "${parity_root}" "${online_root}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export DRIVEVLA_EVAL_PRECISION=32
export PYTHONPATH="${repo_root}:${repo_root}/nuplan-devkit${PYTHONPATH:+:${PYTHONPATH}}"

validate_audit() {
  local audit_dir="$1"
  local skill_validator=/root/.codex/skills/navsim-scorer-evaluation/scripts/validate_audit.sh
  if [[ -x "${skill_validator}" ]]; then
    DRIVEVLA_REPO_ROOT="${repo_root}" DRIVEVLA_PYTHON="${python_bin}" \
      "${skill_validator}" "${audit_dir}"
  else
    "${python_bin}" "${repo_root}/local_stage2/validate_navtest_proposal_audit.py" \
      --audit-dir "${audit_dir}" --expected-scenes 12146 --expected-candidates 64
  fi
}

manifest_args=()
for name in "${names[@]}"; do
  manifest_args+=(--refit-residual-run "${run_root}/${name}")
done
if [[ ! -f "${manifest}" ]]; then
  "${python_bin}" "${repo_root}/local_stage2/build_m0_native_promotion_manifest.py" \
    "${manifest_args[@]}" --minimum-ci-lower 0 --output "${manifest}" \
    >"${log_root}/build_refit_manifest.log" 2>&1
elif [[ "${resume_post}" != "1" ]]; then
  echo "refusing existing refit manifest: ${manifest}" >&2
  exit 2
fi

mapfile -t promoted < <(
  "${python_bin}" - "${manifest}" <<'PY'
import json, sys
for row in json.load(open(sys.argv[1]))["promoted"]:
    print(f"{row['name']}\t{row['artifact']}")
PY
)
[[ "${#promoted[@]}" -eq "${#names[@]}" ]] || { echo "refit manifest coverage failure" >&2; exit 2; }

declare -A packages
for line in "${promoted[@]}"; do
  IFS=$'\t' read -r name artifact <<< "${line}"
  package="${package_root}/${name}.pt"
  if [[ ! -f "${package}" ]]; then
    "${python_bin}" "${repo_root}/local_stage2/package_m0_native_private_scorer.py" \
      --ranker-artifact "${artifact}" --base-checkpoint "${base_checkpoint}" \
      --shortlist-size 64 --output "${package}" \
      >"${log_root}/package_${name}.log" 2>&1
  elif [[ "${resume_post}" != "1" ]]; then
    echo "refusing existing refit package: ${package}" >&2
    exit 2
  fi
  packages["${name}"]="${package}"
done

validate_audit "${navtest_scores}" >"${log_root}/validate_matching_bank.log" 2>&1

pids=()
job_names=()
for index in "${!promoted[@]}"; do
  IFS=$'\t' read -r name _artifact <<< "${promoted[${index}]}"
  if [[ -f "${navtest_root}/${name}/summary.json" ]]; then
    validate_audit "${navtest_root}/${name}" \
      >"${log_root}/validate_${name}.log" 2>&1
    echo "M0_ALL_LOG_REFIT_NAVTEST_REUSED name=${name}"
    continue
  fi
  [[ ! -e "${navtest_root}/${name}" ]] || {
    echo "refusing incomplete refit Navtest output: ${navtest_root}/${name}" >&2
    exit 2
  }
  (
    export CUDA_VISIBLE_DEVICES="${gpus[${index}]}"
    exec "${python_bin}" "${repo_root}/local_stage2/evaluate_m0_private_residual_navtest_cache.py" \
      --artifact "${packages[${name}]}" \
      --feature-cache "${navtest_features}" \
      --candidate-matrix "${navtest_scores}/candidate_scores.npz" \
      --public-audit-dir "${navtest_scores}" \
      --output-dir "${navtest_root}/${name}" \
      --batch-size 64 --bootstrap-replicates 10000 --seed 20260942 --device cuda
  ) >"${log_root}/navtest_${name}.log" 2>&1 &
  pids+=("$!")
  job_names+=("${name}")
done
failure=0
for index in "${!pids[@]}"; do
  name="${job_names[${index}]}"
  if wait "${pids[${index}]}" && \
    validate_audit "${navtest_root}/${name}" \
      >"${log_root}/validate_${name}.log" 2>&1; then
    echo "M0_ALL_LOG_REFIT_NAVTEST_COMPLETE name=${name}"
  else
    echo "M0_ALL_LOG_REFIT_NAVTEST_FAILED name=${name}" >&2
    failure=1
  fi
done
(( failure == 0 )) || exit 1

pids=()
job_names=()
for index in "${!promoted[@]}"; do
  IFS=$'\t' read -r name _artifact <<< "${promoted[${index}]}"
  if [[ -f "${parity_root}/${name}.json" ]]; then
    echo "M0_ALL_LOG_REFIT_PARITY_REUSED name=${name}"
    continue
  fi
  experiment="${name}_online_fp32_smoke"
  (
    export MASTER_PORT="$((30000 + index))"
    export DRIVEVLA_REPO_ROOT="${repo_root}"
    export DRIVEVLA_PYTHON="${python_bin}"
    export DRIVEVLA_STAGE2_RUN_ROOT="${online_root}"
    export DRIVEVLA_SCORE_WORKERS=2
    export DRIVEVLA_EVAL_PRECISION=32
    bash "${repo_root}/local_stage2/run_navtest_proposal_audit.sh" \
      "${packages[${name}]}" \
      local_stage2.m0_native_private_scorer_agent.M0NativePrivateScorerAgent \
      "${experiment}" "${gpus[${index}]}" \
      +proposal_audit_limit_scenes=4 +proposal_audit_skip_cpu_scoring=true \
      "+proposal_audit_resolved_agent_config=${resolved_config}" worker.threads_per_node=2
    export CUDA_VISIBLE_DEVICES="${gpus[${index}]}"
    "${python_bin}" "${repo_root}/local_stage2/validate_m0_native_online_cache_parity.py" \
      --online-predictions "${online_root}/ke_candidate_audit/${experiment}/proposal_predictions.pkl" \
      --public-feature-cache "${navtest_features}" \
      --artifact "${packages[${name}]}" \
      --output "${parity_root}/${name}.json" --device cuda --atol 1e-6
  ) >"${log_root}/parity_${name}.log" 2>&1 &
  pids+=("$!")
  job_names+=("${name}")
done
for index in "${!pids[@]}"; do
  wait "${pids[${index}]}" || failure=1
done
(( failure == 0 )) || exit 1

"${python_bin}" "${repo_root}/local_stage2/summarize_no_vqa_scene_token_campaign.py" \
  --promotion-manifest "${manifest}" --baseline-audit "${navtest_scores}" \
  --navtest-root "${navtest_root}" --parity-root "${parity_root}" \
  --output-json "${report_root}/CAMPAIGN_RESULTS.json" \
  --output-csv "${report_root}/CAMPAIGN_RESULTS.csv" \
  --output-md "${report_root}/CAMPAIGN_RESULTS.md" \
  >"${log_root}/campaign_summary.log" 2>&1
echo "M0_ALL_LOG_REFIT_NAVTEST_CAMPAIGN_COMPLETE report=${report_root}/CAMPAIGN_RESULTS.md"
