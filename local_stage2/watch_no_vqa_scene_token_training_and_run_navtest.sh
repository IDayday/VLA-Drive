#!/usr/bin/env bash

# Promote only held-out-log significant No-VQA scene-token scorers, evaluate
# every promoted artifact on the complete matching FP32 Navtest bank, and run
# real-agent/cache parity on four identical scenes per artifact.

set -uo pipefail

repo_root="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
python_bin="${DRIVEVLA_PYTHON:-/mnt/project/DriveVLA-M0-env/bin/python}"
run_root="${NO_VQA_SCORER_RUN_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_scene_token_v1}"
calibrated_root="${NO_VQA_CALIBRATED_RUN_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_scene_token_v1_calibrated}"
log_root="${NO_VQA_POST_LOG_ROOT:-/root/scorer_pdms93_logs/no_vqa_e35_scene_token_v1_post}"
report_root="${NO_VQA_REPORT_ROOT:-${repo_root}/reports/m0_independent_scorer_representation/no_vqa_e35_scene_token_v1}"
promotion_manifest="${report_root}/VALIDATION_PROMOTION.json"
package_root="${NO_VQA_PACKAGE_ROOT:-/root/scorer_pdms93_artifacts/no_vqa_e35_scene_token_v1}"
navtest_root="${NO_VQA_NAVTEST_RESULT_ROOT:-/root/scorer_pdms93_navtest/no_vqa_e35_scene_token_v1}"
parity_root="${report_root}/online_cache_parity"
online_root="${NO_VQA_ONLINE_ROOT:-/root/scorer_pdms93_online/no_vqa_e35_scene_token_v1}"
base_checkpoint="${NO_VQA_CHECKPOINT:-/mnt/project/DriveVLA-M0-no-vqa/runs/training/no_vqa_full_ft_seed0_e36/lightning_logs/version_0/checkpoints/best-epoch=35-step=232416.ckpt}"
resolved_config="${NO_VQA_RESOLVED_CONFIG:-/mnt/project/DriveVLA-M0-no-vqa/runs/training/no_vqa_full_ft_seed0_e36/code/hydra/config.yaml}"
navtest_features="${NO_VQA_NAVTEST_FEATURES:-/mnt/project/DriveVLA-M0-stage2/runs/ke_candidate_audit/no_vqa_e35_navtest_scorer_features_fp32_v1/proposal_predictions.pkl}"
navtest_scores="${NO_VQA_NAVTEST_SCORES:-/root/scorer_pdms93_cache/no_vqa_e35_navtest_candidate_scores_fp32_v1}"
private_observation_root="${NO_VQA_PRIVATE_OBSERVATION_ROOT:-}"
wait_seconds="${NO_VQA_POST_POLL_SECONDS:-30}"
skip_calibration="${NO_VQA_SKIP_CALIBRATION:-0}"
reuse_promotion_manifest="${NO_VQA_REUSE_PROMOTION_MANIFEST:-0}"
include_calibrated="${NO_VQA_INCLUDE_CALIBRATED:-1}"
expected_epochs="${NO_VQA_EXPECTED_EPOCHS:-8}"
gpu_csv="${NO_VQA_POST_GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a gpu_ids <<< "${gpu_csv}"
(( ${#gpu_ids[@]} > 0 )) || { echo "NO_VQA_POST_GPU_IDS is empty" >&2; exit 2; }

if [[ -n "${NO_VQA_SCORER_NAMES_CSV:-}" ]]; then
  IFS=',' read -r -a names <<< "${NO_VQA_SCORER_NAMES_CSV}"
else
  names=(
    primary_hybrid_actor050_seed2
    control_hybrid_no_actor_seed2
    factor_actor050_seed2
    direct_actor050_seed2
    primary_hybrid_actor050_seed11
    primary_hybrid_actor050_seed23
    hybrid_actor050_deep_seed2
    hybrid_actor050_future025_seed2
  )
fi
(( ${#names[@]} > 0 )) || { echo "No scorer names configured" >&2; exit 2; }
if [[ "${include_calibrated}" != "0" && "${include_calibrated}" != "1" ]]; then
  echo "NO_VQA_INCLUDE_CALIBRATED must be 0 or 1" >&2
  exit 2
fi
if [[ -n "${private_observation_root}" && ! -d "${private_observation_root}" ]]; then
  echo "Missing private-observation cache: ${private_observation_root}" >&2
  exit 2
fi
[[ "${expected_epochs}" =~ ^[1-9][0-9]*$ ]] || {
  echo "NO_VQA_EXPECTED_EPOCHS must be a positive integer" >&2
  exit 2
}

mkdir -p "${log_root}" "${report_root}" "${package_root}" "${navtest_root}" "${parity_root}" "${online_root}" "${calibrated_root}"

training_ready() {
  [[ -f "${navtest_scores}/summary.json" ]] || return 1
  [[ -f "${navtest_scores}/candidate_scores.npz" ]] || return 1
  for name in "${names[@]}"; do
    summary="${run_root}/${name}/training_summary.json"
    [[ -f "${summary}" ]] || return 1
    "${python_bin}" - "${summary}" "${expected_epochs}" <<'PY' >/dev/null 2>&1
import json
import sys

payload = json.load(open(sys.argv[1]))
raise SystemExit(0 if len(payload.get("history", [])) == int(sys.argv[2]) else 1)
PY
  done
}

while ! training_ready; do
  echo "NO_VQA_NAVTEST_WATCH waiting_for_training_and_candidate_matrix utc=$(date -u +%FT%TZ)"
  sleep "${wait_seconds}"
done

export PYTHONPATH="${repo_root}:${repo_root}/nuplan-devkit${PYTHONPATH:+:${PYTHONPATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export DRIVEVLA_EVAL_PRECISION=32

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

# Freeze each learned ranker, choose only deployment-time residual/gating
# parameters on one balanced half of the held-out physical logs, and report
# promotion on the disjoint half.  Raw and calibrated artifacts are both kept;
# Navtest is not read during calibration.
calibrated_names=()
if [[ "${include_calibrated}" == "1" ]]; then
  for name in "${names[@]}"; do
    calibrated_name="${name}__calibrated"
    calibrated_names+=("${calibrated_name}")
  done
fi
if [[ "${include_calibrated}" == "1" && "${skip_calibration}" != "1" ]]; then
calibration_pids=()
for index in "${!names[@]}"; do
  gpu="${gpu_ids[$((index % ${#gpu_ids[@]}))]}"
  name="${names[${index}]}"
  calibrated_name="${calibrated_names[${index}]}"
  output_dir="${calibrated_root}/${calibrated_name}"
  if [[ -e "${output_dir}" ]]; then
    echo "NO_VQA_NAVTEST_WATCH refusing_existing_calibration=${output_dir}" >&2
    exit 2
  fi
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    exec "${python_bin}" "${repo_root}/local_stage2/calibrate_m0_private_residual_policy.py" \
      --source no_vqa_e35 \
        /root/scorer_pdms93_cache/no_vqa_e35_features_full_v1 \
        /root/scorer_pdms93_cache/no_vqa_e35_labels_full_v1 \
      --split-manifest "${repo_root}/reports/scorer_pdms93/OFFICIAL_SCORER_SPLIT.json" \
      --selection-source no_vqa_e35 \
      --artifact "${run_root}/${name}/best_m0_private_residual_scorer.pt" \
      --output-dir "${output_dir}" \
      --seed "$((20260902 + gpu))" \
      --eval-batch-size 64 \
      --bootstrap-replicates 1000 \
      --device cuda
  ) >"${log_root}/calibrate_${name}.log" 2>&1 &
  calibration_pids+=("$!")
  echo "NO_VQA_SCORER_CALIBRATION_STARTED gpu=${gpu} name=${name} pid=$!"
done

calibration_failure=0
for index in "${!calibration_pids[@]}"; do
  if wait "${calibration_pids[${index}]}"; then
    echo "NO_VQA_SCORER_CALIBRATION_COMPLETE name=${calibrated_names[${index}]}"
  else
    status=$?
    echo "NO_VQA_SCORER_CALIBRATION_FAILED name=${calibrated_names[${index}]} status=${status}" >&2
    calibration_failure=1
  fi
done
if (( calibration_failure != 0 )); then
  exit 1
fi
elif [[ "${include_calibrated}" == "1" ]]; then
  for calibrated_name in "${calibrated_names[@]}"; do
    summary="${calibrated_root}/${calibrated_name}/training_summary.json"
    [[ -f "${summary}" ]] || { echo "missing reused calibration: ${summary}" >&2; exit 2; }
  done
  echo "NO_VQA_SCORER_CALIBRATION_REUSED root=${calibrated_root}"
else
  echo "NO_VQA_SCORER_CALIBRATION_DISABLED"
fi

promotion_args=()
for name in "${names[@]}"; do
  promotion_args+=(--residual-run "${run_root}/${name}")
done
for calibrated_name in "${calibrated_names[@]}"; do
  promotion_args+=(--residual-run "${calibrated_root}/${calibrated_name}")
done
if [[ "${reuse_promotion_manifest}" == "1" ]]; then
  [[ -f "${promotion_manifest}" ]] || { echo "missing reused manifest: ${promotion_manifest}" >&2; exit 2; }
  echo "NO_VQA_SCORER_PROMOTION_REUSED path=${promotion_manifest}"
else
  if [[ -e "${promotion_manifest}" ]]; then
    echo "NO_VQA_NAVTEST_WATCH refusing_existing_manifest=${promotion_manifest}" >&2
    exit 2
  fi
  "${python_bin}" "${repo_root}/local_stage2/build_m0_native_promotion_manifest.py" \
    "${promotion_args[@]}" \
    --minimum-ci-lower 0 \
    --output "${promotion_manifest}" \
    >"${log_root}/validation_promotion.log" 2>&1
fi

mapfile -t promoted < <(
  "${python_bin}" - "${promotion_manifest}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
for row in payload.get("promoted", []):
    print(f"{row['name']}\t{row['artifact']}")
PY
)

if (( ${#promoted[@]} == 0 )); then
  echo "NO_VQA_NAVTEST_WATCH no_validation_promoted_artifacts"
  exit 0
fi

declare -A package_for_name
private_args=()
if [[ -n "${private_observation_root}" ]]; then
  private_args+=(--private-observation-root "${private_observation_root}")
fi
for line in "${promoted[@]}"; do
  IFS=$'\t' read -r name artifact <<< "${line}"
  package="${package_root}/${name}.pt"
  if [[ -e "${package}" ]]; then
    echo "NO_VQA_NAVTEST_WATCH refusing_existing_package=${package}" >&2
    exit 2
  fi
  "${python_bin}" "${repo_root}/local_stage2/package_m0_native_private_scorer.py" \
    --ranker-artifact "${artifact}" \
    --base-checkpoint "${base_checkpoint}" \
    "${private_args[@]}" \
    --shortlist-size 64 \
    --output "${package}" \
    >"${log_root}/package_${name}.log" 2>&1
  package_for_name["${name}"]="${package}"
  echo "NO_VQA_SCORER_PACKAGED name=${name} path=${package}"
done

validate_audit "${navtest_scores}" \
  >"${log_root}/validate_matching_base_bank.log" 2>&1

eval_pids=()
eval_names=()
job_index=0
for line in "${promoted[@]}"; do
  IFS=$'\t' read -r name _artifact <<< "${line}"
  gpu="${gpu_ids[$((job_index % ${#gpu_ids[@]}))]}"
  output="${navtest_root}/${name}"
  if [[ -e "${output}" ]]; then
    echo "NO_VQA_NAVTEST_WATCH refusing_existing_eval=${output}" >&2
    exit 2
  fi
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    exec "${python_bin}" "${repo_root}/local_stage2/evaluate_m0_private_residual_navtest_cache.py" \
      --artifact "${package_for_name[${name}]}" \
      --feature-cache "${navtest_features}" \
      "${private_args[@]}" \
      --candidate-matrix "${navtest_scores}/candidate_scores.npz" \
      --public-audit-dir "${navtest_scores}" \
      --output-dir "${output}" \
      --batch-size 64 \
      --bootstrap-replicates 10000 \
      --seed 20260902 \
      --device cuda
  ) >"${log_root}/navtest_${name}.log" 2>&1 &
  eval_pids+=("$!")
  eval_names+=("${name}")
  echo "NO_VQA_SCORER_NAVTEST_STARTED gpu=${gpu} name=${name} pid=$!"
  job_index=$((job_index + 1))
done

failure=0
for index in "${!eval_pids[@]}"; do
  name="${eval_names[${index}]}"
  if wait "${eval_pids[${index}]}"; then
    if validate_audit "${navtest_root}/${name}" \
      >"${log_root}/validate_navtest_${name}.log" 2>&1; then
      echo "NO_VQA_SCORER_NAVTEST_COMPLETE name=${name}"
    else
      echo "NO_VQA_SCORER_NAVTEST_VALIDATION_FAILED name=${name}" >&2
      failure=1
    fi
  else
    status=$?
    echo "NO_VQA_SCORER_NAVTEST_FAILED name=${name} status=${status}" >&2
    failure=1
  fi
done
if (( failure != 0 )); then
  exit 1
fi

# Verify the actual custom Agent against cached scorer inference on the same
# CUDA device. Use a distinct DDP port per concurrent one-GPU smoke.
parity_pids=()
parity_names=()
job_index=0
for line in "${promoted[@]}"; do
  IFS=$'\t' read -r name _artifact <<< "${line}"
  gpu="${gpu_ids[$((job_index % ${#gpu_ids[@]}))]}"
  experiment="${name}_online_fp32_smoke"
  online_dir="${online_root}/ke_candidate_audit/${experiment}"
  (
    export MASTER_PORT="$((29600 + job_index))"
    export CUDA_VISIBLE_DEVICES="${gpu}"
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
      "${private_args[@]}" \
      --output "${parity_root}/${name}.json" \
      --device cuda \
      --atol 1e-6
  ) >"${log_root}/online_parity_${name}.log" 2>&1 &
  parity_pids+=("$!")
  parity_names+=("${name}")
  echo "NO_VQA_SCORER_ONLINE_PARITY_STARTED gpu=${gpu} name=${name} pid=$!"
  job_index=$((job_index + 1))
done

for index in "${!parity_pids[@]}"; do
  name="${parity_names[${index}]}"
  if wait "${parity_pids[${index}]}"; then
    echo "NO_VQA_SCORER_ONLINE_PARITY_COMPLETE name=${name}"
  else
    status=$?
    echo "NO_VQA_SCORER_ONLINE_PARITY_FAILED name=${name} status=${status}" >&2
    failure=1
  fi
done
if (( failure != 0 )); then
  exit 1
fi

"${python_bin}" "${repo_root}/local_stage2/summarize_no_vqa_scene_token_campaign.py" \
  --promotion-manifest "${promotion_manifest}" \
  --baseline-audit "${navtest_scores}" \
  --navtest-root "${navtest_root}" \
  --parity-root "${parity_root}" \
  --output-json "${report_root}/CAMPAIGN_RESULTS.json" \
  --output-csv "${report_root}/CAMPAIGN_RESULTS.csv" \
  --output-md "${report_root}/CAMPAIGN_RESULTS.md" \
  >"${log_root}/campaign_summary.log" 2>&1

echo "NO_VQA_SCENE_TOKEN_NAVTEST_CAMPAIGN_COMPLETE report=${report_root}/CAMPAIGN_RESULTS.md"
