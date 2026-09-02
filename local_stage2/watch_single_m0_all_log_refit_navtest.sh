#!/usr/bin/env bash

# Usage: watch_single_m0_all_log_refit_navtest.sh RUN_DIR CAMPAIGN_NAME GPU
# Wait for one provenance-locked all-log refit, then run the complete FP32
# Navtest cache audit and four-scene online/cache parity.

set -euo pipefail

[[ "$#" -eq 3 ]] || { echo "usage: $0 RUN_DIR CAMPAIGN_NAME GPU" >&2; exit 2; }
run_dir="$1"
campaign="$2"
gpu="$3"
repo_root="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
python_bin="${DRIVEVLA_PYTHON:-/mnt/project/DriveVLA-M0-env/bin/python}"
report_root="${repo_root}/reports/m0_independent_scorer_representation/${campaign}"
log_root="/root/scorer_pdms93_logs/${campaign}_post"
package_root="/root/scorer_pdms93_artifacts/${campaign}"
navtest_root="/root/scorer_pdms93_navtest/${campaign}"
online_root="/root/scorer_pdms93_online/${campaign}"
parity_root="${report_root}/online_cache_parity"
manifest="${report_root}/REFIT_PROMOTION.json"
base_checkpoint="/mnt/project/DriveVLA-M0-no-vqa/runs/training/no_vqa_full_ft_seed0_e36/lightning_logs/version_0/checkpoints/best-epoch=35-step=232416.ckpt"
resolved_config="/mnt/project/DriveVLA-M0-no-vqa/runs/training/no_vqa_full_ft_seed0_e36/code/hydra/config.yaml"
navtest_features="/mnt/project/DriveVLA-M0-stage2/runs/ke_candidate_audit/no_vqa_e35_navtest_scorer_features_fp32_v1/proposal_predictions.pkl"
navtest_scores="/root/scorer_pdms93_cache/no_vqa_e35_navtest_candidate_scores_fp32_v1"
resume_post="${M0_SINGLE_REFIT_POST_RESUME:-0}"

until [[ -f "${run_dir}/refit_m0_private_residual_scorer.pt" ]]; do
  echo "M0_SINGLE_REFIT_NAVTEST waiting campaign=${campaign} utc=$(date -u +%FT%TZ)"
  sleep 30
done
if [[ "${resume_post}" != "0" && "${resume_post}" != "1" ]]; then
  echo "M0_SINGLE_REFIT_POST_RESUME must be 0 or 1" >&2
  exit 2
fi
if [[ "${resume_post}" == "0" ]]; then
  for path in "${report_root}" "${log_root}" "${package_root}" "${navtest_root}" "${online_root}"; do
    [[ ! -e "${path}" ]] || { echo "refusing existing output: ${path}" >&2; exit 2; }
  done
fi
mkdir -p "${report_root}" "${log_root}" "${package_root}" "${navtest_root}" "${online_root}" "${parity_root}"
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

if [[ ! -f "${manifest}" ]]; then
  "${python_bin}" "${repo_root}/local_stage2/build_m0_native_promotion_manifest.py" \
    --refit-residual-run "${run_dir}" --minimum-ci-lower 0 --output "${manifest}" \
    >"${log_root}/manifest.log" 2>&1
elif [[ "${resume_post}" != "1" ]]; then
  echo "refusing existing promotion manifest: ${manifest}" >&2
  exit 2
fi
read -r name artifact < <(
  "${python_bin}" - "${manifest}" <<'PY'
import json, sys
rows=json.load(open(sys.argv[1]))["promoted"]
assert len(rows)==1
print(rows[0]["name"], rows[0]["artifact"])
PY
)
package="${package_root}/${name}.pt"
if [[ ! -f "${package}" ]]; then
  "${python_bin}" "${repo_root}/local_stage2/package_m0_native_private_scorer.py" \
    --ranker-artifact "${artifact}" --base-checkpoint "${base_checkpoint}" \
    --shortlist-size 64 --output "${package}" >"${log_root}/package.log" 2>&1
elif [[ "${resume_post}" != "1" ]]; then
  echo "refusing existing package: ${package}" >&2
  exit 2
fi
validate_audit "${navtest_scores}" >"${log_root}/validate_matching_bank.log" 2>&1
export CUDA_VISIBLE_DEVICES="${gpu}"
if [[ -f "${navtest_root}/${name}/summary.json" ]]; then
  validate_audit "${navtest_root}/${name}" >"${log_root}/validate_navtest.log" 2>&1
elif [[ ! -e "${navtest_root}/${name}" ]]; then
  "${python_bin}" "${repo_root}/local_stage2/evaluate_m0_private_residual_navtest_cache.py" \
    --artifact "${package}" --feature-cache "${navtest_features}" \
    --candidate-matrix "${navtest_scores}/candidate_scores.npz" \
    --public-audit-dir "${navtest_scores}" --output-dir "${navtest_root}/${name}" \
    --batch-size 64 --bootstrap-replicates 10000 --seed 20260952 --device cuda \
    >"${log_root}/navtest.log" 2>&1
  validate_audit "${navtest_root}/${name}" >"${log_root}/validate_navtest.log" 2>&1
else
  echo "refusing incomplete Navtest output: ${navtest_root}/${name}" >&2
  exit 2
fi

experiment="${name}_online_fp32_smoke"
export MASTER_PORT=30100
export DRIVEVLA_REPO_ROOT="${repo_root}"
export DRIVEVLA_PYTHON="${python_bin}"
export DRIVEVLA_STAGE2_RUN_ROOT="${online_root}"
export DRIVEVLA_SCORE_WORKERS=2
if [[ ! -f "${parity_root}/${name}.json" ]]; then
  if [[ ! -f "${online_root}/ke_candidate_audit/${experiment}/proposal_predictions.pkl" ]]; then
    bash "${repo_root}/local_stage2/run_navtest_proposal_audit.sh" \
      "${package}" local_stage2.m0_native_private_scorer_agent.M0NativePrivateScorerAgent \
      "${experiment}" "${gpu}" +proposal_audit_limit_scenes=4 \
      +proposal_audit_skip_cpu_scoring=true \
      "+proposal_audit_resolved_agent_config=${resolved_config}" worker.threads_per_node=2 \
      >"${log_root}/online.log" 2>&1
  fi
  "${python_bin}" "${repo_root}/local_stage2/validate_m0_native_online_cache_parity.py" \
    --online-predictions "${online_root}/ke_candidate_audit/${experiment}/proposal_predictions.pkl" \
    --public-feature-cache "${navtest_features}" --artifact "${package}" \
    --output "${parity_root}/${name}.json" --device cuda --atol 1e-6 \
    >"${log_root}/parity.log" 2>&1
fi

"${python_bin}" "${repo_root}/local_stage2/summarize_no_vqa_scene_token_campaign.py" \
  --promotion-manifest "${manifest}" --baseline-audit "${navtest_scores}" \
  --navtest-root "${navtest_root}" --parity-root "${parity_root}" \
  --output-json "${report_root}/CAMPAIGN_RESULTS.json" \
  --output-csv "${report_root}/CAMPAIGN_RESULTS.csv" \
  --output-md "${report_root}/CAMPAIGN_RESULTS.md" >"${log_root}/summary.log" 2>&1
echo "M0_SINGLE_REFIT_NAVTEST_COMPLETE report=${report_root}/CAMPAIGN_RESULTS.md"
