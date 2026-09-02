#!/usr/bin/env bash

# Apply the Wave-12 common-policy/refit/Navtest protocol to the predeclared
# dense current-observation representation.

set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
dense_train_root="${NO_VQA_DENSE_TRAIN_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_multiview_trainval_pool4_tiles4_v1_8shard}"
dense_navtest_root="${NO_VQA_DENSE_NAVTEST_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_multiview_navtest_pool4_tiles4_v1_8shard}"

export REPO_ROOT="${repo_root}"
export NO_VQA_WAVE12_RUN_ROOT="${NO_VQA_WAVE13_RUN_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_dense_risk_cv_wave13_v1}"
export NO_VQA_WAVE12_FOLD_ROOT="${NO_VQA_WAVE13_FOLD_ROOT:-${repo_root}/reports/m0_independent_scorer_representation/no_vqa_e35_risk_cv_wave12_v1/folds}"
export NO_VQA_WAVE12_REPORT_ROOT="${NO_VQA_WAVE13_REPORT_ROOT:-${repo_root}/reports/m0_independent_scorer_representation/no_vqa_e35_dense_risk_cv_wave13_v1}"
export NO_VQA_WAVE12_POST_LOG_ROOT="${NO_VQA_WAVE13_POST_LOG_ROOT:-/root/scorer_pdms93_logs/no_vqa_e35_dense_risk_cv_wave13_post_v1}"
export NO_VQA_WAVE12_SELECTION_ARTIFACT="${NO_VQA_WAVE13_SELECTION_ARTIFACT:-/root/scorer_pdms93_artifacts/no_vqa_e35_dense_risk_cv_wave13_v1/CV_SELECTION_ARTIFACT.pt}"
export NO_VQA_WAVE12_FIXED_POLICY_ARTIFACT="${NO_VQA_WAVE13_FIXED_POLICY_ARTIFACT:-/root/scorer_pdms93_artifacts/no_vqa_e35_dense_risk_cv_wave13_v1/FIXED_POLICY_DIAGNOSTIC_ARTIFACT.pt}"
export NO_VQA_WAVE12_SWEEP_ROOT="${NO_VQA_WAVE13_SWEEP_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_dense_risk_cv_wave13_v1/common_policy_sweeps}"
export NO_VQA_WAVE12_REFIT_ROOT="${NO_VQA_WAVE13_REFIT_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_dense_risk_cv_wave13_all_log_refit_v1}"
export NO_VQA_WAVE12_REFIT_CAMPAIGN="${NO_VQA_WAVE13_REFIT_CAMPAIGN:-no_vqa_e35_dense_risk_cv_wave13_all_log_refit_v1}"
export NO_VQA_WAVE12_PRIVATE_TRAIN_ROOT="${dense_train_root}"
export NO_VQA_MULTIVIEW_TEST_ROOT="${dense_navtest_root}"

exec bash "${repo_root}/local_stage2/watch_no_vqa_e35_risk_cv_wave12_remote.sh"
