#!/usr/bin/env bash

# Evaluate every predeclared Wave-5 all-log refit on the immutable matching
# FP32 Navtest bank, then verify the true online M0 agent against cache replay.

set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
export NO_VQA_REFIT_RUN_ROOT="${NO_VQA_REFIT_RUN_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_all_log_refit_wave5_reference_v1}"
export NO_VQA_REFIT_POST_LOG_ROOT="${NO_VQA_REFIT_POST_LOG_ROOT:-/root/scorer_pdms93_logs/no_vqa_e35_all_log_refit_wave5_reference_post_v1}"
export NO_VQA_REFIT_PACKAGE_ROOT="${NO_VQA_REFIT_PACKAGE_ROOT:-/root/scorer_pdms93_artifacts/no_vqa_e35_all_log_refit_wave5_reference_v1}"
export NO_VQA_REFIT_NAVTEST_ROOT="${NO_VQA_REFIT_NAVTEST_ROOT:-/root/scorer_pdms93_navtest/no_vqa_e35_all_log_refit_wave5_reference_v1}"
export NO_VQA_REFIT_ONLINE_ROOT="${NO_VQA_REFIT_ONLINE_ROOT:-/root/scorer_pdms93_online/no_vqa_e35_all_log_refit_wave5_reference_v1}"
export NO_VQA_REFIT_REPORT_ROOT="${NO_VQA_REFIT_REPORT_ROOT:-${repo_root}/reports/m0_independent_scorer_representation/no_vqa_e35_all_log_refit_wave5_reference_v1}"
export NO_VQA_REFIT_NAMES_CSV="combined_top8_reference_q50_strict_actor_all_logs,combined_top16_reference_q50_strict_actor_all_logs,combined_top16_reference_q10_balanced_actor_all_logs"
export NO_VQA_REFIT_GPU_IDS="1,2,3"

exec bash "${repo_root}/local_stage2/watch_no_vqa_e35_all_log_refit_navtest_remote.sh"
