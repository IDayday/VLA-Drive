#!/usr/bin/env bash

# Strict held-out promotion, full FP32 Navtest, and online/cache parity for the
# single predeclared Wave-9 Top-64 path-attention model.

set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
private_navtest_root="${NO_VQA_MULTIVIEW_TEST_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_multiview_navtest_pool2_tiles4_v1_8shard}"
[[ -f "${private_navtest_root}/.complete" ]] || {
  echo "incomplete Wave-9 Navtest cache: ${private_navtest_root}" >&2
  exit 2
}

export NO_VQA_SCORER_RUN_ROOT="${NO_VQA_WAVE9_RUN_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_point_top64_wave9_v1}"
export NO_VQA_CALIBRATED_RUN_ROOT="${NO_VQA_WAVE9_CALIBRATED_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_point_top64_wave9_calibrated_v1}"
export NO_VQA_POST_LOG_ROOT="${NO_VQA_WAVE9_POST_LOG_ROOT:-/root/scorer_pdms93_logs/no_vqa_e35_point_top64_wave9_post_v1}"
export NO_VQA_PACKAGE_ROOT="${NO_VQA_WAVE9_PACKAGE_ROOT:-/root/scorer_pdms93_artifacts/no_vqa_e35_point_top64_wave9_v1}"
export NO_VQA_NAVTEST_RESULT_ROOT="${NO_VQA_WAVE9_NAVTEST_ROOT:-/root/scorer_pdms93_navtest/no_vqa_e35_point_top64_wave9_v1}"
export NO_VQA_ONLINE_ROOT="${NO_VQA_WAVE9_ONLINE_ROOT:-/root/scorer_pdms93_online/no_vqa_e35_point_top64_wave9_v1}"
export NO_VQA_REPORT_ROOT="${NO_VQA_WAVE9_REPORT_ROOT:-${repo_root}/reports/m0_independent_scorer_representation/no_vqa_e35_point_top64_wave9_v1}"
export NO_VQA_PRIVATE_OBSERVATION_ROOT="${private_navtest_root}"
export NO_VQA_INCLUDE_CALIBRATED=0
export NO_VQA_EXPECTED_EPOCHS=8
export NO_VQA_POST_GPU_IDS="${NO_VQA_WAVE9_POST_GPU_IDS:-0}"
export NO_VQA_SCORER_NAMES_CSV="rawpointcombined_top64_reference_q50_strict_noactor_seed2"

exec bash "${repo_root}/local_stage2/watch_no_vqa_scene_token_training_and_run_navtest.sh"
