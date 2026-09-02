#!/usr/bin/env bash

# Promote and strictly evaluate the predeclared No-VQA conservative-reference
# wave.  Thresholds are fixed before Navtest; no post-hoc calibration is run.

set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
export NO_VQA_SCORER_RUN_ROOT="${NO_VQA_WAVE5_RUN_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_scene_token_wave5_reference_v1}"
export NO_VQA_CALIBRATED_RUN_ROOT="${NO_VQA_WAVE5_CALIBRATED_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_scene_token_wave5_reference_calibrated_v1}"
export NO_VQA_POST_LOG_ROOT="${NO_VQA_WAVE5_POST_LOG_ROOT:-/root/scorer_pdms93_logs/no_vqa_e35_scene_token_wave5_reference_post_v1}"
export NO_VQA_PACKAGE_ROOT="${NO_VQA_WAVE5_PACKAGE_ROOT:-/root/scorer_pdms93_artifacts/no_vqa_e35_scene_token_wave5_reference_v1}"
export NO_VQA_NAVTEST_RESULT_ROOT="${NO_VQA_WAVE5_NAVTEST_ROOT:-/root/scorer_pdms93_navtest/no_vqa_e35_scene_token_wave5_reference_v1}"
export NO_VQA_ONLINE_ROOT="${NO_VQA_WAVE5_ONLINE_ROOT:-/root/scorer_pdms93_online/no_vqa_e35_scene_token_wave5_reference_v1}"
export NO_VQA_REPORT_ROOT="${NO_VQA_WAVE5_REPORT_ROOT:-${repo_root}/reports/m0_independent_scorer_representation/no_vqa_e35_scene_token_wave5_reference_v1}"
export NO_VQA_INCLUDE_CALIBRATED=0
export NO_VQA_EXPECTED_EPOCHS=8
export NO_VQA_POST_GPU_IDS="${NO_VQA_WAVE5_POST_GPU_IDS:-0,1,2,3,4,5,6,7}"
export NO_VQA_SCORER_NAMES_CSV="combined_top8_reference_q10_strict_actor_seed2,combined_top8_reference_q50_strict_actor_seed2,combined_top16_reference_q10_strict_actor_seed2,combined_top16_reference_q50_strict_actor_seed2,combined_top32_reference_q10_strict_actor_seed2,private_top16_reference_q10_strict_actor_seed2,candidateonly_top16_reference_q10_strict_seed2,combined_top16_reference_q10_balanced_actor_seed2"

exec bash "${repo_root}/local_stage2/watch_no_vqa_scene_token_training_and_run_navtest.sh"
