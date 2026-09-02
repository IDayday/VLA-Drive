#!/usr/bin/env bash

# Wait for the independent No-VQA multiview Navtest cache, then promote every
# held-out-log-positive wave-6 artifact and run strict full Navtest + parity.

set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
private_navtest_root="${NO_VQA_MULTIVIEW_TEST_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_multiview_navtest_pool2_tiles4_v1_8shard}"
poll_seconds="${NO_VQA_WAVE6_POLL_SECONDS:-30}"
while [[ ! -f "${private_navtest_root}/.complete" ]]; do
  echo "NO_VQA_WAVE6_NAVTEST waiting_for_multiview_cache utc=$(date -u +%FT%TZ)"
  sleep "${poll_seconds}"
done

export NO_VQA_SCORER_RUN_ROOT="${NO_VQA_WAVE6_RUN_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_multiview_wave6_v1}"
export NO_VQA_CALIBRATED_RUN_ROOT="${NO_VQA_WAVE6_CALIBRATED_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_multiview_wave6_calibrated_v1}"
export NO_VQA_POST_LOG_ROOT="${NO_VQA_WAVE6_POST_LOG_ROOT:-/root/scorer_pdms93_logs/no_vqa_e35_multiview_wave6_post_v1}"
export NO_VQA_PACKAGE_ROOT="${NO_VQA_WAVE6_PACKAGE_ROOT:-/root/scorer_pdms93_artifacts/no_vqa_e35_multiview_wave6_v1}"
export NO_VQA_NAVTEST_RESULT_ROOT="${NO_VQA_WAVE6_NAVTEST_ROOT:-/root/scorer_pdms93_navtest/no_vqa_e35_multiview_wave6_v1}"
export NO_VQA_ONLINE_ROOT="${NO_VQA_WAVE6_ONLINE_ROOT:-/root/scorer_pdms93_online/no_vqa_e35_multiview_wave6_v1}"
export NO_VQA_REPORT_ROOT="${NO_VQA_WAVE6_REPORT_ROOT:-${repo_root}/reports/m0_independent_scorer_representation/no_vqa_e35_multiview_wave6_v1}"
export NO_VQA_PRIVATE_OBSERVATION_ROOT="${private_navtest_root}"
export NO_VQA_INCLUDE_CALIBRATED=0
export NO_VQA_EXPECTED_EPOCHS=8
export NO_VQA_POST_GPU_IDS="${NO_VQA_WAVE6_POST_GPU_IDS:-0,1,2,3,4,5,6,7}"
export NO_VQA_SCORER_NAMES_CSV="rawcombined_top16_hybrid_standard_actor_seed2,rawcombined_top8_hybrid_topregret_actor_seed2,rawcombined_top16_reference_q50_strict_actor_seed2,rawcombined_top32_reference_q50_strict_actor_seed2,rawprivate_top16_reference_q50_strict_actor_seed2,rawcontextcombined_top16_reference_q50_strict_actor_seed2,rawcombined_top16_reference_q50_balanced_actor_seed2,rawcombined_top16_reference_q50_strict_noactor_seed2"

exec bash "${repo_root}/local_stage2/watch_no_vqa_scene_token_training_and_run_navtest.sh"
