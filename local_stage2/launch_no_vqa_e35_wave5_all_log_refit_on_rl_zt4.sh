#!/usr/bin/env bash

# Wait for Wave-5 on training-vla-zt2, copy only the three lightweight selected
# ranker artifacts through the shared project volume, and launch the locked
# all-log refit plus strict Navtest watcher on rl-zt4.  No cache or result is
# deleted and no running job is preempted.

set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
source_host="${NO_VQA_WAVE5_SOURCE_HOST:-training-vla-zt2}"
target_host="${NO_VQA_WAVE5_REFIT_HOST:-training-rl-zt4}"
remote_run_root="/root/scorer_pdms93_runs/no_vqa_e35_scene_token_wave5_reference_v1"
selection_root="${NO_VQA_WAVE5_SELECTION_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_wave5_reference_selection_v1}"
poll_seconds="${NO_VQA_REFIT_POLL_SECONDS:-30}"
names=(
  combined_top8_reference_q50_strict_actor_seed2
  combined_top16_reference_q50_strict_actor_seed2
  combined_top16_reference_q10_balanced_actor_seed2
)

until ssh "${source_host}" "test -f '${remote_run_root}/.wave5_complete'"; do
  echo "M0_WAVE5_REFIT_TRANSFER waiting_for_wave5 utc=$(date -u +%FT%TZ)"
  sleep "${poll_seconds}"
done

mkdir -p "${selection_root}"
for name in "${names[@]}"; do
  destination="${selection_root}/${name}.pt"
  if [[ ! -f "${destination}" ]]; then
    rsync -a --partial \
      "${source_host}:${remote_run_root}/${name}/best_m0_private_residual_scorer.pt" \
      "${destination}"
  fi
done

ssh "${target_host}" \
  'if [[ -e /root/scorer_pdms93_runs/no_vqa_e35_all_log_refit_wave5_reference_v1 || -e /root/scorer_pdms93_logs/no_vqa_e35_all_log_refit_wave5_reference_v1 ]]; then echo "Wave-5 refit output exists" >&2; exit 2; fi'
ssh "${target_host}" \
  "nohup bash '${repo_root}/local_stage2/run_no_vqa_e35_wave5_all_log_refit_remote.sh' >'/root/scorer_pdms93_logs/no_vqa_e35_all_log_refit_wave5_reference_launcher.log' 2>&1 </dev/null & echo \$!"
ssh "${target_host}" \
  "nohup bash '${repo_root}/local_stage2/watch_no_vqa_e35_wave5_all_log_refit_navtest_remote.sh' >'/root/scorer_pdms93_logs/no_vqa_e35_all_log_refit_wave5_reference_watcher.log' 2>&1 </dev/null & echo \$!"
echo "M0_WAVE5_ALL_LOG_REFIT_REMOTE_LAUNCHED host=${target_host}"
