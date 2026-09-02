#!/usr/bin/env bash

# Copy the verified immutable replay cache to rl-zt4 and launch wave 3.  The
# rsync is additive/resumable and never deletes remote data.

set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
host="${NO_VQA_WAVE3_HOST:-root@training-rl-zt4-ssh.world-model.svc.cluster.local}"
ssh_key="${NO_VQA_WAVE3_SSH_KEY:-/root/.ssh/id_ed25519}"
source_root="/root/scorer_pdms93_cache/no_vqa_e35_features_full_v1"
label_root="/root/scorer_pdms93_cache/no_vqa_e35_labels_full_v1"
verification="${repo_root}/reports/m0_independent_scorer_representation/no_vqa_e35_scene_token_v1/CACHE_VERIFICATION.json"
navtest_scores="/root/scorer_pdms93_cache/no_vqa_e35_navtest_candidate_scores_fp32_v1"
wait_seconds="${NO_VQA_WAVE3_POLL_SECONDS:-30}"
ssh_args=(-i "${ssh_key}" -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
rsync_ssh="ssh -i ${ssh_key} -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

while true; do
  if [[ -f "${verification}" ]] && python - "${verification}" <<'PY' >/dev/null 2>&1
import json
import sys
value = json.load(open(sys.argv[1]))
raise SystemExit(0 if value.get("status") == "PASS" and value.get("scene_count") == 103288 else 1)
PY
  then
    break
  fi
  echo "NO_VQA_WAVE3_TRANSFER waiting_for_verified_cache utc=$(date -u +%FT%TZ)"
  sleep "${wait_seconds}"
done

ssh "${ssh_args[@]}" "${host}" \
  'if [[ -e /root/scorer_pdms93_runs/no_vqa_e35_scene_token_wave3_v1 || -e /root/scorer_pdms93_runs/no_vqa_e35_scene_token_wave3_calibrated_v1 || -e /root/scorer_pdms93_logs/no_vqa_e35_scene_token_wave3_v1 ]]; then echo "wave3 remote output already exists" >&2; exit 2; fi; mkdir -p /root/scorer_pdms93_cache/no_vqa_e35_features_full_v1 /root/scorer_pdms93_cache/no_vqa_e35_labels_full_v1 /root/scorer_pdms93_logs'
rsync -a --partial -e "${rsync_ssh}" "${source_root}/" "${host}:${source_root}/"
rsync -a --partial -e "${rsync_ssh}" "${label_root}/" "${host}:${label_root}/"

ssh "${ssh_args[@]}" "${host}" \
  "nohup bash '${repo_root}/local_stage2/run_no_vqa_e35_scene_token_wave3_remote.sh' >'/root/scorer_pdms93_logs/no_vqa_e35_scene_token_wave3_launcher.log' 2>&1 </dev/null & echo \$!"
echo "NO_VQA_WAVE3_REMOTE_LAUNCHED host=${host}"

while [[ ! -f "${navtest_scores}/summary.json" || ! -f "${navtest_scores}/candidate_scores.npz" ]]; do
  echo "NO_VQA_WAVE3_TRANSFER waiting_for_local_candidate_matrix utc=$(date -u +%FT%TZ)"
  sleep "${wait_seconds}"
done
DRIVEVLA_REPO_ROOT="${repo_root}" \
  /root/.codex/skills/navsim-scorer-evaluation/scripts/validate_audit.sh \
  "${navtest_scores}"
ssh "${ssh_args[@]}" "${host}" \
  'mkdir -p /root/scorer_pdms93_cache/no_vqa_e35_navtest_candidate_scores_fp32_v1'
rsync -a --partial -e "${rsync_ssh}" \
  "${navtest_scores}/" "${host}:${navtest_scores}/"
echo "NO_VQA_WAVE3_CANDIDATE_MATRIX_TRANSFERRED host=${host}"
