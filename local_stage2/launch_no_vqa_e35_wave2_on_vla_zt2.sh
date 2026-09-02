#!/usr/bin/env bash

# Wait for the locally verified immutable No-VQA replay cache, copy it without
# deletion to training-vla-zt2 host-local storage, and start the independent
# eight-GPU top-K/candidate-hidden-state wave.

set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
host="${NO_VQA_WAVE2_HOST:-training-vla-zt2}"
source_root="/root/scorer_pdms93_cache/no_vqa_e35_features_full_v1"
label_root="/root/scorer_pdms93_cache/no_vqa_e35_labels_full_v1"
verification="${repo_root}/reports/m0_independent_scorer_representation/no_vqa_e35_scene_token_v1/CACHE_VERIFICATION.json"
wait_seconds="${NO_VQA_WAVE2_POLL_SECONDS:-30}"

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
  echo "NO_VQA_WAVE2_TRANSFER waiting_for_verified_cache utc=$(date -u +%FT%TZ)"
  sleep "${wait_seconds}"
done

ssh "${host}" 'if [[ -e /root/scorer_pdms93_cache/no_vqa_e35_features_full_v1 || -e /root/scorer_pdms93_cache/no_vqa_e35_labels_full_v1 || -e /root/scorer_pdms93_runs/no_vqa_e35_scene_token_wave2_v1 ]]; then echo "wave2 remote target already exists" >&2; exit 2; fi; mkdir -p /root/scorer_pdms93_cache/no_vqa_e35_features_full_v1 /root/scorer_pdms93_cache/no_vqa_e35_labels_full_v1 /root/scorer_pdms93_logs'

rsync -a --partial "${source_root}/" "${host}:${source_root}/"
rsync -a --partial "${label_root}/" "${host}:${label_root}/"

ssh "${host}" "nohup bash '${repo_root}/local_stage2/run_no_vqa_e35_scene_token_wave2_remote.sh' >'/root/scorer_pdms93_logs/no_vqa_e35_scene_token_wave2_launcher.log' 2>&1 </dev/null & echo \$!"
ssh "${host}" "setsid -f bash -c \"exec bash '${repo_root}/local_stage2/watch_no_vqa_e35_wave2_and_run_navtest_remote.sh' >'/root/scorer_pdms93_logs/no_vqa_e35_scene_token_wave2_post_watcher.log' 2>&1\" </dev/null"
echo "NO_VQA_WAVE2_REMOTE_LAUNCHED host=${host}"

navtest_scores="/root/scorer_pdms93_cache/no_vqa_e35_navtest_candidate_scores_fp32_v1"
while [[ ! -f "${navtest_scores}/summary.json" || ! -f "${navtest_scores}/candidate_scores.npz" ]]; do
  echo "NO_VQA_WAVE2_TRANSFER waiting_for_local_candidate_matrix utc=$(date -u +%FT%TZ)"
  sleep "${wait_seconds}"
done
/root/.codex/skills/navsim-scorer-evaluation/scripts/validate_audit.sh "${navtest_scores}"
ssh "${host}" 'if [[ -e /root/scorer_pdms93_cache/no_vqa_e35_navtest_candidate_scores_fp32_v1 ]]; then echo "remote candidate matrix target exists" >&2; exit 2; fi; mkdir -p /root/scorer_pdms93_cache/no_vqa_e35_navtest_candidate_scores_fp32_v1'
rsync -a --partial "${navtest_scores}/" "${host}:${navtest_scores}/"
echo "NO_VQA_WAVE2_CANDIDATE_MATRIX_TRANSFERRED host=${host}"
