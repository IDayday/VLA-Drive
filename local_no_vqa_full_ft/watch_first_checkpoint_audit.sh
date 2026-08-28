#!/usr/bin/env bash

set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/local_stage2/common.sh"

NO_VQA_RUN_ROOT="${NO_VQA_RUN_ROOT:-/mnt/project/DriveVLA-M0-no-vqa/runs}"
NO_VQA_EXPERIMENT="${NO_VQA_EXPERIMENT:-no_vqa_full_ft_seed0_e36}"
training_dir="${NO_VQA_RUN_ROOT}/training/${NO_VQA_EXPERIMENT}"
checkpoint="${training_dir}/lightning_logs/version_0/checkpoints/last.ckpt"
pass_marker="${training_dir}/.epoch0_checkpoint_audit_passed"
poll_seconds="${DRIVEVLA_FIRST_AUDIT_POLL_SECONDS:-60}"
max_polls="${DRIVEVLA_FIRST_AUDIT_MAX_POLLS:-300}"

printf 'FIRST_CHECKPOINT_AUDIT_WATCH_START timestamp=%s checkpoint=%s\n' \
  "$(date -u +%FT%TZ)" "${checkpoint}"

for ((poll = 1; poll <= max_polls; poll++)); do
  if [[ -e "${checkpoint}" ]]; then
    # ModelCheckpoint may expose the path while the multi-GB payload is still
    # being flushed. Require an unchanged, nonzero size across one poll.
    first_size="$(stat -Lc %s "${checkpoint}" 2>/dev/null || true)"
    sleep "${poll_seconds}"
    second_size="$(stat -Lc %s "${checkpoint}" 2>/dev/null || true)"
    if [[ -n "${first_size}" && "${first_size}" != "0" \
      && "${first_size}" == "${second_size}" ]]; then
      for attempt in 1 2 3; do
        if "${DRIVEVLA_PYTHON}" \
          "${DRIVEVLA_REPO_ROOT}/local_no_vqa_full_ft/audit_full_checkpoint.py" \
          "${checkpoint}" \
          "${DRIVEVLA_VLM_DIR}" \
          --expected-step 6456 \
          --expected-epoch 0; then
          touch "${pass_marker}"
          printf 'FIRST_CHECKPOINT_AUDIT_PASS timestamp=%s marker=%s\n' \
            "$(date -u +%FT%TZ)" "${pass_marker}"
          exit 0
        fi
        printf 'FIRST_CHECKPOINT_AUDIT_RETRY timestamp=%s attempt=%d\n' \
          "$(date -u +%FT%TZ)" "${attempt}" >&2
        sleep 30
      done
      echo "FIRST_CHECKPOINT_AUDIT_ERROR stable checkpoint failed three audits" >&2
      exit 1
    fi
  else
    sleep "${poll_seconds}"
  fi
done

echo "FIRST_CHECKPOINT_AUDIT_TIMEOUT checkpoint was not available" >&2
exit 1
