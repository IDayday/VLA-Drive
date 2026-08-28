#!/usr/bin/env bash

set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/local_stage2/common.sh"

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 TRAIN_PID TRAIN_PGID TRAIN_EXPERIMENT EVAL_EXPERIMENT" >&2
  exit 2
fi

train_pid="$1"
train_pgid="$2"
train_experiment="$3"
eval_experiment="$4"
poll_seconds="${DRIVEVLA_WATCH_POLL_SECONDS:-60}"
expected_steps="${DRIVEVLA_EXPECTED_FINAL_STEP:-232416}"
expected_epoch="${DRIVEVLA_EXPECTED_FINAL_EPOCH:-35}"
NO_VQA_RUN_ROOT="${NO_VQA_RUN_ROOT:-/mnt/project/DriveVLA-M0-no-vqa/runs}"
training_dir="${NO_VQA_RUN_ROOT}/training/${train_experiment}"
watch_log="${training_dir}/train_then_evaluate.log"

mkdir -p "${training_dir}"
exec > >(tee -a "${watch_log}") 2>&1

printf 'WATCH_START timestamp=%s pid=%s pgid=%s experiment=%s\n' \
  "$(date -u +%FT%TZ)" "${train_pid}" "${train_pgid}" "${train_experiment}"

while kill -0 "${train_pid}" 2>/dev/null; do
  if [[ ! -r "/proc/${train_pid}/cmdline" ]]; then
    break
  fi
  train_command="$(tr '\0' ' ' < "/proc/${train_pid}/cmdline")"
  if [[ "${train_command}" != *"run_training_full.py"* ]] \
    || [[ "${train_command}" != *"experiment_name=${train_experiment}"* ]]; then
    echo "WATCH_ERROR PID no longer belongs to the expected training command" >&2
    exit 1
  fi
  sleep "${poll_seconds}"
done

while pgrep -g "${train_pgid}" >/dev/null 2>&1; do
  sleep 2
done

checkpoint_dir="${training_dir}/lightning_logs/version_0/checkpoints"
last_checkpoint="${checkpoint_dir}/last.ckpt"
if [[ ! -e "${last_checkpoint}" ]]; then
  echo "WATCH_ERROR training exited without last.ckpt" >&2
  exit 1
fi

best_checkpoint="$({
  LAST_CHECKPOINT="${last_checkpoint}" \
  EXPECTED_STEPS="${expected_steps}" \
  EXPECTED_EPOCH="${expected_epoch}" \
  "${DRIVEVLA_PYTHON}" - <<'PY'
import os
from pathlib import Path
import torch

last_path = Path(os.environ["LAST_CHECKPOINT"])
payload = torch.load(last_path, map_location="cpu", weights_only=False)
expected_steps = int(os.environ["EXPECTED_STEPS"])
expected_epoch = int(os.environ["EXPECTED_EPOCH"])
global_step = int(payload.get("global_step", -1))
epoch = int(payload.get("epoch", -1))
if (global_step, epoch) != (expected_steps, expected_epoch):
    raise RuntimeError(
        f"Incomplete training: epoch={epoch}, step={global_step}; "
        f"expected epoch={expected_epoch}, step={expected_steps}"
    )

matches = []
for state in payload.get("callbacks", {}).values():
    if isinstance(state, dict) and state.get("best_model_path"):
        candidate = Path(state["best_model_path"])
        if candidate.name.startswith("best-"):
            matches.append(candidate)
matches = sorted(set(matches))
if len(matches) != 1 or not matches[0].is_file():
    raise RuntimeError(f"Could not resolve one valid best checkpoint: {matches}")
print(matches[0])
PY
} | tail -n 1)"

printf 'TRAIN_COMPLETE timestamp=%s last=%s best=%s\n' \
  "$(date -u +%FT%TZ)" "${last_checkpoint}" "${best_checkpoint}"

if [[ "${DRIVEVLA_WATCH_SKIP_EVALUATION:-0}" == "1" ]]; then
  echo "EVALUATION_SKIPPED_BY_REQUEST"
  exit 0
fi

"${DRIVEVLA_REPO_ROOT}/local_no_vqa_full_ft/evaluate_checkpoint.sh" \
  "${best_checkpoint}" "${eval_experiment}"
printf 'EVALUATION_COMPLETE timestamp=%s experiment=%s\n' \
  "$(date -u +%FT%TZ)" "${eval_experiment}"
