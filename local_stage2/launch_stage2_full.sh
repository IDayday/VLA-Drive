#!/usr/bin/env bash

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/common.sh"

experiment="${STAGE2_EXPERIMENT:-stage2_full_seed0_pipeline_v8_restart}"
output_dir="${STAGE2_OUTPUT_DIR:-${DRIVEVLA_STAGE2_RUN_ROOT}/training/${experiment}}"
evaluation_experiment="${STAGE2_EVAL_EXPERIMENT:-${experiment}_navtest}"
train_log="${output_dir}/train.log"
launcher_state="${output_dir}/launcher_state.env"
resume_checkpoint="${STAGE2_TRAIN_CKPT:-}"
train_entrypoint="${STAGE2_TRAIN_ENTRYPOINT:-${script_dir}/train_stage2_full.sh}"
evaluation_flash_attention="${STAGE2_EVAL_FLASH_ATTENTION:-${STAGE2_FLASH_ATTENTION:-true}}"
if [[ ! -x "${train_entrypoint}" ]]; then
  echo "Training entrypoint is not executable: ${train_entrypoint}" >&2
  exit 1
fi

if [[ -d "${output_dir}" ]] \
  && [[ -n "$(find "${output_dir}" -mindepth 1 -print -quit 2>/dev/null)" ]] \
  && [[ -z "${resume_checkpoint}" ]]; then
  echo "Refusing to overwrite nonempty run directory: ${output_dir}" >&2
  exit 1
fi

mapfile -t training_pids < <(pgrep -f '[r]un_training_full[.]py' || true)
if (( ${#training_pids[@]} > 0 )); then
  echo "Refusing to launch beside existing training PIDs: ${training_pids[*]}" >&2
  ps -o pid,pgid,stat,etime,args -p "$(IFS=,; echo "${training_pids[*]}")" >&2
  exit 1
fi

unrelated_gpu_pids=()
while IFS= read -r gpu_pid; do
  [[ -n "${gpu_pid}" && -r "/proc/${gpu_pid}/cmdline" ]] || continue
  gpu_command="$(tr '\0' ' ' < "/proc/${gpu_pid}/cmdline")"
  if [[ "${gpu_command}" != *"/mnt/project/gpu_stress.py"* ]]; then
    unrelated_gpu_pids+=("${gpu_pid}")
  fi
done < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | sort -u
)
if (( ${#unrelated_gpu_pids[@]} > 0 )); then
  echo "Refusing to take GPUs used by unrelated PIDs: ${unrelated_gpu_pids[*]}" >&2
  exit 1
fi

mkdir -p "${output_dir}"
printf 'DETACHED_LAUNCH_START timestamp=%s experiment=%s output=%s resume=%s\n' \
  "$(date -u +%FT%TZ)" "${experiment}" "${output_dir}" \
  "${resume_checkpoint:-none}" > "${train_log}"

nohup setsid env \
  STAGE2_EXPERIMENT="${experiment}" \
  STAGE2_OUTPUT_DIR="${output_dir}" \
  STAGE2_TRAIN_CKPT="${resume_checkpoint}" \
  DRIVEVLA_SCORE_PROCESSES="${DRIVEVLA_SCORE_PROCESSES:-16}" \
  DRIVEVLA_SCORE_PARTITIONS="${DRIVEVLA_SCORE_PARTITIONS:-8}" \
  DRIVEVLA_TRAIN_LOG_INTERVAL="${DRIVEVLA_TRAIN_LOG_INTERVAL:-10}" \
  DRIVEVLA_TIMING_INTERVAL="${DRIVEVLA_TIMING_INTERVAL:-100}" \
  "${train_entrypoint}" \
  +trainer.params.enable_model_summary=false \
  +trainer.params.log_every_n_steps=10 \
  "$@" >> "${train_log}" 2>&1 < /dev/null &
launcher_pid=$!

rank_zero_pid=""
for _ in $(seq 1 120); do
  if ! kill -0 "${launcher_pid}" 2>/dev/null; then
    echo "Detached launcher exited before rank zero started" >&2
    tail -n 80 "${train_log}" >&2
    exit 1
  fi
  while IFS= read -r child_pid; do
    [[ -r "/proc/${child_pid}/cmdline" ]] || continue
    child_command="$(tr '\0' ' ' < "/proc/${child_pid}/cmdline")"
    if [[ "${child_command}" == *"run_training_full.py"* ]] \
      && [[ "${child_command}" == *"experiment_name=${experiment}"* ]]; then
      rank_zero_pid="${child_pid}"
      break 2
    fi
  done < <(pgrep -P "${launcher_pid}" || true)
  sleep 1
done
if [[ -z "${rank_zero_pid}" ]]; then
  echo "Timed out waiting for the rank-zero training process" >&2
  exit 1
fi

launcher_pgid="$(ps -o pgid= -p "${launcher_pid}" | tr -d ' ')"
launcher_sid="$(ps -o sid= -p "${launcher_pid}" | tr -d ' ')"
if [[ "${launcher_pgid}" != "${launcher_pid}" \
  || "${launcher_sid}" != "${launcher_pid}" ]]; then
  echo "Detached process is not its own session: pid=${launcher_pid} " \
    "pgid=${launcher_pgid} sid=${launcher_sid}" >&2
  exit 1
fi

nohup setsid env \
  DRIVEVLA_WATCH_POLL_SECONDS="${DRIVEVLA_WATCH_POLL_SECONDS:-60}" \
  DRIVEVLA_EXPECTED_FINAL_STEP="${DRIVEVLA_EXPECTED_FINAL_STEP:-174312}" \
  DRIVEVLA_EXPECTED_FINAL_EPOCH="${DRIVEVLA_EXPECTED_FINAL_EPOCH:-26}" \
  DRIVEVLA_WATCH_SKIP_EVALUATION="${DRIVEVLA_WATCH_SKIP_EVALUATION:-0}" \
  STAGE2_EVAL_FLASH_ATTENTION="${evaluation_flash_attention}" \
  DRIVEVLA_TRAINING_DIR="${output_dir}" \
  "${script_dir}/watch_stage2_and_evaluate.sh" \
  "${rank_zero_pid}" "${launcher_pgid}" "${experiment}" \
  "${evaluation_experiment}" > /dev/null 2>&1 < /dev/null &
watcher_pid=$!
sleep 1
if ! kill -0 "${watcher_pid}" 2>/dev/null; then
  echo "Detached watcher failed to remain alive" >&2
  exit 1
fi

{
  printf 'experiment=%q\n' "${experiment}"
  printf 'output_dir=%q\n' "${output_dir}"
  printf 'launcher_pid=%q\n' "${launcher_pid}"
  printf 'launcher_pgid=%q\n' "${launcher_pgid}"
  printf 'launcher_sid=%q\n' "${launcher_sid}"
  printf 'rank_zero_pid=%q\n' "${rank_zero_pid}"
  printf 'watcher_pid=%q\n' "${watcher_pid}"
  printf 'evaluation_flash_attention=%q\n' "${evaluation_flash_attention}"
  printf 'launched_at=%q\n' "$(date -u +%FT%TZ)"
} > "${launcher_state}"

printf 'DETACHED_LAUNCH_OK experiment=%s launcher_pid=%s rank_zero_pid=%s watcher_pid=%s log=%s\n' \
  "${experiment}" "${launcher_pid}" "${rank_zero_pid}" "${watcher_pid}" \
  "${train_log}"
