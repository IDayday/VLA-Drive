#!/usr/bin/env bash

# Launch one controlled Stage-2 run across this host and vla-zt2.  The public
# checkpoint step count plus the paper's 16-GPU statement imply 16 ranks x
# batch 1, so this is the closest available rank layout to the reported run.

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/common.sh"

remote_host="${STAGE2_REMOTE_HOST:-training-vla-zt2}"
portable_python="${STAGE2_PORTABLE_PYTHON:-/mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/envs/navsim_py39_exact/bin/python}"
portable_extra_site="${STAGE2_PORTABLE_EXTRA_SITE:-/mnt/project/DriveVLA-M0-env/lib/python3.9/site-packages}"
lightning_overlay="${STAGE2_LIGHTNING_OVERLAY:-}"
transformers_overlay="${STAGE2_TRANSFORMERS_OVERLAY:-}"
required_lightning="${STAGE2_REQUIRE_LIGHTNING_VERSION:-2.6.0}"
required_transformers="${STAGE2_REQUIRE_TRANSFORMERS_VERSION:-}"
experiment="${STAGE2_EXPERIMENT:-stage2_reproduction_seed2_eager_lr1e4_16x1}"
output_dir="${STAGE2_OUTPUT_DIR:-${DRIVEVLA_STAGE2_RUN_ROOT}/training/${experiment}}"
master_addr="${STAGE2_MASTER_ADDR:-$(hostname -i | awk '{print $1}')}"
master_port="${STAGE2_MASTER_PORT:-29531}"
node0_log="${output_dir}/train-node0.log"
node1_log="${output_dir}/train-node1.log"
launcher_state="${output_dir}/multinode_launcher_state.env"
train_entrypoint="${script_dir}/train_stage2_reproduction.sh"

if [[ ! -x "${portable_python}" ]]; then
  echo "Portable Python is not executable: ${portable_python}" >&2
  exit 1
fi
if [[ ! -x "${train_entrypoint}" ]]; then
  echo "Training entrypoint is not executable: ${train_entrypoint}" >&2
  exit 1
fi
if [[ -n "${lightning_overlay}" && ! -d "${lightning_overlay}" ]]; then
  echo "Lightning overlay does not exist: ${lightning_overlay}" >&2
  exit 1
fi
if [[ -n "${transformers_overlay}" && ! -d "${transformers_overlay}" ]]; then
  echo "Transformers overlay does not exist: ${transformers_overlay}" >&2
  exit 1
fi
if [[ -d "${output_dir}" ]] \
  && [[ -n "$(find "${output_dir}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite nonempty run directory: ${output_dir}" >&2
  exit 1
fi

runtime_probe=(
  "${portable_python}" -c
  'import json,pytorch_lightning,torch,transformers; print(json.dumps({"lightning":pytorch_lightning.__version__,"torch":torch.__version__,"cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),"transformers":transformers.__version__},sort_keys=True))'
)
probe_pythonpath="${DRIVEVLA_REPO_ROOT}:${DRIVEVLA_REPO_ROOT}/nuplan-devkit:${portable_extra_site}"
if [[ -n "${lightning_overlay}" ]]; then
  probe_pythonpath="${lightning_overlay}:${probe_pythonpath}"
fi
if [[ -n "${transformers_overlay}" ]]; then
  probe_pythonpath="${transformers_overlay}:${probe_pythonpath}"
fi
local_runtime="$(PYTHONPATH="${probe_pythonpath}" "${runtime_probe[@]}")"
remote_runtime="$(ssh -o BatchMode=yes "${remote_host}" \
  "cd $(printf '%q' "${DRIVEVLA_REPO_ROOT}") && PYTHONPATH=$(printf '%q' "${probe_pythonpath}") $(printf '%q ' "${runtime_probe[@]}")")"
if [[ "${local_runtime}" != "${remote_runtime}" ]]; then
  echo "Runtime mismatch between nodes" >&2
  echo "node0: ${local_runtime}" >&2
  echo "node1: ${remote_runtime}" >&2
  exit 1
fi
if [[ "${local_runtime}" != *"\"lightning\": \"${required_lightning}\""* ]]; then
  echo "Expected the locked Lightning ${required_lightning} runtime, got: ${local_runtime}" >&2
  exit 1
fi
if [[ -n "${required_transformers}" ]] \
  && [[ "${local_runtime}" != *"\"transformers\": \"${required_transformers}\""* ]]; then
  echo "Expected the locked Transformers ${required_transformers} runtime, got: ${local_runtime}" >&2
  exit 1
fi

check_local_idle() {
  mapfile -t training_pids < <(pgrep -f '[r]un_training_full[.]py' || true)
  if (( ${#training_pids[@]} > 0 )); then
    echo "Local training processes are already running: ${training_pids[*]}" >&2
    return 1
  fi
  local gpu_pid gpu_command
  while IFS= read -r gpu_pid; do
    [[ -n "${gpu_pid}" && -r "/proc/${gpu_pid}/cmdline" ]] || continue
    gpu_command="$(tr '\0' ' ' < "/proc/${gpu_pid}/cmdline")"
    [[ "${gpu_command}" == *"/mnt/project/gpu_stress.py"* ]] && continue
    echo "Local GPU is occupied by PID ${gpu_pid}: ${gpu_command}" >&2
    return 1
  done < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sort -u)
}

check_remote_idle() {
  ssh -o BatchMode=yes "${remote_host}" 'bash -s' <<'REMOTE_CHECK'
set -euo pipefail
mapfile -t training_pids < <(pgrep -f '[r]un_training_full[.]py' || true)
if (( ${#training_pids[@]} > 0 )); then
  echo "Remote training processes are already running: ${training_pids[*]}" >&2
  exit 1
fi
while IFS= read -r gpu_pid; do
  [[ -n "${gpu_pid}" && -r "/proc/${gpu_pid}/cmdline" ]] || continue
  gpu_command="$(tr '\0' ' ' < "/proc/${gpu_pid}/cmdline")"
  [[ "${gpu_command}" == *"/mnt/project/gpu_stress.py"* ]] && continue
  echo "Remote GPU is occupied by PID ${gpu_pid}: ${gpu_command}" >&2
  exit 1
done < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sort -u)
REMOTE_CHECK
}

check_local_idle
check_remote_idle
mkdir -p "${output_dir}"

common_environment=(
  "DRIVEVLA_REPO_ROOT=${DRIVEVLA_REPO_ROOT}"
  "DRIVEVLA_PYTHON=${portable_python}"
  "DRIVEVLA_NAVTRAIN_FEATURE_CACHE=${DRIVEVLA_NAVTRAIN_FEATURE_CACHE}"
  "PYTHONPATH=${probe_pythonpath}"
  "STAGE2_REQUIRE_LIGHTNING_VERSION=${required_lightning}"
  "STAGE2_REQUIRE_TRANSFORMERS_VERSION=${required_transformers}"
  "STAGE2_EXPERIMENT=${experiment}"
  "STAGE2_OUTPUT_DIR=${output_dir}"
  "STAGE2_NUM_GPUS=8"
  "STAGE2_NUM_NODES=2"
  "STAGE2_WORLD_SIZE=16"
  "STAGE2_BATCH_SIZE=1"
  "STAGE2_ACCUMULATE_GRAD_BATCHES=1"
  "STAGE2_EFFECTIVE_GLOBAL_BATCH_SIZE=16"
  "STAGE2_SEED=${STAGE2_SEED:-2}"
  "STAGE2_MAX_EPOCHS=${STAGE2_MAX_EPOCHS:-27}"
  "STAGE2_BASE_LR=${STAGE2_BASE_LR:-1e-4}"
  "STAGE2_BASE_BATCH_SIZE=${STAGE2_BASE_BATCH_SIZE:-16}"
  "STAGE2_SCHEDULER=${STAGE2_SCHEDULER:-none}"
  "STAGE2_FLASH_ATTENTION=false"
  "STAGE2_FROZEN_BACKBONE_MODE=train"
  "STAGE2_DECAY_NORM_AND_BIAS=true"
  "STAGE2_PREPAD_DATASET=false"
  "STAGE2_OFFICIAL_SAMPLER=true"
  "STAGE2_LONG_TRAJECTORY_ADDITIONAL_POSES=${STAGE2_LONG_TRAJECTORY_ADDITIONAL_POSES:--1}"
  "DRIVEVLA_SCORE_PROCESSES=${DRIVEVLA_SCORE_PROCESSES:-16}"
  "DRIVEVLA_SCORE_PARTITIONS=${DRIVEVLA_SCORE_PARTITIONS:-8}"
  "DRIVEVLA_TRAIN_LOG_INTERVAL=${DRIVEVLA_TRAIN_LOG_INTERVAL:-10}"
  "DRIVEVLA_TIMING_INTERVAL=${DRIVEVLA_TIMING_INTERVAL:-100}"
  "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
  "MASTER_ADDR=${master_addr}"
  "MASTER_PORT=${master_port}"
)
training_arguments=(
  +trainer.params.enable_model_summary=false
  +trainer.params.log_every_n_steps=10
  "$@"
)

build_command() {
  local node_rank="$1"
  local command=(
    env "${common_environment[@]}" "NODE_RANK=${node_rank}"
    "${train_entrypoint}" "${training_arguments[@]}"
  )
  printf '%q ' "${command[@]}"
}

remote_command="$(build_command 1)"
remote_pid="$(ssh -o BatchMode=yes "${remote_host}" \
  "cd $(printf '%q' "${DRIVEVLA_REPO_ROOT}"); nohup setsid ${remote_command} >> $(printf '%q' "${node1_log}") 2>&1 < /dev/null & remote_pid=\$!; printf '%s\\n' \"\${remote_pid}\"")"
if [[ ! "${remote_pid}" =~ ^[0-9]+$ ]]; then
  echo "Failed to obtain remote launcher PID: ${remote_pid}" >&2
  exit 1
fi

local_command="$(build_command 0)"
nohup setsid bash -lc \
  "cd $(printf '%q' "${DRIVEVLA_REPO_ROOT}") && exec ${local_command}" \
  >> "${node0_log}" 2>&1 < /dev/null &
local_pid=$!

cleanup_failed_launch() {
  kill -- "-${local_pid}" 2>/dev/null || true
  ssh -o BatchMode=yes "${remote_host}" \
    "kill -- -${remote_pid} 2>/dev/null || true" >/dev/null 2>&1 || true
}

sleep 10
if ! kill -0 "${local_pid}" 2>/dev/null; then
  echo "Local node launcher exited during startup" >&2
  tail -n 100 "${node0_log}" >&2
  cleanup_failed_launch
  exit 1
fi
if ! ssh -o BatchMode=yes "${remote_host}" "kill -0 ${remote_pid}"; then
  echo "Remote node launcher exited during startup" >&2
  tail -n 100 "${node1_log}" >&2
  cleanup_failed_launch
  exit 1
fi

rank_zero_pid=""
for _ in $(seq 1 180); do
  while IFS= read -r child_pid; do
    [[ -r "/proc/${child_pid}/cmdline" ]] || continue
    child_command="$(tr '\0' ' ' < "/proc/${child_pid}/cmdline")"
    if [[ "${child_command}" == *"run_training_full.py"* ]] \
      && [[ "${child_command}" == *"experiment_name=${experiment}"* ]]; then
      rank_zero_pid="${child_pid}"
      break 2
    fi
  done < <(pgrep -P "${local_pid}" || true)
  sleep 1
done
if [[ -z "${rank_zero_pid}" ]]; then
  echo "Timed out waiting for node-0 rank-zero process" >&2
  cleanup_failed_launch
  exit 1
fi

local_pgid="$(ps -o pgid= -p "${local_pid}" | tr -d ' ')"
if [[ "${local_pgid}" != "${local_pid}" ]]; then
  echo "Local launcher is not its own process group" >&2
  cleanup_failed_launch
  exit 1
fi

evaluation_experiment="${STAGE2_EVAL_EXPERIMENT:-${experiment}_navtest}"
nohup setsid env \
  DRIVEVLA_REPO_ROOT="${DRIVEVLA_REPO_ROOT}" \
  DRIVEVLA_PYTHON="${portable_python}" \
  PYTHONPATH="${probe_pythonpath}" \
  DRIVEVLA_WATCH_POLL_SECONDS="${DRIVEVLA_WATCH_POLL_SECONDS:-60}" \
  DRIVEVLA_EXPECTED_FINAL_STEP="${DRIVEVLA_EXPECTED_FINAL_STEP:-174312}" \
  DRIVEVLA_EXPECTED_FINAL_EPOCH="${DRIVEVLA_EXPECTED_FINAL_EPOCH:-26}" \
  DRIVEVLA_WATCH_SKIP_EVALUATION="${DRIVEVLA_WATCH_SKIP_EVALUATION:-0}" \
  STAGE2_EVAL_FLASH_ATTENTION=false \
  DRIVEVLA_TRAINING_DIR="${output_dir}" \
  "${script_dir}/watch_stage2_and_evaluate.sh" \
  "${rank_zero_pid}" "${local_pgid}" "${experiment}" \
  "${evaluation_experiment}" > /dev/null 2>&1 < /dev/null &
watcher_pid=$!

{
  printf 'experiment=%q\n' "${experiment}"
  printf 'output_dir=%q\n' "${output_dir}"
  printf 'runtime=%q\n' "${local_runtime}"
  printf 'master_addr=%q\n' "${master_addr}"
  printf 'master_port=%q\n' "${master_port}"
  printf 'remote_host=%q\n' "${remote_host}"
  printf 'local_launcher_pid=%q\n' "${local_pid}"
  printf 'local_launcher_pgid=%q\n' "${local_pgid}"
  printf 'rank_zero_pid=%q\n' "${rank_zero_pid}"
  printf 'remote_launcher_pid=%q\n' "${remote_pid}"
  printf 'watcher_pid=%q\n' "${watcher_pid}"
  printf 'launched_at=%q\n' "$(date -u +%FT%TZ)"
} > "${launcher_state}"

printf 'MULTINODE_LAUNCH_OK experiment=%s local_pid=%s remote_pid=%s watcher_pid=%s output=%s\n' \
  "${experiment}" "${local_pid}" "${remote_pid}" "${watcher_pid}" "${output_dir}"
