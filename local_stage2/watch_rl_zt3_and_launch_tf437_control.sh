#!/usr/bin/env bash

# Wait for the user-authorized rl-zt3 GPUs 3,5,6,7, then launch a bounded
# 1,000-optimizer-step Transformers-4.37.2 control.  The four ranks accumulate
# four microbatches so each optimizer step replays the reference global batch
# of 16 used by the 16x1 run.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
remote_host="${STAGE2_RL_ZT3_HOST:-training-rl-zt3}"
poll_seconds="${STAGE2_RL_ZT3_POLL_SECONDS:-30}"
portable_python="/mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/envs/navsim_py39_exact/bin/python"
transformers_overlay="/mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/envs/transformers_4_37_2"
lightning_overlay="/mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/envs/lightning_2_5_1"
extra_site="/mnt/project/DriveVLA-M0-env/lib/python3.9/site-packages"
experiment="stage2_source_cosine_seed2_tf437_peft010_4x1_acc4_step1000_rlzt3"
output_dir="/mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/ablations/${experiment}"
launch_log="${output_dir}.launch-monitor.log"
remote_log="${output_dir}/train.log"
gpu_list="3,5,6,7"

mkdir -p "$(dirname "${output_dir}")"
if [[ -e "${output_dir}" ]] \
  && [[ -n "$(find "${output_dir}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite nonempty control directory: ${output_dir}" >&2
  exit 1
fi

exec >> "${launch_log}" 2>&1
printf 'MONITOR_START timestamp=%s host=%s gpus=%s\n' \
  "$(date -u +%FT%TZ)" "${remote_host}" "${gpu_list}"

while true; do
  if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "${remote_host}" \
    'test -x /mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/envs/navsim_py39_exact/bin/python' \
    >/dev/null 2>&1; then
    sleep "${poll_seconds}"
    continue
  fi

  occupied="$({ ssh -o BatchMode=yes "${remote_host}" 'bash -s' <<'REMOTE_CHECK'
set -euo pipefail
for gpu in 3 5 6 7; do
  while IFS= read -r pid; do
    [[ -n "${pid}" && -r "/proc/${pid}/cmdline" ]] || continue
    command="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
    [[ "${command}" == *"/mnt/project/gpu_stress.py"* ]] && continue
    printf 'gpu=%s pid=%s command=%s\n' "${gpu}" "${pid}" "${command}"
  done < <(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)
done
REMOTE_CHECK
  } 2>&1)" || {
    printf 'REMOTE_CHECK_RETRY timestamp=%s detail=%q\n' \
      "$(date -u +%FT%TZ)" "${occupied}"
    sleep "${poll_seconds}"
    continue
  }
  if [[ -n "${occupied}" ]]; then
    printf 'GPU_BUSY_RETRY timestamp=%s detail=%q\n' \
      "$(date -u +%FT%TZ)" "${occupied}"
    sleep "${poll_seconds}"
    continue
  fi
  break
done

pythonpath="${repo_root}:${repo_root}/nuplan-devkit:${transformers_overlay}:${lightning_overlay}:${extra_site}"
runtime="$(ssh -o BatchMode=yes "${remote_host}" \
  "PYTHONPATH=$(printf '%q' "${pythonpath}") $(printf '%q' "${portable_python}") -c 'import json,peft,pytorch_lightning,torch,transformers; print(json.dumps({\"torch\":torch.__version__,\"transformers\":transformers.__version__,\"peft\":peft.__version__,\"lightning\":pytorch_lightning.__version__},sort_keys=True))'")"
if [[ "${runtime}" != *'"transformers": "4.37.2"'* ]] \
  || [[ "${runtime}" != *'"peft": "0.10.0"'* ]] \
  || [[ "${runtime}" != *'"lightning": "2.5.1"'* ]]; then
  printf 'RUNTIME_MISMATCH %s\n' "${runtime}" >&2
  exit 2
fi

mkdir -p "${output_dir}"
remote_command=(
  env
  "DRIVEVLA_REPO_ROOT=${repo_root}"
  "DRIVEVLA_PYTHON=${portable_python}"
  "PYTHONPATH=${pythonpath}"
  "CUDA_VISIBLE_DEVICES=${gpu_list}"
  "STAGE2_REQUIRE_TRANSFORMERS_VERSION=4.37.2"
  "STAGE2_REQUIRE_LIGHTNING_VERSION=2.5.1"
  "STAGE2_EXPERIMENT=${experiment}"
  "STAGE2_OUTPUT_DIR=${output_dir}"
  STAGE2_NUM_GPUS=4
  STAGE2_NUM_NODES=1
  STAGE2_BATCH_SIZE=1
  STAGE2_ACCUMULATE_GRAD_BATCHES=4
  STAGE2_EFFECTIVE_GLOBAL_BATCH_SIZE=16
  STAGE2_SEED=2
  STAGE2_MAX_EPOCHS=27
  STAGE2_BASE_LR=1e-4
  STAGE2_BASE_BATCH_SIZE=16
  STAGE2_SCHEDULER=source_cosine
  STAGE2_SCHEDULER_DATASET_SIZE=103288
  STAGE2_SCHEDULER_WARMUP_RATIO=0.1
  STAGE2_SCHEDULER_START_LR_RATIO=1e-6
  STAGE2_SCHEDULER_MIN_LR_RATIO=0.0
  STAGE2_FLASH_ATTENTION=false
  STAGE2_FROZEN_BACKBONE_MODE=train
  STAGE2_DECAY_NORM_AND_BIAS=true
  STAGE2_PREPAD_DATASET=false
  STAGE2_OFFICIAL_SAMPLER=true
  DRIVEVLA_SCORE_PROCESSES=8
  DRIVEVLA_SCORE_PARTITIONS=8
  DRIVEVLA_KILL_GPU_STRESS=1
  "${repo_root}/local_stage2/train_stage2_reproduction.sh"
  +trainer.params.enable_model_summary=false
  +trainer.params.log_every_n_steps=10
  trainer.params.max_epochs=1
  trainer.params.limit_train_batches=4000
  trainer.params.limit_val_batches=128
)
printf -v quoted_command '%q ' "${remote_command[@]}"
remote_pid="$(ssh -o BatchMode=yes "${remote_host}" \
  "cd $(printf '%q' "${repo_root}"); nohup setsid ${quoted_command} >> $(printf '%q' "${remote_log}") 2>&1 < /dev/null & printf '%s' \"\$!\"")"
if [[ ! "${remote_pid}" =~ ^[0-9]+$ ]]; then
  printf 'LAUNCH_FAILED invalid_pid=%q\n' "${remote_pid}" >&2
  exit 3
fi
printf '%s\n' "${remote_pid}" > "${output_dir}/remote_launcher.pid"
printf 'LAUNCH_OK timestamp=%s pid=%s runtime=%s output=%s\n' \
  "$(date -u +%FT%TZ)" "${remote_pid}" "${runtime}" "${output_dir}"
