#!/usr/bin/env bash
# Direct V3 execution: codec training -> cache materialization -> formal training.
# Deliberately omits runtime/tail/cache validation, smoke runs and post-train gates.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

visible_device_count="$(python -c 'import torch; print(torch.cuda.device_count())')"
if ! [[ "$visible_device_count" =~ ^[1-9][0-9]*$ ]]; then
  echo "[vggt-v3-direct] no CUDA-compatible PPU is visible" >&2
  exit 2
fi

requested_local_processes="${LOCAL_NUM_PROCESSES:-$visible_device_count}"
topology_was_clamped=0
if (( requested_local_processes > visible_device_count )); then
  echo "[vggt-v3-direct] requested $requested_local_processes local processes, " \
       "but only $visible_device_count devices are visible; using all visible devices"
  requested_local_processes="$visible_device_count"
  topology_was_clamped=1
fi

export NUM_MACHINES=1
export MACHINE_RANK=0
export LOCAL_NUM_PROCESSES="$requested_local_processes"
export NUM_PROCESSES="$LOCAL_NUM_PROCESSES"
export VGGT_CODEC_NUM_PROCESSES="$LOCAL_NUM_PROCESSES"
export VGGT_CACHE_NUM_PROCESSES="$LOCAL_NUM_PROCESSES"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
target_effective_batch="${TARGET_EFFECTIVE_BATCH_SIZE:-32}"
per_step_batch=$((LOCAL_NUM_PROCESSES * PER_DEVICE_BATCH_SIZE))
if (( topology_was_clamped == 1 )) || [[ -z "${GRADIENT_ACCUMULATION_STEPS:-}" ]]; then
  if (( target_effective_batch % per_step_batch != 0 )); then
    echo "[vggt-v3-direct] target batch $target_effective_batch is not divisible by " \
         "$LOCAL_NUM_PROCESSES devices x batch $PER_DEVICE_BATCH_SIZE" >&2
    exit 2
  fi
  export GRADIENT_ACCUMULATION_STEPS=$((target_effective_batch / per_step_batch))
else
  export GRADIENT_ACCUMULATION_STEPS
  target_effective_batch=$((per_step_batch * GRADIENT_ACCUMULATION_STEPS))
fi
export TARGET_EFFECTIVE_BATCH_SIZE="$target_effective_batch"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
export TRAINING_LOGGING_FREQUENCY="${TRAINING_LOGGING_FREQUENCY:-50}"
export VGGT_CODEC_STEPS="${VGGT_CODEC_STEPS:-10000}"
export VGGT_DEBUG=0
export TRAINING_SKIP_FINAL_SAVE=0
export RUN_ID="${RUN_ID:-vggt-v3-layer11-global-codec-m195-${PAI_JOB_ID:-$(date +'%Y%m%d_%H%M%S')}}"

echo "[vggt-v3-direct] devices=$LOCAL_NUM_PROCESSES per_device_batch=$PER_DEVICE_BATCH_SIZE " \
     "gradient_accumulation=$GRADIENT_ACCUMULATION_STEPS effective_batch=$TARGET_EFFECTIVE_BATCH_SIZE"

if [[ "${V3_DIRECT_TOPOLOGY_ONLY:-0}" == "1" ]]; then
  exit 0
fi

bash "$project_root/tools/train_vggt_native_codec.sh"
bash "$project_root/tools/cache_vggt_v3_queries.sh"
exec bash "$project_root/8-train_vggt_v3_action.sh"
