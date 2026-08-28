#!/usr/bin/env bash

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DRIVEVLA_KILL_GPU_STRESS=0
export DRIVEVLA_SCORE_PROCESSES="${DRIVEVLA_SCORE_PROCESSES:-2}"
export DRIVEVLA_SCORE_PARTITIONS="${DRIVEVLA_SCORE_PARTITIONS:-2}"
export DRIVEVLA_BIND_RANK_CPUS=0
export NO_VQA_DEVICES=1
export NO_VQA_BATCH_SIZE=1
export NO_VQA_MAX_EPOCHS=1
export NO_VQA_EXPERIMENT="${NO_VQA_EXPERIMENT:-no_vqa_full_ft_smoke}"
export NO_VQA_OUTPUT_DIR="${NO_VQA_OUTPUT_DIR:-/mnt/project/DriveVLA-M0-no-vqa/runs/smoke/${NO_VQA_EXPERIMENT}}"

exec "${script_dir}/train_no_vqa_full.sh" \
  trainer.params.fast_dev_run=true \
  trainer.params.limit_train_batches=1 \
  trainer.params.limit_val_batches=1
