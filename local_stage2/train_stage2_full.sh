#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export DRIVEVLA_SCORE_RAY=0
export DRIVEVLA_SCORE_PROCESSES="${DRIVEVLA_SCORE_PROCESSES:-16}"
export DRIVEVLA_SCORE_PARTITIONS="${DRIVEVLA_SCORE_PARTITIONS:-8}"
export DRIVEVLA_SCORE_START_METHOD="${DRIVEVLA_SCORE_START_METHOD:-forkserver}"
export DRIVEVLA_BIND_RANK_CPUS="${DRIVEVLA_BIND_RANK_CPUS:-1}"
export DRIVEVLA_SYNC_TRAIN_METRICS="${DRIVEVLA_SYNC_TRAIN_METRICS:-0}"
export DRIVEVLA_TRAIN_LOG_INTERVAL="${DRIVEVLA_TRAIN_LOG_INTERVAL:-10}"
export DRIVEVLA_FUSE_VALIDATION_SCORING="${DRIVEVLA_FUSE_VALIDATION_SCORING:-1}"
export DRIVEVLA_TIMING_INTERVAL="${DRIVEVLA_TIMING_INTERVAL:-100}"
export DRIVEVLA_STAGE1_CHECKPOINT="${DRIVEVLA_PUBLIC_BASE}"
export NAVSIM_TRAIN_METRIC_CACHE="${DRIVEVLA_NAVTRAIN_METRIC_CACHE}"
export INTERNVL_VERBOSE_DYNAMIC_BATCH=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# The machine's synthetic GPU pressure job can be relaunched while Lightning
# spends time constructing all DDP ranks.  The user explicitly authorized
# stopping only this script.  Keep a narrowly matched guard alive for the
# lifetime of training so it cannot race a rank onto an otherwise free GPU.
stress_guard_pid=""
if [[ "${DRIVEVLA_KILL_GPU_STRESS:-1}" == "1" ]]; then
  guard_gpu_stress() {
    while true; do
      mapfile -t stress_pids < <(pgrep -f '/mnt/project/gpu_stress[.]py' || true)
      if (( ${#stress_pids[@]} > 0 )); then
        targets=()
        for stress_pid in "${stress_pids[@]}"; do
          [[ -r "/proc/${stress_pid}/cmdline" ]] || continue
          stress_command="$(tr '\0' ' ' < "/proc/${stress_pid}/cmdline")"
          [[ "${stress_command}" == *"/mnt/project/gpu_stress.py"* ]] || continue
          targets+=("${stress_pid}")
          mapfile -t children < <(pgrep -P "${stress_pid}" || true)
          targets+=("${children[@]}")
        done
        if (( ${#targets[@]} > 0 )); then
          printf 'Stopping authorized gpu_stress.py processes: %s\n' "${targets[*]}"
          kill "${targets[@]}" 2>/dev/null || true
          sleep 1
          for target_pid in "${targets[@]}"; do
            if kill -0 "${target_pid}" 2>/dev/null; then
              kill -9 "${target_pid}" 2>/dev/null || true
            fi
          done
        fi
      fi
      sleep 2
    done
  }
  guard_gpu_stress &
  stress_guard_pid=$!
  trap 'kill "${stress_guard_pid}" 2>/dev/null || true' EXIT
fi

STAGE2_EXPERIMENT="${STAGE2_EXPERIMENT:-stage2_full_seed0}"
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-${DRIVEVLA_STAGE2_RUN_ROOT}/training/${STAGE2_EXPERIMENT}}"
STAGE2_SEED="${STAGE2_SEED:-0}"
STAGE2_MAX_EPOCHS="${STAGE2_MAX_EPOCHS:-27}"
STAGE2_TRAIN_CKPT="${STAGE2_TRAIN_CKPT:-}"

resume_args=(auto_resume=false train_ckpt_path=null)
if [[ -n "${STAGE2_TRAIN_CKPT}" ]]; then
  resume_args=(auto_resume=false "train_ckpt_path=${STAGE2_TRAIN_CKPT}")
fi

"${DRIVEVLA_PYTHON}" "${DRIVEVLA_REPO_ROOT}/navsim/planning/script/run_training_full.py" \
  train_test_split=navtrain \
  "experiment_name=${STAGE2_EXPERIMENT}" \
  "output_dir=${STAGE2_OUTPUT_DIR}" \
  "seed=${STAGE2_SEED}" \
  "cache_path=${DRIVEVLA_NAVTRAIN_FEATURE_CACHE}" \
  use_cache_without_dataset=true \
  preprocess_images_in_workers=true \
  preprocess_image_dtype=bfloat16 \
  pretokenize_inputs_in_workers=true \
  pad_datasets_to_global_batch=true \
  force_cache_computation=false \
  agent.checkpoint_path=null \
  "agent.stage1_checkpoint_path=${DRIVEVLA_PUBLIC_BASE}" \
  agent.cache_data=false \
  agent.vlm_config.freeze_backbone=true \
  agent.vlm_config.cache_hidden_state=false \
  agent.vlm_config.cache_mode=false \
  agent.vlm_config.initialize_from_config=true \
  agent.vlm_config.use_flash_attn=true \
  agent.vlm_config.extra_token_count=8 \
  agent.vlm_config.target_vocab_size=151682 \
  agent.lora_config.use_lora=true \
  agent.batch_size=2 \
  agent.num_gpus=8 \
  agent.lr_args.name=AdamW \
  agent.lr_args.base_lr=1e-4 \
  agent.lr_args.base_batch_size=16 \
  dataloader.params.batch_size=2 \
  dataloader.params.num_workers=4 \
  dataloader.params.prefetch_factor=4 \
  dataloader.params.persistent_workers=true \
  dataloader.params.multiprocessing_context=forkserver \
  trainer.params.devices=8 \
  trainer.params.strategy=ddp \
  trainer.params.precision=bf16-mixed \
  "trainer.params.max_epochs=${STAGE2_MAX_EPOCHS}" \
  trainer.params.check_val_every_n_epoch=1 \
  "${resume_args[@]}" \
  "$@"
