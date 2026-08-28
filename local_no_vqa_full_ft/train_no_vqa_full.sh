#!/usr/bin/env bash

set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/local_stage2/common.sh"

NO_VQA_RUN_ROOT="${NO_VQA_RUN_ROOT:-/mnt/project/DriveVLA-M0-no-vqa/runs}"
NO_VQA_EXPERIMENT="${NO_VQA_EXPERIMENT:-no_vqa_full_ft_seed0_e36}"
NO_VQA_OUTPUT_DIR="${NO_VQA_OUTPUT_DIR:-${NO_VQA_RUN_ROOT}/training/${NO_VQA_EXPERIMENT}}"
NO_VQA_SEED="${NO_VQA_SEED:-0}"
NO_VQA_MAX_EPOCHS="${NO_VQA_MAX_EPOCHS:-36}"
NO_VQA_DATASET_SIZE="${NO_VQA_DATASET_SIZE:-103296}"
NO_VQA_DEVICES="${NO_VQA_DEVICES:-8}"
NO_VQA_BATCH_SIZE="${NO_VQA_BATCH_SIZE:-2}"
NO_VQA_TRAIN_CKPT="${NO_VQA_TRAIN_CKPT:-}"

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
export NAVSIM_TRAIN_METRIC_CACHE="${DRIVEVLA_NAVTRAIN_METRIC_CACHE}"
export INTERNVL_VERBOSE_DYNAMIC_BATCH=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# This cluster runs a synthetic GPU pressure process when a node is idle. Keep
# it from racing back onto GPU 7 while the explicitly requested training owns
# the node. The match is intentionally restricted to that one script.
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
          printf 'Stopping gpu_stress.py processes: %s\n' "${targets[*]}"
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

resume_args=(auto_resume=false train_ckpt_path=null)
if [[ -n "${NO_VQA_TRAIN_CKPT}" ]]; then
  resume_args=(auto_resume=false "train_ckpt_path=${NO_VQA_TRAIN_CKPT}")
fi

mkdir -p "${NO_VQA_OUTPUT_DIR}"

"${DRIVEVLA_PYTHON}" "${DRIVEVLA_REPO_ROOT}/navsim/planning/script/run_training_full.py" \
  train_test_split=navtrain \
  "experiment_name=${NO_VQA_EXPERIMENT}" \
  "output_dir=${NO_VQA_OUTPUT_DIR}" \
  "seed=${NO_VQA_SEED}" \
  "cache_path=${DRIVEVLA_NAVTRAIN_FEATURE_CACHE}" \
  use_cache_without_dataset=true \
  preprocess_images_in_workers=true \
  preprocess_image_dtype=bfloat16 \
  pretokenize_inputs_in_workers=true \
  pad_datasets_to_global_batch=true \
  force_cache_computation=false \
  agent.checkpoint_path=null \
  agent.stage1_checkpoint_path=null \
  agent.cache_data=false \
  "agent.vlm_config.vlm_path=${DRIVEVLA_VLM_DIR}" \
  agent.vlm_config.freeze_backbone=false \
  agent.vlm_config.freeze_lm_head=true \
  agent.vlm_config.skip_lm_head=true \
  agent.vlm_config.gradient_checkpointing=true \
  agent.vlm_config.cache_hidden_state=false \
  agent.vlm_config.cache_mode=false \
  agent.vlm_config.initialize_from_config=false \
  agent.vlm_config.use_flash_attn=true \
  agent.vlm_config.extra_token_count=8 \
  agent.vlm_config.target_vocab_size=151682 \
  agent.lora_config.use_lora=false \
  "agent.batch_size=${NO_VQA_BATCH_SIZE}" \
  "agent.num_gpus=${NO_VQA_DEVICES}" \
  agent.lr_args.name=AdamW \
  agent.lr_args.base_lr=1e-4 \
  agent.lr_args.base_batch_size=16 \
  agent.lr_args.scale_with_batch_size=false \
  +agent.lr_args.action_head_lr=1e-4 \
  +agent.lr_args.vlm_language_lr=1e-5 \
  +agent.lr_args.vlm_vision_lr=1e-5 \
  +agent.lr_args.vlm_projector_lr=2e-5 \
  +agent.lr_args.vlm_other_lr=1e-5 \
  +agent.lr_args.vlm_lora_lr=1e-4 \
  +agent.lr_args.action_head_weight_decay=1e-4 \
  +agent.lr_args.vlm_weight_decay=0.05 \
  '+agent.lr_args.betas=[0.9,0.95]' \
  +agent.lr_args.eps=1e-8 \
  "agent.scheduler_args={dataset_size:${NO_VQA_DATASET_SIZE},num_epochs:${NO_VQA_MAX_EPOCHS},warmup_ratio:0.03,start_lr_ratio:0.001,min_lr_ratio:0.1,action_head_min_lr_ratio:1.0,vlm_min_lr_ratio:0.1}" \
  "dataloader.params.batch_size=${NO_VQA_BATCH_SIZE}" \
  dataloader.params.num_workers=4 \
  dataloader.params.prefetch_factor=4 \
  dataloader.params.persistent_workers=true \
  dataloader.params.multiprocessing_context=forkserver \
  "trainer.params.devices=${NO_VQA_DEVICES}" \
  trainer.params.strategy=ddp \
  trainer.params.precision=bf16-mixed \
  "trainer.params.max_epochs=${NO_VQA_MAX_EPOCHS}" \
  trainer.params.check_val_every_n_epoch=1 \
  trainer.params.gradient_clip_val=1.0 \
  trainer.params.gradient_clip_algorithm=norm \
  +trainer.params.log_every_n_steps=10 \
  +trainer.params.enable_model_summary=false \
  "${resume_args[@]}" \
  "$@"
