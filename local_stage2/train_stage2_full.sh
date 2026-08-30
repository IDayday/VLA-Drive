#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# Record the actual training stack before allocating GPUs.  Host-local virtual
# environments can live at the same absolute path while containing different
# Lightning versions, so the executable path alone is not an environment lock.
stage2_runtime="$("${DRIVEVLA_PYTHON}" - <<'PY'
import json
import pytorch_lightning
import torch

print(json.dumps({
    "pytorch_lightning": pytorch_lightning.__version__,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
}))
PY
)"
printf 'STAGE2_RUNTIME %s\n' "${stage2_runtime}"

required_lightning="${STAGE2_REQUIRE_LIGHTNING_VERSION:-}"
if [[ -n "${required_lightning}" ]]; then
  actual_lightning="$("${DRIVEVLA_PYTHON}" -c \
    'import pytorch_lightning; print(pytorch_lightning.__version__)')"
  if [[ "${actual_lightning}" != "${required_lightning}" ]]; then
    echo "Required pytorch-lightning ${required_lightning}, found ${actual_lightning}" >&2
    exit 2
  fi
fi

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

stress_guard_pid=""

cleanup() {
  status=$?
  if [[ -n "${stress_guard_pid}" ]]; then
    kill "${stress_guard_pid}" 2>/dev/null || true
  fi
  printf 'TRAIN_LAUNCH_EXIT timestamp=%s status=%d\n' \
    "$(date -u +%FT%TZ)" "${status}"
}

handle_signal() {
  signal_name="$1"
  exit_status="$2"
  printf 'TRAIN_LAUNCH_SIGNAL timestamp=%s signal=%s\n' \
    "$(date -u +%FT%TZ)" "${signal_name}" >&2
  exit "${exit_status}"
}

trap cleanup EXIT
trap 'handle_signal HUP 129' HUP
trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM

# The machine's synthetic GPU pressure job can be relaunched while Lightning
# spends time constructing all DDP ranks.  The user explicitly authorized
# stopping only this script.  Keep a narrowly matched guard alive for the
# lifetime of training so it cannot race a rank onto an otherwise free GPU.
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
fi

STAGE2_EXPERIMENT="${STAGE2_EXPERIMENT:-stage2_full_seed0}"
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-${DRIVEVLA_STAGE2_RUN_ROOT}/training/${STAGE2_EXPERIMENT}}"
STAGE2_SEED="${STAGE2_SEED:-0}"
STAGE2_MAX_EPOCHS="${STAGE2_MAX_EPOCHS:-27}"
STAGE2_TRAIN_CKPT="${STAGE2_TRAIN_CKPT:-}"
STAGE2_NUM_GPUS="${STAGE2_NUM_GPUS:-8}"
STAGE2_NUM_NODES="${STAGE2_NUM_NODES:-1}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-2}"
STAGE2_ACCUMULATE_GRAD_BATCHES="${STAGE2_ACCUMULATE_GRAD_BATCHES:-1}"
STAGE2_EFFECTIVE_GLOBAL_BATCH_SIZE="${STAGE2_EFFECTIVE_GLOBAL_BATCH_SIZE:-16}"
STAGE2_BASE_LR="${STAGE2_BASE_LR:-1e-4}"
STAGE2_BASE_BATCH_SIZE="${STAGE2_BASE_BATCH_SIZE:-16}"
STAGE2_SCHEDULER="${STAGE2_SCHEDULER:-none}"
STAGE2_SCHEDULER_DATASET_SIZE="${STAGE2_SCHEDULER_DATASET_SIZE:-103288}"
STAGE2_SCHEDULER_WARMUP_RATIO="${STAGE2_SCHEDULER_WARMUP_RATIO:-0.1}"
STAGE2_SCHEDULER_MIN_LR_RATIO="${STAGE2_SCHEDULER_MIN_LR_RATIO:-0.0}"
STAGE2_SCHEDULER_START_LR_RATIO="${STAGE2_SCHEDULER_START_LR_RATIO:-1e-6}"
STAGE2_FLASH_ATTENTION="${STAGE2_FLASH_ATTENTION:-true}"
STAGE2_FROZEN_BACKBONE_MODE="${STAGE2_FROZEN_BACKBONE_MODE:-eval}"
STAGE2_DECAY_NORM_AND_BIAS="${STAGE2_DECAY_NORM_AND_BIAS:-false}"
STAGE2_PREPAD_DATASET="${STAGE2_PREPAD_DATASET:-true}"
STAGE2_OFFICIAL_SAMPLER="${STAGE2_OFFICIAL_SAMPLER:-false}"

scheduler_args=(agent.scheduler_args=null)
case "${STAGE2_SCHEDULER}" in
  none)
    ;;
  source_cosine|released_cosine)
    scheduler_args=(
      "agent.scheduler_args={dataset_size:${STAGE2_SCHEDULER_DATASET_SIZE},num_epochs:${STAGE2_MAX_EPOCHS},warmup_ratio:${STAGE2_SCHEDULER_WARMUP_RATIO},min_lr_ratio:${STAGE2_SCHEDULER_MIN_LR_RATIO},action_head_min_lr_ratio:${STAGE2_SCHEDULER_MIN_LR_RATIO},vlm_min_lr_ratio:${STAGE2_SCHEDULER_MIN_LR_RATIO},start_lr_ratio:${STAGE2_SCHEDULER_START_LR_RATIO}}"
    )
    ;;
  *)
    echo "Unsupported STAGE2_SCHEDULER=${STAGE2_SCHEDULER}; expected none or source_cosine" >&2
    exit 2
    ;;
esac

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
  "pad_datasets_to_global_batch=${STAGE2_PREPAD_DATASET}" \
  "official_stage2_sampler=${STAGE2_OFFICIAL_SAMPLER}" \
  official_stage2_reference_global_batch_size=16 \
  force_cache_computation=false \
  agent.checkpoint_path=null \
  "agent.stage1_checkpoint_path=${DRIVEVLA_PUBLIC_BASE}" \
  agent.cache_data=false \
  agent.vlm_config.freeze_backbone=true \
  agent.vlm_config.cache_hidden_state=false \
  agent.vlm_config.cache_mode=false \
  agent.vlm_config.initialize_from_config=true \
  "agent.vlm_config.use_flash_attn=${STAGE2_FLASH_ATTENTION}" \
  "agent.vlm_config.frozen_backbone_mode=${STAGE2_FROZEN_BACKBONE_MODE}" \
  agent.vlm_config.extra_token_count=8 \
  agent.vlm_config.target_vocab_size=151682 \
  agent.lora_config.use_lora=true \
  "agent.batch_size=${STAGE2_BATCH_SIZE}" \
  "agent.num_gpus=${STAGE2_NUM_GPUS}" \
  agent.lr_args.name=AdamW \
  "agent.lr_args.base_lr=${STAGE2_BASE_LR}" \
  "agent.lr_args.base_batch_size=${STAGE2_BASE_BATCH_SIZE}" \
  "agent.lr_args.effective_global_batch_size=${STAGE2_EFFECTIVE_GLOBAL_BATCH_SIZE}" \
  "agent.lr_args.decay_norm_and_bias=${STAGE2_DECAY_NORM_AND_BIAS}" \
  "${scheduler_args[@]}" \
  "dataloader.params.batch_size=${STAGE2_BATCH_SIZE}" \
  dataloader.params.num_workers=4 \
  dataloader.params.prefetch_factor=4 \
  dataloader.params.persistent_workers=true \
  dataloader.params.multiprocessing_context=forkserver \
  "trainer.params.devices=${STAGE2_NUM_GPUS}" \
  "trainer.params.num_nodes=${STAGE2_NUM_NODES}" \
  trainer.params.strategy=ddp \
  trainer.params.precision=bf16-mixed \
  "trainer.params.accumulate_grad_batches=${STAGE2_ACCUMULATE_GRAD_BATCHES}" \
  "trainer.params.max_epochs=${STAGE2_MAX_EPOCHS}" \
  trainer.params.check_val_every_n_epoch=1 \
  "${resume_args[@]}" \
  "$@"
