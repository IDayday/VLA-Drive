#!/usr/bin/env bash
# Train the independent DDP action-only planner while fine-tuning Qwen visual.
# No Wan/PPD/GS/reward/DINO/VGGT teacher or feature cache is used.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

# A visual-training run must read raw images. Inherited optional cache paths
# from another collaborator or branch are deliberately removed.
unset NAVSIM_FEATURE_CACHE_ROOT
unset NAVSIM_AGENT_DINO_CACHE_ROOT
unset NAVSIM_VGGT_CACHE_ROOT
export NAVSIM_USE_FEATURE_CACHE=0

num_machines="${NUM_MACHINES:-${WORLD_SIZE:-1}}"
machine_rank="${MACHINE_RANK:-${RANK:-0}}"
local_processes="${LOCAL_NUM_PROCESSES:-${NPROC_PER_NODE:-16}}"
num_processes="${NUM_PROCESSES:-$((num_machines * local_processes))}"
# Match the released DDP action-only optimization geometry by default. Visual
# activation checkpointing keeps the trainable vision tower within the PPU
# memory budget; callers can still opt into 1x2 through one-shot overrides.
per_device_batch="${PER_DEVICE_BATCH_SIZE:-2}"
gradient_accumulation="${GRADIENT_ACCUMULATION_STEPS:-1}"
target_effective_batch="${TARGET_EFFECTIVE_BATCH_SIZE:-32}"

for pair in \
  "NUM_MACHINES:$num_machines" \
  "MACHINE_RANK:$machine_rank" \
  "LOCAL_NUM_PROCESSES:$local_processes" \
  "NUM_PROCESSES:$num_processes" \
  "PER_DEVICE_BATCH_SIZE:$per_device_batch" \
  "GRADIENT_ACCUMULATION_STEPS:$gradient_accumulation" \
  "TARGET_EFFECTIVE_BATCH_SIZE:$target_effective_batch"; do
  variable="${pair%%:*}"
  value="${pair#*:}"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "$variable must be a non-negative integer, got: $value" >&2
    exit 2
  fi
done
if (( num_machines < 1 || local_processes < 1 || num_processes < 1 )); then
  echo "Training process counts must be positive" >&2
  exit 2
fi
if (( per_device_batch < 1 || gradient_accumulation < 1 )); then
  echo "Batch size and gradient accumulation must be positive" >&2
  exit 2
fi
if (( num_processes != num_machines * local_processes )); then
  echo "Invalid topology: global=$num_processes nodes=$num_machines local=$local_processes" >&2
  exit 2
fi
effective_batch=$((num_processes * per_device_batch * gradient_accumulation))
if (( effective_batch != target_effective_batch )); then
  echo "Refusing effective batch $effective_batch; expected $target_effective_batch" >&2
  echo "Formula: $num_processes x $per_device_batch x $gradient_accumulation" >&2
  exit 2
fi

split="${SPLIT:-train}"
datalist="${NAVSIM_DATALIST_PATH:-$DRIVEDREAMER_ROOT/${split}_meta.json}"
base_config="${TRAIN_CONFIG_YAML:-$DRIVEDREAMER_ROOT/starVLA/config/training/cfg_yaw_1225.yaml}"
experiment_config="$DRIVEDREAMER_ROOT/starVLA/config/training/qwen_visual_action_only.yaml"
accelerate_config="${TRAIN_ACCELERATE_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}"
max_train_steps="${MAX_TRAIN_STEPS:-100000}"
warmup_steps="${NUM_WARMUP_STEPS:-5000}"
save_interval="${SAVE_INTERVAL:-10000}"
logging_frequency="${TRAINING_LOGGING_FREQUENCY:-50}"
qwen_learning_rate="${QWEN_LEARNING_RATE:-1e-5}"
visual_learning_rate="${VISUAL_LEARNING_RATE:-2e-6}"
action_learning_rate="${ACTION_LEARNING_RATE:-1e-5}"
weight_decay="${OPTIMIZER_WEIGHT_DECAY:-1e-3}"
timestamp="$(date +'%Y%m%d_%H%M%S')"
run_id="${RUN_ID:-qwen-visual-action-only-${PAI_JOB_ID:-$timestamp}}"

visible_devices="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "$visible_devices" ]]; then
  for ((device_index = 0; device_index < local_processes; device_index++)); do
    [[ -n "$visible_devices" ]] && visible_devices+=","
    visible_devices+="$device_index"
  done
fi

launch_args=(
  --main_process_port "${MAIN_PROCESS_PORT:-${MASTER_PORT:-29721}}"
  --config_file "$accelerate_config"
  --num_processes "$num_processes"
  --num_machines "$num_machines"
  --machine_rank "$machine_rank"
  --mixed_precision bf16
)
if (( num_machines > 1 )); then
  launch_args+=(
    --main_process_ip "${MAIN_PROCESS_IP:-${MASTER_ADDR:?Set MASTER_ADDR for multi-node training}}"
    --same_network
  )
fi

training_args=(
  --config_yaml "$base_config"
  --config_overlay "$experiment_config"
  --framework.qwenvl.base_vlm "$BASE_VLM"
  --framework.qwenvl.attn_implementation "${QWEN_VISUAL_ATTN_IMPLEMENTATION:-sdpa}"
  --run_root_dir "$NAVSIM_EXP_ROOT"
  --run_id "$run_id"
  --wandb_project "$WANDB_PROJECT"
  --wandb_entity "$WANDB_ENTITY"
  --datasets.vla_data.datalist_path "$datalist"
  --datasets.vla_data.data_root "$DATA_ROOT"
  --datasets.vla_data.split "$split"
  --datasets.vla_data.per_device_batch_size "$per_device_batch"
  --trainer.gradient_accumulation_steps "$gradient_accumulation"
  --trainer.max_train_steps "$max_train_steps"
  --trainer.num_warmup_steps "$warmup_steps"
  --trainer.save_interval "$save_interval"
  --trainer.logging_frequency "$logging_frequency"
  --trainer.optimizer.weight_decay "$weight_decay"
  --trainer.learning_rate.base "$qwen_learning_rate"
  --trainer.learning_rate.qwen_vl_interface "$qwen_learning_rate"
  --trainer.learning_rate.qwen_visual "$visual_learning_rate"
  --trainer.learning_rate.action_model "$action_learning_rate"
  --framework.action_model.repeated_diffusion_steps "${FM_REPEAT:-8}"
  --framework.action_model.hidden_size "${ACTION_HIDDEN_SIZE:-1536}"
  --framework.action_model.diffusion_model_cfg.cross_attention_dim "${ACTION_HIDDEN_SIZE:-1536}"
  --framework.action_model.diffusion_model_cfg.output_dim "${ACTION_HIDDEN_SIZE:-1536}"
  --framework.action_model.diffusion_model_cfg.num_layers "${ACTION_LAYERS:-24}"
)

echo "[qwen-visual] project=$DRIVEDREAMER_ROOT run_id=$run_id"
echo "[qwen-visual] topology=nodes:$num_machines rank:$machine_rank local:$local_processes global:$num_processes"
echo "[qwen-visual] effective_batch=$effective_batch (per_device=$per_device_batch accumulation=$gradient_accumulation)"
echo "[qwen-visual] lr=qwen:$qwen_learning_rate visual:$visual_learning_rate action:$action_learning_rate"
echo "[qwen-visual] auxiliary_models=disabled feature_caches=disabled attention=${QWEN_VISUAL_ATTN_IMPLEMENTATION:-sdpa}"

if [[ "${QWEN_VISUAL_TUNE_DRY_RUN:-0}" == "1" ]]; then
  printf '[qwen-visual] DRY-RUN:'
  printf ' %q' env CUDA_VISIBLE_DEVICES="$visible_devices" accelerate launch "${launch_args[@]}" starVLA/training/train_starvla.py "${training_args[@]}"
  printf '\n'
  exit 0
fi

required_paths=(
  "$BASE_VLM/config.json"
  "$base_config"
  "$experiment_config"
  "$datalist"
  "$DATA_ROOT/meta/$split"
)
for required_path in "${required_paths[@]}"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Missing action-only visual-training asset: $required_path" >&2
    exit 2
  fi
done
run_dir="$NAVSIM_EXP_ROOT/$run_id"
if [[ -e "$run_dir" ]]; then
  echo "Refusing to overwrite existing experiment: $run_dir" >&2
  exit 2
fi

if [[ "${QWEN_VISUAL_RUN_SMOKE_BEFORE_FORMAL:-1}" == "1" && "${QWEN_VISUAL_SMOKE_ACTIVE:-0}" != "1" ]]; then
  smoke_run_id="${run_id}-smoke"
  smoke_dir="$NAVSIM_EXP_ROOT/$smoke_run_id"
  smoke_marker="$smoke_dir/.qwen_visual_smoke_complete"
  if [[ -f "$smoke_marker" ]]; then
    echo "[qwen-visual] prior forward/backward smoke PASS: $smoke_marker"
  else
    if [[ -e "$smoke_dir" ]]; then
      echo "Incomplete visual-training smoke directory exists: $smoke_dir" >&2
      echo "Move it aside or choose a new RUN_ID; this launcher will not overwrite it." >&2
      exit 2
    fi
    echo "[qwen-visual] starting 2-step full-gradient smoke: $smoke_run_id"
    env \
      QWEN_VISUAL_SMOKE_ACTIVE=1 \
      QWEN_VISUAL_RUN_SMOKE_BEFORE_FORMAL=0 \
      RUN_ID="$smoke_run_id" \
      MAX_TRAIN_STEPS=2 \
      NUM_WARMUP_STEPS=2 \
      SAVE_INTERVAL=999999 \
      TRAINING_LOGGING_FREQUENCY=1 \
      TRAINING_SKIP_FINAL_SAVE=1 \
      bash "$DRIVEDREAMER_ROOT/8-train_action-only-qwen-visual.sh"
    touch "$smoke_marker"
    echo "[qwen-visual] full-gradient smoke PASS; starting formal training"
  fi
elif [[ "${QWEN_VISUAL_RUN_SMOKE_BEFORE_FORMAL:-1}" != "0" && "${QWEN_VISUAL_RUN_SMOKE_BEFORE_FORMAL:-1}" != "1" ]]; then
  echo "QWEN_VISUAL_RUN_SMOKE_BEFORE_FORMAL must be 0 or 1" >&2
  exit 2
fi

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TRITON_LOCAL_CACHE_ROOT:-/tmp/drivedreamer-triton}/${run_id}/node${machine_rank}}"
mkdir -p "$TRITON_CACHE_DIR"

set -x
CUDA_VISIBLE_DEVICES="$visible_devices" accelerate launch \
  "${launch_args[@]}" \
  starVLA/training/train_starvla.py \
  "${training_args[@]}"
