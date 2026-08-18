#!/usr/bin/env bash
# Train the independent dense VGGT bottleneck while keeping the Action DiT API unchanged.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

# Capability isolation: this route only needs the standard action VLM and its
# own dense offline cache. It never imports VGGT or unrelated cached teachers.
unset NAVSIM_AGENT_DINO_CACHE_ROOT
unset NAVSIM_VGGT_CACHE_ROOT
unset NAVSIM_FEATURE_CACHE_ROOT
export NAVSIM_USE_FEATURE_CACHE=0

# Resolve explicit path overrides before checking the selected capability.
effective_base_vlm="$BASE_VLM"
effective_dense_cache_root="$NAVSIM_VGGT_DENSE_CACHE_ROOT"
effective_action_checkpoint="${ACTION_ONLY_CHECKPOINT:-}"
effective_run_root="$NAVSIM_EXP_ROOT"
effective_run_id="${RUN_ID:-}"
cli_arguments=("$@")
for ((argument_index = 0; argument_index < ${#cli_arguments[@]}; argument_index++)); do
  argument="${cli_arguments[$argument_index]}"
  case "$argument" in
    --framework.qwenvl.base_vlm)
      ((argument_index += 1))
      effective_base_vlm="${cli_arguments[$argument_index]:-}"
      ;;
    --framework.qwenvl.base_vlm=*) effective_base_vlm="${argument#*=}" ;;
    --framework.vggt_bottleneck.cache.root)
      ((argument_index += 1))
      effective_dense_cache_root="${cli_arguments[$argument_index]:-}"
      ;;
    --framework.vggt_bottleneck.cache.root=*)
      effective_dense_cache_root="${argument#*=}"
      ;;
    --trainer.pretrained_checkpoint)
      ((argument_index += 1))
      effective_action_checkpoint="${cli_arguments[$argument_index]:-}"
      ;;
    --trainer.pretrained_checkpoint=*) effective_action_checkpoint="${argument#*=}" ;;
    --run_root_dir)
      ((argument_index += 1))
      effective_run_root="${cli_arguments[$argument_index]:-}"
      ;;
    --run_root_dir=*) effective_run_root="${argument#*=}" ;;
    --run_id)
      ((argument_index += 1))
      effective_run_id="${cli_arguments[$argument_index]:-}"
      ;;
    --run_id=*) effective_run_id="${argument#*=}" ;;
  esac
done

if [[ ! -d "$effective_base_vlm" ]]; then
  echo "Missing standard action VLM: $effective_base_vlm" >&2
  exit 2
fi
if [[ ! -f "$effective_dense_cache_root/vggt_dense/manifest.json" ]]; then
  echo "Missing dense VGGT cache manifest under: $effective_dense_cache_root" >&2
  echo "Run 11-precompute_vggt_dense_cache.sh first." >&2
  exit 2
fi

num_machines="${NUM_MACHINES:-${WORLD_SIZE:-1}}"
machine_rank="${MACHINE_RANK:-${RANK:-0}}"
local_num_processes="${LOCAL_NUM_PROCESSES:-${NPROC_PER_NODE:-16}}"
num_processes="${NUM_PROCESSES:-$((num_machines * local_num_processes))}"
batch_size="${PER_DEVICE_BATCH_SIZE:-2}"
gradient_accumulation="${GRADIENT_ACCUMULATION_STEPS:-1}"
target_effective_batch="${TARGET_EFFECTIVE_BATCH_SIZE:-32}"
effective_batch=$((num_processes * batch_size * gradient_accumulation))
if (( num_processes != num_machines * local_num_processes )); then
  echo "Invalid topology: processes $num_processes != nodes $num_machines x local $local_num_processes" >&2
  exit 2
fi
if (( effective_batch != target_effective_batch )); then
  echo "Refusing effective batch $effective_batch; expected $target_effective_batch" >&2
  exit 2
fi

split="${SPLIT:-train}"
datalist="${NAVSIM_DATALIST_PATH:-$DRIVEDREAMER_ROOT/${split}_meta.json}"
base_config="${TRAIN_CONFIG_YAML:-$DRIVEDREAMER_ROOT/starVLA/config/training/cfg_yaw_1225.yaml}"
overlay="$DRIVEDREAMER_ROOT/starVLA/config/training/vggt_dense_bottleneck.yaml"
accelerate_config="${TRAIN_ACCELERATE_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}"
max_train_steps="${MAX_TRAIN_STEPS:-100000}"
warmup_steps="${NUM_WARMUP_STEPS:-5000}"
save_interval="${SAVE_INTERVAL:-5000}"
logging_frequency="${TRAINING_LOGGING_FREQUENCY:-50}"
expected_train_samples="${EXPECTED_TRAIN_SAMPLES:-}"
attention_implementation="${VGGT_DENSE_VLM_ATTN_IMPLEMENTATION:-${VGGT_VLM_ATTN_IMPLEMENTATION:-sdpa}}"
timestamp="$(date +'%Y%m%d_%H%M%S')"
run_id="${RUN_ID:-vggt-dense-bottleneck-${PAI_JOB_ID:-$timestamp}}"
effective_run_id="${effective_run_id:-$run_id}"
for pair in \
  "MAX_TRAIN_STEPS:$max_train_steps" \
  "NUM_WARMUP_STEPS:$warmup_steps" \
  "SAVE_INTERVAL:$save_interval" \
  "TRAINING_LOGGING_FREQUENCY:$logging_frequency"; do
  variable="${pair%%:*}"
  value="${pair#*:}"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$variable must be a positive integer, got: $value" >&2
    exit 2
  fi
done
if [[ -n "$expected_train_samples" ]] && ! [[ "$expected_train_samples" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXPECTED_TRAIN_SAMPLES must be a positive integer, got: $expected_train_samples" >&2
  exit 2
fi
visible_devices="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "$visible_devices" ]]; then
  for ((device_index = 0; device_index < local_num_processes; device_index++)); do
    [[ -n "$visible_devices" ]] && visible_devices+=","
    visible_devices+="$device_index"
  done
fi

launch_args=(
  --main_process_port "${MAIN_PROCESS_PORT:-${MASTER_PORT:-29691}}"
  --config_file "$accelerate_config"
  --num_processes "$num_processes"
  --num_machines "$num_machines"
  --machine_rank "$machine_rank"
  --mixed_precision bf16
)
if (( num_machines > 1 )); then
  launch_args+=(
    --main_process_ip "${MAIN_PROCESS_IP:-${MASTER_ADDR:?DLC must provide MASTER_ADDR}}"
    --same_network
  )
fi

training_args=(
  --config_yaml "$base_config"
  --config_overlay "$overlay"
  --framework.qwenvl.base_vlm "$BASE_VLM"
  --framework.qwenvl.attn_implementation "$attention_implementation"
  --framework.vggt_bottleneck.cache.root "$NAVSIM_VGGT_DENSE_CACHE_ROOT"
  --run_root_dir "$NAVSIM_EXP_ROOT"
  --run_id "$run_id"
  --wandb_project "$WANDB_PROJECT"
  --wandb_entity "$WANDB_ENTITY"
  --datasets.vla_data.datalist_path "$datalist"
  --datasets.vla_data.data_root "$DATA_ROOT"
  --datasets.vla_data.split "$split"
  --datasets.vla_data.per_device_batch_size "$batch_size"
  --datasets.vla_data.load_act_data 1
  --datasets.video_data.load_2d_data 0
  --datasets.gs_data.load_3d_data 0
  --datasets.reward_data.load_reward_data 0
  --w_depth 0
  --enable_image_aug 0
  --trainer.gradient_accumulation_steps "$gradient_accumulation"
  --trainer.max_train_steps "$max_train_steps"
  --trainer.num_warmup_steps "$warmup_steps"
  --trainer.save_interval "$save_interval"
  --trainer.logging_frequency "$logging_frequency"
  --trainer.optimizer.weight_decay "${OPTIMIZER_WEIGHT_DECAY:-1e-3}"
  --trainer.learning_rate.base "${BASE_LEARNING_RATE:-1e-5}"
  --trainer.learning_rate.action_model "${ACTION_LEARNING_RATE:-1e-5}"
  --trainer.learning_rate.vggt_dense_bottleneck "${VGGT_DENSE_LEARNING_RATE:-5e-5}"
  --trainer.learning_rate.vggt_bottleneck_aux_plan_head "${VGGT_DENSE_LEARNING_RATE:-5e-5}"
  --framework.vggt_bottleneck.diagnostics.intervention_interval "${VGGT_DENSE_INTERVENTION_INTERVAL:-500}"
  --framework.action_model.repeated_diffusion_steps "${FM_REPEAT:-8}"
  --framework.action_model.hidden_size "${ACTION_HIDDEN_SIZE:-1536}"
  --framework.action_model.diffusion_model_cfg.cross_attention_dim "${ACTION_HIDDEN_SIZE:-1536}"
  --framework.action_model.diffusion_model_cfg.output_dim "${ACTION_HIDDEN_SIZE:-1536}"
  --framework.action_model.diffusion_model_cfg.num_layers "${ACTION_LAYERS:-24}"
)
if [[ -n "${ACTION_ONLY_CHECKPOINT:-}" ]]; then
  training_args+=(--trainer.pretrained_checkpoint "$ACTION_ONLY_CHECKPOINT")
fi
if [[ -n "$expected_train_samples" ]]; then
  training_args+=(--datasets.vla_data.expected_sample_count "$expected_train_samples")
fi
# User CLI is appended last and overrides environment-derived arguments.
training_args+=("$@")

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TRITON_LOCAL_CACHE_ROOT:-/tmp/drivedreamer-triton}/${run_id}/node${machine_rank}}"
echo "Dense VGGT topology: nodes=$num_machines node_rank=$machine_rank local_ppus=$local_num_processes global_processes=$num_processes"
echo "Dense VGGT effective batch: $effective_batch (target=$target_effective_batch)"
echo "Dense VGGT optimization: max_steps=$max_train_steps warmup=$warmup_steps save_every=$save_interval log_every=$logging_frequency"
echo "Dense VGGT model: attention=$attention_implementation action_hidden=${ACTION_HIDDEN_SIZE:-1536} action_layers=${ACTION_LAYERS:-24} fm_repeat=${FM_REPEAT:-8}"

if [[ "${VGGT_DENSE_TRAIN_DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' CUDA_VISIBLE_DEVICES="$visible_devices" accelerate launch "${launch_args[@]}" starVLA/training/train_starvla.py "${training_args[@]}"
  printf '\n'
  exit 0
fi

if [[ -n "$effective_action_checkpoint" && ! -f "$effective_action_checkpoint" ]]; then
  echo "Missing action-only initialization checkpoint: $effective_action_checkpoint" >&2
  exit 2
fi
run_dir="$effective_run_root/$effective_run_id"
if [[ -e "$run_dir" ]]; then
  echo "Refusing to overwrite existing dense VGGT experiment: $run_dir" >&2
  exit 2
fi
mkdir -p "$TRITON_CACHE_DIR"

set -x
CUDA_VISIBLE_DEVICES="$visible_devices" accelerate launch \
  "${launch_args[@]}" \
  starVLA/training/train_starvla.py \
  "${training_args[@]}"
