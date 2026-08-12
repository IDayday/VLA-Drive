#!/usr/bin/env bash
# Full end-to-end VGGT-query planner training; no baseline planner/draft needed.

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

# Capability isolation: a collaborator may configure an agent-DINO cache in
# env.local.sh for feature/add-agent-query. QwenOFT_VGGT uses the minimal
# prompt and must not discover or validate that unrelated optional cache.
unset NAVSIM_AGENT_DINO_CACHE_ROOT
unset NAVSIM_FEATURE_CACHE_ROOT
export NAVSIM_USE_FEATURE_CACHE=0

if [[ ! -d "$VGGT_BASE_VLM" ]]; then
  echo "Missing VGGT-token VLM: $VGGT_BASE_VLM. Run 7-add_vggt_tokens.sh first." >&2
  exit 2
fi
vggt_require_teacher_cache="${VGGT_REQUIRE_TEACHER_CACHE:-1}"
if [[ "$vggt_require_teacher_cache" != "0" && "$vggt_require_teacher_cache" != "1" ]]; then
  echo "VGGT_REQUIRE_TEACHER_CACHE must be 0 or 1, got: $vggt_require_teacher_cache" >&2
  exit 2
fi
if [[ "$vggt_require_teacher_cache" == "1" && ! -f "$NAVSIM_VGGT_CACHE_ROOT/vggt_query/manifest.json" ]]; then
  echo "Missing VGGT query cache manifest under: $NAVSIM_VGGT_CACHE_ROOT" >&2
  echo "Run tools/cache_vggt_queries.sh first." >&2
  exit 2
fi

experiment_overlay="${VGGT_EXPERIMENT_OVERLAY:-}"
if [[ -n "$experiment_overlay" && ! -f "$experiment_overlay" ]]; then
  echo "Missing VGGT experiment overlay: $experiment_overlay" >&2
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
  echo "Invalid topology: global processes $num_processes != nodes $num_machines x local processes $local_num_processes" >&2
  exit 2
fi
if (( effective_batch != target_effective_batch )); then
  echo "Refusing effective batch $effective_batch; expected $target_effective_batch" >&2
  exit 2
fi
split="${SPLIT:-train}"
datalist="${NAVSIM_DATALIST_PATH:-$DRIVEDREAMER_ROOT/${split}_meta.json}"
base_config="${TRAIN_CONFIG_YAML:-$DRIVEDREAMER_ROOT/starVLA/config/training/cfg_yaw_1225.yaml}"
main_overlay="$DRIVEDREAMER_ROOT/starVLA/config/training/vggt_query_main.yaml"
debug_overlay="$DRIVEDREAMER_ROOT/starVLA/config/training/vggt_query_debug.yaml"
accelerate_config="${TRAIN_ACCELERATE_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}"
timestamp="$(date +"%m%d_%H")"
run_id="${RUN_ID:-${timestamp}-vggt-query-v2-layer11-global-bz_${batch_size}-ga_${gradient_accumulation}-${split}}"
visible_devices="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "$visible_devices" ]]; then
  for ((device_index = 0; device_index < local_num_processes; device_index++)); do
    [[ -n "$visible_devices" ]] && visible_devices+=","
    visible_devices+="$device_index"
  done
fi

launch_args=(
  --main_process_port "${MAIN_PROCESS_PORT:-${MASTER_PORT:-29689}}"
  --config_file "$accelerate_config"
  --num_processes "$num_processes"
  --num_machines "$num_machines"
  --machine_rank "$machine_rank"
  --mixed_precision bf16
)
if (( num_machines > 1 )); then
  launch_args+=(--main_process_ip "${MAIN_PROCESS_IP:-${MASTER_ADDR:?DLC must provide MASTER_ADDR for multi-node training}}" --same_network)
fi

training_args=(
  --config_yaml "$base_config"
  --config_overlay "$main_overlay"
  --framework.qwenvl.base_vlm "$VGGT_BASE_VLM"
  --framework.qwenvl.attn_implementation "${VLM_ATTN_IMPLEMENTATION:-flash_attention_2}"
  --framework.vggt.cache.root "$NAVSIM_VGGT_CACHE_ROOT"
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
  --rgb_query_loss 0
  --gs_query_loss 0
  --trainer.gradient_accumulation_steps "$gradient_accumulation"
  --trainer.optimizer.weight_decay "${OPTIMIZER_WEIGHT_DECAY:-1e-3}"
  --trainer.learning_rate.base "${BASE_LEARNING_RATE:-1e-5}"
  --trainer.learning_rate.qwen_vl_interface "${QWEN_LEARNING_RATE:-1e-5}"
  --trainer.learning_rate.action_model "${ACTION_LEARNING_RATE:-1e-5}"
  --trainer.learning_rate.vggt_geometry_adapter "${VGGT_LEARNING_RATE:-3e-5}"
  --trainer.learning_rate.vggt_aligner "${VGGT_LEARNING_RATE:-3e-5}"
  --trainer.learning_rate.vggt_waypoint_reader "${VGGT_LEARNING_RATE:-3e-5}"
  --trainer.learning_rate.vggt_geometry_probe "${VGGT_LEARNING_RATE:-3e-5}"
  --trainer.learning_rate.vggt_aux_plan_head "${VGGT_LEARNING_RATE:-3e-5}"
  --framework.action_model.repeated_diffusion_steps "${FM_REPEAT:-8}"
  --framework.action_model.hidden_size "${ACTION_HIDDEN_SIZE:-1536}"
  --framework.action_model.diffusion_model_cfg.cross_attention_dim "${ACTION_HIDDEN_SIZE:-1536}"
  --framework.action_model.diffusion_model_cfg.output_dim "${ACTION_HIDDEN_SIZE:-1536}"
  --framework.action_model.diffusion_model_cfg.num_layers "${ACTION_LAYERS:-24}"
)
if [[ -n "$experiment_overlay" ]]; then
  training_args+=(--config_overlay "$experiment_overlay")
fi
if [[ "${VGGT_DEBUG:-0}" == "1" ]]; then
  training_args+=(--config_overlay "$debug_overlay")
fi
intervention_interval="${VGGT_INTERVENTION_INTERVAL:-}"
if [[ -z "$intervention_interval" && "${VGGT_DEBUG:-0}" == "1" && -z "$experiment_overlay" ]]; then
  # The main V2 smoke must exercise the same intervention/predict_action path
  # used every 500 formal steps. Controls retain their explicit interval=0.
  intervention_interval=1
fi
if [[ -n "$intervention_interval" ]]; then
  if ! [[ "$intervention_interval" =~ ^[0-9]+$ ]]; then
    echo "VGGT_INTERVENTION_INTERVAL must be a non-negative integer, got: $intervention_interval" >&2
    exit 2
  fi
  training_args+=(
    --framework.vggt.diagnostics.intervention_interval "$intervention_interval"
  )
fi
if [[ -n "${MAX_TRAIN_STEPS:-}" ]]; then
  training_args+=(--trainer.max_train_steps "$MAX_TRAIN_STEPS")
fi
if [[ -n "${NUM_WARMUP_STEPS:-}" ]]; then
  training_args+=(--trainer.num_warmup_steps "$NUM_WARMUP_STEPS")
fi
if [[ -n "${SAVE_INTERVAL:-}" ]]; then
  training_args+=(--trainer.save_interval "$SAVE_INTERVAL")
fi
if [[ -n "${TRAINING_LOGGING_FREQUENCY:-}" ]]; then
  training_args+=(--trainer.logging_frequency "$TRAINING_LOGGING_FREQUENCY")
fi

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TRITON_LOCAL_CACHE_ROOT:-/tmp/drivedreamer-triton}/${run_id}/node${machine_rank}}"
mkdir -p "$TRITON_CACHE_DIR"
echo "VGGT training topology: nodes=$num_machines node_rank=$machine_rank local_ppus=$local_num_processes global_processes=$num_processes"
echo "VGGT effective batch: $effective_batch (target=$target_effective_batch)"
echo "VGGT optimization: max_steps=${MAX_TRAIN_STEPS:-100000} base_lr=${BASE_LEARNING_RATE:-1e-5} action_lr=${ACTION_LEARNING_RATE:-1e-5} vggt_lr=${VGGT_LEARNING_RATE:-3e-5} weight_decay=${OPTIMIZER_WEIGHT_DECAY:-1e-3}"
echo "VGGT experiment: overlay=${experiment_overlay:-none} require_teacher_cache=$vggt_require_teacher_cache"
echo "VGGT diagnostics: intervention_interval=${intervention_interval:-config-default}"

set -x
CUDA_VISIBLE_DEVICES="$visible_devices" accelerate launch \
  "${launch_args[@]}" \
  starVLA/training/train_starvla.py \
  "${training_args[@]}"
