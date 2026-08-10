#!/usr/bin/env bash
# Step 8 (train): Agent-action training with DeepSpeed ZeRO-2 and minimal_agent prompt.
# This script loads env.sh automatically so it can run on its own.
# Prerequisites:
#   1. Prepare data (steps 0-3) and generate the meta-list JSON.
# Run: bash 8-train_agent_action.sh

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/scripts/load_env.sh"

# Agent-action training does not rely on the shared feature cache.
unset NAVSIM_FEATURE_CACHE_ROOT
export NAVSIM_USE_FEATURE_CACHE=0
export NAVSIM_AGENT_DINO_CACHE_ROOT="${NAVSIM_AGENT_DINO_CACHE_ROOT:-$DRIVEDREAMER_ROOT/navsim_feature_cache/agent_dino_vits14_train_top4}"
export NAVSIM_AGENT_DINO_CACHE_STRICT="${NAVSIM_AGENT_DINO_CACHE_STRICT:-1}"

# -- Experiment ID ------------------------------------------------------------
debug=false
if [ "${debug,,}" = "true" ]; then
  timestamp="debug"
else
  timestamp="$(date +"%m%d_%H")"
fi
echo "timestamp: $timestamp"

# -- Hyper-parameters ---------------------------------------------------------
num_processes="${NUM_PROCESSES:-16}"
num_machines="${NUM_MACHINES:-1}"
machine_rank="${MACHINE_RANK:-0}"
local_num_processes="${LOCAL_NUM_PROCESSES:-${num_processes}}"
bz="${PER_DEVICE_BATCH_SIZE:-2}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-1}"
act_fm_size=1536
act_fm_layer=24
fm_repeat=8
max_train_steps="${MAX_TRAIN_STEPS:-100000}"

split=train
datalist="${NAVSIM_DATALIST_PATH:-${DRIVEDREAMER_ROOT}/${split}_meta.json}"

run_id="${RUN_ID:-${timestamp}-agent-action-lr1e5-16g-bz_${bz}-ga_${gradient_accumulation_steps}-${split}}"

Framework_name=QwenOFT
vl_hidden_dim=2048

export NAVSIM_VIDEO_SOURCE="${NAVSIM_VIDEO_SOURCE:-images}"
export NAVSIM_NUM_WORKERS="${NAVSIM_NUM_WORKERS:-3}"
export NAVSIM_PREFETCH_FACTOR="${NAVSIM_PREFETCH_FACTOR:-2}"
visible_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
accelerate_config="${TRAIN_ACCELERATE_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}"
main_process_ip="${MAIN_PROCESS_IP:-${MASTER_ADDR:-127.0.0.1}}"
main_process_port="${MAIN_PROCESS_PORT:-29687}"

launch_args=(
  --main_process_port "${main_process_port}"
  --config_file "${accelerate_config}"
  --num_processes "${num_processes}"
  --num_machines "${num_machines}"
  --machine_rank "${machine_rank}"
  --mixed_precision bf16
)
training_extra_args=()
if [ -n "${TRAINING_LOGGING_FREQUENCY:-}" ]; then
  training_extra_args+=(--trainer.logging_frequency "${TRAINING_LOGGING_FREQUENCY}")
fi
if (( num_machines > 1 )); then
  launch_args+=(
    --main_process_ip "${main_process_ip}"
    --same_network
  )
fi

set -x
pwd
echo "launch topology: nodes=${num_machines} node_rank=${machine_rank} local_processes=${local_num_processes} global_processes=${num_processes} master=${main_process_ip}:${main_process_port}"

CUDA_VISIBLE_DEVICES="${visible_devices}" accelerate launch \
  "${launch_args[@]}" \
  starVLA/training/train_starvla.py \
  --config_yaml ./starVLA/config/training/cfg_yaw_1225.yaml \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${BASE_VLM} \
  --framework.qwenvl.attn_implementation "${VLM_ATTN_IMPLEMENTATION:-flash_attention_2}" \
  --framework.qwenvl.vl_hidden_dim ${vl_hidden_dim} \
  --framework.action_prompt_mode minimal_agent \
  --framework.agent_dino.loss_weight 0.1 \
  --framework.agent_dino.feature_dim 384 \
  --run_root_dir ${NAVSIM_EXP_ROOT} \
  --run_id ${run_id} \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_entity ${WANDB_ENTITY} \
  --datasets.vla_data.datalist_path ${datalist} \
  --datasets.vla_data.data_root ${DATA_ROOT} \
  --datasets.vla_data.split ${split} \
  --datasets.vla_data.per_device_batch_size ${bz} \
  --datasets.vla_data.load_act_data 1 \
  --datasets.video_data.load_2d_data 0 \
  --datasets.gs_data.load_3d_data 0 \
  --datasets.reward_data.load_reward_data 0 \
  --w_depth 0 \
  --rgb_query_loss 0 \
  --gs_query_loss 0 \
  --trainer.gradient_accumulation_steps ${gradient_accumulation_steps} \
  --framework.action_model.repeated_diffusion_steps ${fm_repeat} \
  --framework.action_model.hidden_size ${act_fm_size} \
  --framework.action_model.diffusion_model_cfg.cross_attention_dim ${act_fm_size} \
  --framework.action_model.diffusion_model_cfg.output_dim ${act_fm_size} \
  --framework.action_model.diffusion_model_cfg.num_layers ${act_fm_layer} \
  --trainer.optimizer.weight_decay 1e-3 \
  --trainer.learning_rate.base 1e-5 \
  --trainer.learning_rate.action_model 1e-5 \
  --trainer.max_train_steps "${max_train_steps}" \
  "${training_extra_args[@]}"
