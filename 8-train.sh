#!/usr/bin/env bash
# Step 8 (train): Multi-node/multi-GPU training with DeepSpeed ZeRO-2.
# Prerequisites:
#   1. source env.sh           (sets NAVSIM_EXP_ROOT, BASE_VLM, WANDB_*, …)
#   2. Prepare data (steps 0-3) and generate the meta-list JSON.
# Run: source env.sh && bash 8-train.sh

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

# ── Required env vars (set in env.sh) ─────────────────────────────────────────
: "${NAVSIM_EXP_ROOT:?Set NAVSIM_EXP_ROOT in env.sh}"
: "${BASE_VLM:?Set BASE_VLM in env.sh}"
: "${VIDEO_MODEL:?Set VIDEO_MODEL in env.sh}"
: "${DATA_ROOT:?Set DATA_ROOT in env.sh}"
: "${WANDB_ENTITY:?Set WANDB_ENTITY in env.sh}"
: "${WANDB_PROJECT:?Set WANDB_PROJECT in env.sh}"

# ── Experiment ID ─────────────────────────────────────────────────────────────
debug=false
if [ "${debug,,}" = "true" ]; then
  timestamp="debug"
else
  timestamp="$(date +"%m%d_%H")"
fi
echo "timestamp: $timestamp"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
num_processes="${NUM_PROCESSES:-2}"
num_machines="${NUM_MACHINES:-1}"
machine_rank="${MACHINE_RANK:-0}"
local_num_processes="${LOCAL_NUM_PROCESSES:-${num_processes}}"
bz="${PER_DEVICE_BATCH_SIZE:-4}"
# training.sh derives this value to preserve the released global batch of 32.
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-4}"
act_fm_size=1536      # action DiT hidden size
act_fm_layer=24       # action DiT number of layers
fm_repeat=8           # repeated diffusion steps
max_train_steps="${MAX_TRAIN_STEPS:-100000}"

training_config="${TRAIN_CONFIG_YAML:-$DRIVEDREAMER_ROOT/starVLA/config/training/cfg_yaw_1225.yaml}"
VIDEO_CONFIG="${VIDEO_CONFIG:-$DRIVEDREAMER_ROOT/starVLA/model/modules/video_model/config/wan2.1/wan_civitai.yaml}"
VIDEO_DATA_DIR="${NAVSIM_VIDEO_ROOT:-${DATA_ROOT}/navsim_video}"

split=train
datalist="${NAVSIM_DATALIST_PATH:-${DRIVEDREAMER_ROOT}/${split}_meta.json}"

run_id="${RUN_ID:-${timestamp}-3d-2d-1d-lr1e5-3d_loss_1e1-decay1e3-${split}_data-bz_${bz}_${num_processes}}"

Framework_name=QwenOFT
vl_hidden_dim=2048

export WANDB_MODE="${WANDB_MODE:-offline}"
export NAVSIM_VIDEO_SOURCE="${NAVSIM_VIDEO_SOURCE:-images}"
export NAVSIM_NUM_WORKERS="${NAVSIM_NUM_WORKERS:-3}"
export NAVSIM_PREFETCH_FACTOR="${NAVSIM_PREFETCH_FACTOR:-2}"
visible_devices="${CUDA_VISIBLE_DEVICES:-0,1}"
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
  --config_yaml "${training_config}" \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --framework.qwenvl.attn_implementation "${VLM_ATTN_IMPLEMENTATION:-flash_attention_2}" \
  --framework.qwenvl.vl_hidden_dim "${vl_hidden_dim}" \
  --run_root_dir "${NAVSIM_EXP_ROOT}" \
  --run_id "${run_id}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}" \
  --datasets.vla_data.datalist_path "${datalist}" \
  --datasets.vla_data.data_root "${DATA_ROOT}" \
  --datasets.vla_data.split "${split}" \
  --datasets.vla_data.per_device_batch_size "${bz}" \
  --trainer.gradient_accumulation_steps "${gradient_accumulation_steps}" \
  --framework.action_model.repeated_diffusion_steps "${fm_repeat}" \
  --datasets.video_data.load_2d_data 1 \
  --w_depth 1 \
  --gs_query_loss 1 \
  --rgb_query_loss 1 \
  --trainer.freeze_modules "rgb_model.vae,rgb_model.clip_image_encoder,rgb_model.text_encoder,qwen_vl_interface.model.visual,qwen_vl_interface.model.lm_head" \
  --framework.action_model.hidden_size "${act_fm_size}" \
  --framework.action_model.diffusion_model_cfg.cross_attention_dim "${act_fm_size}" \
  --framework.action_model.diffusion_model_cfg.output_dim "${act_fm_size}" \
  --framework.action_model.diffusion_model_cfg.num_layers "${act_fm_layer}" \
  --trainer.optimizer.weight_decay 1e-3 \
  --trainer.learning_rate.base 1e-5 \
  --trainer.learning_rate.rgb_model 1e-5 \
  --framework.video_model.model_name "${VIDEO_MODEL}" \
  --framework.video_model.config_path "${VIDEO_CONFIG}" \
  --datasets.video_data.rgb_meta_dir "${VIDEO_DATA_DIR}" \
  --trainer.max_train_steps "${max_train_steps}" \
  "${training_extra_args[@]}"
