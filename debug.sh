#!/usr/bin/env bash
# Debug / quick-sanity single-GPU training run (mini dataset, 1 process).
# Run: source env.sh && bash debug.sh

set -euo pipefail

: "${NAVSIM_EXP_ROOT:?Set NAVSIM_EXP_ROOT in env.sh}"
: "${BASE_VLM:?Set BASE_VLM in env.sh}"
: "${VIDEO_MODEL:?Set VIDEO_MODEL in env.sh}"
: "${DATA_ROOT:?Set DATA_ROOT in env.sh}"
: "${WANDB_ENTITY:?Set WANDB_ENTITY in env.sh}"
: "${WANDB_PROJECT:?Set WANDB_PROJECT in env.sh}"

timestamp="debug"
num_processes="${NUM_PROCESSES:-1}"
GPU="${GPU:-0}"
visible_devices="${CUDA_VISIBLE_DEVICES:-$GPU}"
PORT="${PORT:-29688}"
bz="${PER_DEVICE_BATCH_SIZE:-1}"
act_fm_size=1536
act_fm_layer=24
fm_repeat=8
max_train_steps="${MAX_TRAIN_STEPS:-1}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-1}"

VIDEO_CONFIG=starVLA/model/modules/video_model/config/wan2.1/wan_civitai.yaml
VIDEO_DATA_DIR="${DATA_ROOT}/navsim_video"

split=mini
datalist="${DRIVEDREAMER_ROOT}/${split}_meta.json"
run_id=${timestamp}-3d-2d-1d-lr1e5-3d_loss_1e1-decay1e3-${split}_data-bz_${bz}_${num_processes}

Framework_name=QwenOFT
vl_hidden_dim=2048

export WANDB_MODE=offline

set -x
pwd

CUDA_VISIBLE_DEVICES="${visible_devices}" accelerate launch \
  --main_process_port ${PORT} \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  --mixed_precision bf16 \
  starVLA/training/train_starvla.py \
  --config_yaml ./starVLA/config/training/cfg_yaw_1225.yaml \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${BASE_VLM} \
  --framework.qwenvl.attn_implementation "${VLM_ATTN_IMPLEMENTATION:-flash_attention_2}" \
  --framework.qwenvl.vl_hidden_dim ${vl_hidden_dim} \
  --run_root_dir ${NAVSIM_EXP_ROOT} \
  --run_id ${run_id} \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_entity ${WANDB_ENTITY} \
  --datasets.vla_data.datalist_path ${datalist} \
  --datasets.vla_data.data_root ${DATA_ROOT} \
  --datasets.vla_data.split ${split} \
  --datasets.vla_data.per_device_batch_size ${bz} \
  --trainer.gradient_accumulation_steps ${gradient_accumulation_steps} \
  --framework.action_model.repeated_diffusion_steps ${fm_repeat} \
  --datasets.video_data.load_2d_data 1 \
  --w_depth 1 \
  --gs_query_loss 1 \
  --rgb_query_loss 1 \
  --trainer.freeze_modules "rgb_model.vae,rgb_model.clip_image_encoder,rgb_model.text_encoder,qwen_vl_interface.model.visual" \
  --framework.action_model.hidden_size ${act_fm_size} \
  --framework.action_model.diffusion_model_cfg.cross_attention_dim ${act_fm_size} \
  --framework.action_model.diffusion_model_cfg.output_dim ${act_fm_size} \
  --framework.action_model.diffusion_model_cfg.num_layers ${act_fm_layer} \
  --trainer.optimizer.weight_decay 1e-3 \
  --trainer.learning_rate.base 1e-5 \
  --trainer.learning_rate.rgb_model 1e-5 \
  --framework.video_model.model_name ${VIDEO_MODEL} \
  --framework.video_model.config_path ${VIDEO_CONFIG} \
  --datasets.video_data.rgb_meta_dir ${VIDEO_DATA_DIR} \
  --trainer.max_train_steps "${max_train_steps}"
