#!/usr/bin/env bash
set -euo pipefail

export OPENBLAS_CORETYPE="${OPENBLAS_CORETYPE:-Haswell}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/GPU_JDTest_fs01/home/zdhs0164/navsim_data/maps}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/GPU_JDTest_fs01/home/zdhs0164/navsim_data/openscene/openscene-v1.1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/GPU_JDTest_fs01/home/zdhs0164/DriveVLAMemory/outputs/episode_drive_navsim1_1}"
export SUBSCORE_PATH="${SUBSCORE_PATH:-${NAVSIM_EXP_ROOT}}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/GPU_JDTest_fs01/home/zdhs0164/DriveVLAMemory/envs/navsim_internvl/bin/python}"
NAVSIM_ROOT="${NAVSIM_ROOT:-${REPO_ROOT}}"
CHECKPOINT_PATH="${DRIVEVLA_BASE_CHECKPOINT:-/GPU_JDTest_fs01/home/zdhs0164/DriveVLAMemory/models/base_model/DriveVLA_M0/checkpoints/best-epoch_26-step_174312.server_merged.ckpt}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/GPU_JDTest_fs01/home/zdhs0164/navsim_workspace/exp/metric_cache}"
AGENT_CONFIG="${AGENT_CONFIG:-episode_drive}"
NUM_GPUS="${NUM_GPUS:-1}"
PL_STRATEGY="${PL_STRATEGY:-auto}"

"${PYTHON_BIN}" "${NAVSIM_ROOT}/navsim/planning/script/run_pdm_score_multi_gpu.py" \
  train_test_split="${TRAIN_TEST_SPLIT:-navtest}" \
  agent="${AGENT_CONFIG}" \
  agent.checkpoint_path="${CHECKPOINT_PATH}" \
  experiment_name="${EXPERIMENT_NAME:-drivevla_base_navtest_pdms}" \
  load_image_path=true \
  dataloader.params.batch_size="${BATCH_SIZE:-2}" \
  +trainer.params.devices="${NUM_GPUS}" \
  trainer.params.strategy="${PL_STRATEGY}" \
  agent.action_head_config.proposal_num=64 \
  agent.action_head_config.refiner_ls_values=0.0 \
  agent.action_head_config.image_backbone.focus_front_cam=false \
  agent.action_head_config.one_token_per_traj=true \
  agent.action_head_config.refiner_num_heads=1 \
  agent.action_head_config.tf_d_model=256 \
  agent.action_head_config.tf_d_ffn=1024 \
  agent.action_head_config.area_pred=false \
  agent.action_head_config.agent_pred=false \
  agent.action_head_config.ref_num=4 \
  agent.action_head_config.noc=1 \
  agent.action_head_config.dac=1 \
  agent.action_head_config.ddc=0.0 \
  agent.action_head_config.ttc=5 \
  agent.action_head_config.ep=5 \
  agent.action_head_config.comfort=2 \
  agent.vlm_config.cam_type=single \
  agent.vlm_config.cache_hidden_state=false \
  agent.vlm_config.cache_mode=false \
  agent.vlm_config.freeze_backbone=true \
  agent.vlm_config.vlm_type=internvl \
  agent.vlm_config.vlm_path="${DRIVEVLA_VLM_CONFIG:-/GPU_JDTest_fs01/home/zdhs0164/DriveVLAMemory/models/base_model/InternVL3-2B}" \
  agent.vlm_config.initialize_from_config=true \
  agent.vlm_config.use_flash_attn=false \
  agent.vlm_config.extra_token_count=8 \
  agent.vlm_config.target_vocab_size=151682 \
  agent.lora_config.use_lora=true \
  agent.lora_config.lora_target_modules="[attn.qkv,attn.proj,q_proj,v_proj,k_proj,o_proj]" \
  metric_cache_path="${METRIC_CACHE_PATH}" \
  "$@"
