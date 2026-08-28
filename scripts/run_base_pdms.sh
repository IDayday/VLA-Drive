#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../load_env.sh
source "${REPO_ROOT}/load_env.sh"

required_variables=(
  NUPLAN_MAPS_ROOT
  OPENSCENE_DATA_ROOT
  DRIVEVLA_BASE_CHECKPOINT
  DRIVEVLA_VLM_CONFIG
  DRIVEVLA_DINO_WEIGHTS
  METRIC_CACHE_PATH
)
for variable in "${required_variables[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "Required environment variable is unset: ${variable}" >&2
    echo "Copy env.local.example.sh to env.local.sh and configure local paths." >&2
    exit 2
  fi
done

NAVSIM_ROOT="${NAVSIM_ROOT:-${REPO_ROOT}}"
CHECKPOINT_PATH="${DRIVEVLA_BASE_CHECKPOINT}"
AGENT_CONFIG="${AGENT_CONFIG:-episode_drive}"
NUM_GPUS="${NUM_GPUS:-1}"
PL_STRATEGY="${PL_STRATEGY:-auto}"
NUM_WORKERS="${NUM_WORKERS:-4}"

if [[ "${RUN_PREFLIGHT:-1}" == "1" ]]; then
  "${SCRIPT_DIR}/preflight_base.sh"
fi

extra_args=()
if [[ -n "${MAX_SCENES:-}" ]]; then
  extra_args+=("train_test_split.scene_filter.max_scenes=${MAX_SCENES}")
fi

"${PYTHON_BIN}" "${NAVSIM_ROOT}/navsim/planning/script/run_pdm_score_multi_gpu.py" \
  train_test_split="${TRAIN_TEST_SPLIT:-navtest}" \
  agent="${AGENT_CONFIG}" \
  agent.checkpoint_path="${CHECKPOINT_PATH}" \
  experiment_name="${EXPERIMENT_NAME:-drivevla_base_no_memory_navtest_pdms}" \
  load_image_path=true \
  dataloader.params.batch_size="${BATCH_SIZE:-1}" \
  dataloader.params.num_workers="${NUM_WORKERS}" \
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
  agent.action_head_config.scorer_ref_num=4 \
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
  agent.vlm_config.vlm_path="${DRIVEVLA_VLM_CONFIG}" \
  agent.vlm_config.initialize_from_config=true \
  agent.vlm_config.use_flash_attn=false \
  agent.vlm_config.extra_token_count=8 \
  agent.vlm_config.target_vocab_size=151682 \
  agent.lora_config.use_lora=true \
  agent.lora_config.lora_target_modules="[attn.qkv,attn.proj,q_proj,v_proj,k_proj,o_proj]" \
  metric_cache_path="${METRIC_CACHE_PATH}" \
  "${extra_args[@]}" \
  "$@"
