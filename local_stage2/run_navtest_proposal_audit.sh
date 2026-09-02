#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 CHECKPOINT AGENT_TARGET EXPERIMENT_NAME CUDA_DEVICE_CSV [HYDRA_OVERRIDES...]" >&2
  exit 2
fi

checkpoint_path="$1"
agent_target="$2"
experiment_name="$3"
device_csv="$4"
shift 4

IFS=',' read -r -a device_ids <<< "${device_csv}"
device_count="${#device_ids[@]}"
score_workers="${DRIVEVLA_SCORE_WORKERS:-64}"
evaluation_precision="${DRIVEVLA_EVAL_PRECISION:-32}"
navtest_sensor_root="${DRIVEVLA_NAVTEST_SENSOR_ROOT:-${DRIVEVLA_SENSOR_ROOT}}"
if (( device_count < 1 )); then
  echo "At least one CUDA device is required" >&2
  exit 2
fi
if [[ ! -f "${checkpoint_path}" ]]; then
  echo "Checkpoint not found: ${checkpoint_path}" >&2
  exit 2
fi

hydra_checkpoint_path="${checkpoint_path//=/\\=}"
output_dir="${DRIVEVLA_STAGE2_RUN_ROOT}/ke_candidate_audit/${experiment_name}"
if [[ -d "${output_dir}" ]] && [[ -n "$(find "${output_dir}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite proposal audit: ${output_dir}" >&2
  exit 3
fi
mkdir -p "${output_dir}"

export CUDA_VISIBLE_DEVICES="${device_csv}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export DRIVEVLA_SCORE_RAY=0
export INTERNVL_VERBOSE_DYNAMIC_BATCH=0
export NAVSIM_TRAIN_METRIC_CACHE="${DRIVEVLA_NAVTEST_METRIC_CACHE}"
# Each Ray PDM worker owns one CPU.  Avoid nested BLAS pools, which otherwise
# oversubscribe a 64-core host by an order of magnitude.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

"${DRIVEVLA_PYTHON}" "${DRIVEVLA_REPO_ROOT}/local_stage2/run_navtest_proposal_audit.py" \
  train_test_split=navtest \
  agent=episode_drive \
  "agent._target_=${agent_target}" \
  "agent.checkpoint_path=${hydra_checkpoint_path}" \
  agent.stage1_checkpoint_path=null \
  "experiment_name=${experiment_name}" \
  "output_dir=${output_dir}" \
  load_image_path=true \
  dataloader.params.batch_size=2 \
  "+trainer.params.devices=${device_count}" \
  trainer.params.strategy=ddp \
  "trainer.params.precision=${evaluation_precision}" \
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
  "agent.vlm_config.vlm_path=${DRIVEVLA_VLM_DIR}" \
  agent.vlm_config.initialize_from_config=true \
  agent.vlm_config.use_flash_attn=false \
  agent.vlm_config.extra_token_count=8 \
  agent.vlm_config.target_vocab_size=151682 \
  agent.lora_config.use_lora=true \
  'agent.lora_config.lora_target_modules=[attn.qkv,attn.proj,q_proj,v_proj,k_proj,o_proj]' \
  "metric_cache_path=${DRIVEVLA_NAVTEST_METRIC_CACHE}" \
  "navsim_log_path=${DRIVEVLA_DATA_ROOT}/navsim_logs/test" \
  "sensor_blobs_path=${navtest_sensor_root}/test" \
  worker=ray_distributed_no_torch \
  "worker.threads_per_node=${score_workers}" \
  worker.log_to_driver=false \
  logger_level=warning \
  "$@"
