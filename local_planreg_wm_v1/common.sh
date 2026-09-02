#!/usr/bin/env bash
set -euo pipefail

PLANREG_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLANREG_REPO_ROOT="$(cd "${PLANREG_SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../load_env.sh
source "${PLANREG_REPO_ROOT}/load_env.sh"

PYTHON_BIN="${PYTHON_BIN:-/mnt/project/DriveVLA-M0-env/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="/mnt/project/DriveVLA-M0-env/bin/python"
fi
PLANREG_RUN_ROOT="${PLANREG_RUN_ROOT:-${NAVSIM_EXP_ROOT}/planreg_wm_v1}"
PLANREG_BASE_CHECKPOINT="${PLANREG_BASE_CHECKPOINT:-${DRIVEVLA_BASE_CHECKPOINT:-/mnt/project/DriveVLA-M0-modelscope/best-epoch_26-step_174312.server_merged.ckpt}}"
PLANREG_VLM_PATH="${PLANREG_VLM_PATH:-${DRIVEVLA_VLM_CONFIG:-/mnt/project/DriveVLA-M0-models/InternVL3-2B-modelscope}}"
PLANREG_NAVSIM_LOG_ROOT="${PLANREG_NAVSIM_LOG_ROOT:-/mnt/project/DriveDreamer-Policy/navsim_raw/navsim_logs}"
PLANREG_SENSOR_BLOB_ROOT="${PLANREG_SENSOR_BLOB_ROOT:-/mnt/project/onevl_navsim_data/sensor_blobs}"
PLANREG_TRAIN_METRIC_CACHE="${PLANREG_TRAIN_METRIC_CACHE:-${NAVSIM_TRAIN_METRIC_CACHE:-/mnt/project/DriveVLA-M0-stage2/cache/metric_cache_navtrain_full}}"
PLANREG_NAVTEST_METRIC_CACHE="${PLANREG_NAVTEST_METRIC_CACHE:-${METRIC_CACHE_PATH:-/mnt/project/DriveDreamer-Policy/navsim_exp/eval_v1_1/metric_cache_navtest}}"
PLANREG_TRAIN_TEST_SPLIT="${PLANREG_TRAIN_TEST_SPLIT:-navtrain}"
PLANREG_MAX_EPOCHS="${PLANREG_MAX_EPOCHS:-20}"
PLANREG_BATCH_SIZE="${PLANREG_BATCH_SIZE:-2}"
PLANREG_NUM_WORKERS="${PLANREG_NUM_WORKERS:-4}"
PLANREG_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a _planreg_gpu_ids <<< "${PLANREG_CUDA_DEVICES}"
PLANREG_NUM_GPUS="${PLANREG_NUM_GPUS:-${#_planreg_gpu_ids[@]}}"
unset _planreg_gpu_ids

export CUDA_VISIBLE_DEVICES="${PLANREG_CUDA_DEVICES}"
export DRIVEVLA_BASE_CHECKPOINT="${PLANREG_BASE_CHECKPOINT}"
export DRIVEVLA_VLM_CONFIG="${PLANREG_VLM_PATH}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/navsim/maps}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/project/DriveDreamer-Policy/navsim_raw}"
export NAVSIM_TRAIN_METRIC_CACHE="${PLANREG_TRAIN_METRIC_CACHE}"
export SUBSCORE_PATH="${PLANREG_RUN_ROOT}"
export DRIVEVLA_SCORE_RAY="${DRIVEVLA_SCORE_RAY:-0}"
export DRIVEVLA_SCORE_PROCESSES="${DRIVEVLA_SCORE_PROCESSES:-16}"
export DRIVEVLA_SCORE_PARTITIONS="${DRIVEVLA_SCORE_PARTITIONS:-8}"
export DRIVEVLA_SCORE_START_METHOD="${DRIVEVLA_SCORE_START_METHOD:-forkserver}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

planreg_validate_seed() {
  local seed="$1"
  local allowed="$2"
  case ",${allowed}," in
    *",${seed},"*) ;;
    *)
      echo "Seed ${seed} is not allowed for this experiment; allowed: ${allowed}" >&2
      return 2
      ;;
  esac
}

planreg_print_command() {
  printf '%q ' "$@"
  printf '\n'
}

planreg_data_split() {
  case "$1" in
    navtrain|trainval) printf '%s\n' trainval ;;
    navmini) printf '%s\n' mini ;;
    navtest) printf '%s\n' test ;;
    *)
      echo "Unsupported PlanReg split: $1" >&2
      return 2
      ;;
  esac
}

planreg_require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file does not exist: $1" >&2
    return 2
  fi
}

planreg_require_directory() {
  if [[ ! -d "$1" ]]; then
    echo "Required directory does not exist: $1" >&2
    return 2
  fi
}

planreg_launch() {
  if [[ $# -lt 2 ]]; then
    echo "planreg_launch requires EXPERIMENT_NAME SEED [HYDRA OVERRIDES...]" >&2
    return 2
  fi
  local experiment_name="$1"
  local seed="$2"
  shift 2
  local output_dir="${PLANREG_OUTPUT_DIR:-${PLANREG_RUN_ROOT}/training/${experiment_name}_seed${seed}}"
  local strategy="ddp"
  if [[ "${PLANREG_NUM_GPUS}" == "1" ]]; then
    strategy="auto"
  fi

  local resume_args=(train_ckpt_path=null)
  if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    if [[ ! -f "${RESUME_CHECKPOINT}" ]]; then
      echo "Explicit RESUME_CHECKPOINT does not exist: ${RESUME_CHECKPOINT}" >&2
      return 2
    fi
    resume_args=("train_ckpt_path=${RESUME_CHECKPOINT}")
  fi

  local split_args=("train_test_split=${PLANREG_TRAIN_TEST_SPLIT}")
  local data_split
  data_split="$(planreg_data_split "${PLANREG_TRAIN_TEST_SPLIT}")"
  local smoke_args=()
  if [[ "${SMOKE_SPLIT:-0}" == "1" ]]; then
    split_args=(train_test_split=navmini)
    data_split=mini
    smoke_args=(
      "train_test_split.scene_filter.max_scenes=${SMOKE_SCENES:-32}"
      trainer.params.max_epochs=1
      trainer.params.limit_train_batches=2
      trainer.params.limit_val_batches=1
    )
  fi

  local command=(
    "${PYTHON_BIN}"
    "${PLANREG_REPO_ROOT}/navsim/planning/script/run_training_full.py"
    "${split_args[@]}"
    agent=episode_drive_planreg_wm_v1
    "experiment_name=${experiment_name}_seed${seed}"
    "output_dir=${output_dir}"
    "seed=${seed}"
    "navsim_log_path=${PLANREG_NAVSIM_LOG_ROOT}/${data_split}"
    "sensor_blobs_path=${PLANREG_SENSOR_BLOB_ROOT}/${data_split}"
    load_image_path=true
    use_cache_without_dataset=false
    force_cache_computation=false
    preprocess_images_in_workers=false
    auto_resume=false
    "agent.checkpoint_path=${PLANREG_BASE_CHECKPOINT}"
    agent.stage1_checkpoint_path=null
    "agent.vlm_config.vlm_path=${PLANREG_VLM_PATH}"
    "agent.batch_size=${PLANREG_BATCH_SIZE}"
    "agent.num_gpus=${PLANREG_NUM_GPUS}"
    "dataloader.params.batch_size=${PLANREG_BATCH_SIZE}"
    "dataloader.params.num_workers=${PLANREG_NUM_WORKERS}"
    "trainer.params.devices=${PLANREG_NUM_GPUS}"
    "trainer.params.strategy=${strategy}"
    trainer.params.precision=bf16-mixed
    trainer.params.gradient_clip_val=1.0
    trainer.params.gradient_clip_algorithm=norm
    "trainer.params.max_epochs=${PLANREG_MAX_EPOCHS}"
    "${resume_args[@]}"
    "${smoke_args[@]}"
    "$@"
  )

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "PLANREG_DRY_RUN experiment=${experiment_name} seed=${seed} output=${output_dir}"
    planreg_print_command "${command[@]}"
    return 0
  fi

  planreg_require_file "${PLANREG_BASE_CHECKPOINT}"
  planreg_require_directory "${PLANREG_VLM_PATH}"
  planreg_require_directory "${PLANREG_NAVSIM_LOG_ROOT}/${data_split}"
  planreg_require_directory "${PLANREG_SENSOR_BLOB_ROOT}/${data_split}"
  planreg_require_directory "${PLANREG_TRAIN_METRIC_CACHE}"
  planreg_require_directory "${NUPLAN_MAPS_ROOT}"

  mkdir -p "${output_dir}/run_metadata"
  git -C "${PLANREG_REPO_ROOT}" rev-parse HEAD > "${output_dir}/run_metadata/git_commit.txt"
  git -C "${PLANREG_REPO_ROOT}" status --short --branch > "${output_dir}/run_metadata/git_status.txt"
  env | LC_ALL=C sort | sed -E \
    's/^([^=]*(TOKEN|SECRET|PASSWORD|API_KEY)[^=]*)=.*/\1=<redacted>/I' \
    > "${output_dir}/run_metadata/environment.txt"
  planreg_print_command "${command[@]}" > "${output_dir}/run_metadata/train_command.txt"
  "${command[@]}" --cfg job --resolve \
    > "${output_dir}/run_metadata/resolved_hydra_config.yaml"
  "${command[@]}" 2>&1 | tee "${output_dir}/run_metadata/train.log"
}
