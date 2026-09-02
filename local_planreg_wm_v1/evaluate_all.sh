#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 EXPERIMENT=CHECKPOINT [EXPERIMENT=CHECKPOINT ...]" >&2
  echo "Example: $0 e3_register_qvlora_wm_seed0=/path/to/best.ckpt" >&2
  exit 2
fi

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export DRIVEVLA_SCORE_RAY="${DRIVEVLA_SCORE_RAY:-0}"
evaluation_root="${PLANREG_EVAL_ROOT:-${PLANREG_RUN_ROOT}/evaluation}"
evaluation_strategy=ddp
if [[ "${PLANREG_NUM_GPUS}" == "1" ]]; then
  evaluation_strategy=auto
fi

experiment_overrides() {
  local label="$1"
  case "${label}" in
    e0_*)
      printf '%s\n' \
        agent.vlm_config.planning_registers_enabled=false \
        agent.vlm_config.vision_qv_lora_enabled=false \
        agent.vision_adaptation.mode=none \
        agent.scene_fusion.mode=semantic_only \
        agent.world_model.enabled=false \
        agent.ema.enabled=false
      ;;
    e1_*)
      printf '%s\n' \
        agent.vlm_config.vision_qv_lora_enabled=false \
        agent.vision_adaptation.mode=none \
        agent.world_model.enabled=false \
        agent.ema.enabled=false
      ;;
    e2_*)
      :
      ;;
    e4_*) printf '%s\n' agent.world_model.future_mode=no_action_condition ;;
    e5_*) printf '%s\n' agent.world_model.future_mode=shuffled_batch ;;
    e6_*) printf '%s\n' agent.world_model.future_mode=repeated_current ;;
    e7_*) printf '%s\n' agent.world_model.predictor_only=true ;;
    e3_*) printf '%s\n' agent.world_model.future_mode=correct ;;
    *)
      echo "Unknown experiment label (must begin e0_...e7_): ${label}" >&2
      return 2
      ;;
  esac
}

for specification in "$@"; do
  if [[ "${specification}" != *=* ]]; then
    echo "Expected EXPERIMENT=CHECKPOINT, got: ${specification}" >&2
    exit 2
  fi
  label="${specification%%=*}"
  checkpoint="${specification#*=}"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Checkpoint does not exist: ${checkpoint}" >&2
    exit 2
  fi
  "${PYTHON_BIN}" \
    "${PLANREG_REPO_ROOT}/scripts/export_planreg_student_checkpoint.py" \
    --verify "${checkpoint}"
  if [[ -n "${PLANREG_EVAL_SEED:-}" ]]; then
    evaluation_seed="${PLANREG_EVAL_SEED}"
  elif [[ "${label}" =~ _seed([0-9]+)$ ]]; then
    evaluation_seed="${BASH_REMATCH[1]}"
  else
    echo "Evaluation label must end in _seedN or PLANREG_EVAL_SEED must be set: ${label}" >&2
    exit 2
  fi
  mapfile -t overrides < <(experiment_overrides "${label}")
  output_dir="${evaluation_root}/${label}"
  command=(
    "${PYTHON_BIN}"
    "${PLANREG_REPO_ROOT}/navsim/planning/script/run_pdm_score_multi_gpu.py"
    train_test_split=navtest
    agent=episode_drive_planreg_wm_v1
    "agent.checkpoint_path=${checkpoint}"
    agent.stage1_checkpoint_path=null
    "agent.vlm_config.vlm_path=${PLANREG_VLM_PATH}"
    "experiment_name=${label}"
    "output_dir=${output_dir}"
    "+seed=${evaluation_seed}"
    "navsim_log_path=${PLANREG_NAVSIM_LOG_ROOT}/test"
    "sensor_blobs_path=${PLANREG_SENSOR_BLOB_ROOT}/test"
    load_image_path=true
    dataloader.params.batch_size=1
    "+trainer.params.devices=${PLANREG_NUM_GPUS}"
    "trainer.params.strategy=${evaluation_strategy}"
    trainer.params.precision=32
    agent.world_model.enabled=false
    agent.ema.enabled=false
    "metric_cache_path=${PLANREG_NAVTEST_METRIC_CACHE}"
    "${overrides[@]}"
  )
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "PLANREG_EVAL_DRY_RUN label=${label} checkpoint=${checkpoint}"
    planreg_print_command "${command[@]}"
    continue
  fi
  planreg_require_directory "${PLANREG_VLM_PATH}"
  planreg_require_directory "${PLANREG_NAVSIM_LOG_ROOT}/test"
  planreg_require_directory "${PLANREG_SENSOR_BLOB_ROOT}/test"
  planreg_require_directory "${PLANREG_NAVTEST_METRIC_CACHE}"
  planreg_require_directory "${NUPLAN_MAPS_ROOT}"
  mkdir -p "${output_dir}/run_metadata"
  sha256sum "${checkpoint}" > "${output_dir}/run_metadata/checkpoint.sha256"
  printf '%s\n' 'BF16 VLM + FP32 action/scorer (not full FP32)' \
    > "${output_dir}/run_metadata/precision_contract.txt"
  git -C "${PLANREG_REPO_ROOT}" rev-parse HEAD > "${output_dir}/run_metadata/git_commit.txt"
  git -C "${PLANREG_REPO_ROOT}" status --short --branch > "${output_dir}/run_metadata/git_status.txt"
  env | LC_ALL=C sort | sed -E \
    's/^([^=]*(TOKEN|SECRET|PASSWORD|API_KEY)[^=]*)=.*/\1=<redacted>/I' \
    > "${output_dir}/run_metadata/environment.txt"
  planreg_print_command "${command[@]}" > "${output_dir}/run_metadata/evaluate_command.txt"
  "${command[@]}" --cfg job --resolve \
    > "${output_dir}/run_metadata/resolved_hydra_config.yaml"
  "${command[@]}" 2>&1 | tee "${output_dir}/run_metadata/evaluate.log"
done
