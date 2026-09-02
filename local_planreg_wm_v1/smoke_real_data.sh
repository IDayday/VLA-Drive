#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

seed="${1:-0}"
planreg_validate_seed "${seed}" "0"
smoke_root="${PLANREG_REAL_SMOKE_ROOT:-${PLANREG_RUN_ROOT}/real_data_smoke_seed${seed}}"
training_dir="${smoke_root}/training"
student_checkpoint="${smoke_root}/planreg_student.ckpt"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  PLANREG_OUTPUT_DIR="${training_dir}" SMOKE_SPLIT=1 \
    planreg_launch real_data_smoke "${seed}" \
      agent.world_model.enabled=true \
      agent.ema.enabled=true \
      agent.world_model.future_mode=correct
  planreg_print_command "${PYTHON_BIN}" \
    "${PLANREG_REPO_ROOT}/scripts/export_planreg_student_checkpoint.py" \
    TRAINING_LAST_CKPT "${student_checkpoint}" \
    --resolved-config "${training_dir}/run_metadata/resolved_hydra_config.yaml"
  planreg_print_command "${PYTHON_BIN}" \
    "${PLANREG_REPO_ROOT}/scripts/smoke_planreg_student_inference.py" \
    "${student_checkpoint}" --vlm-path "${PLANREG_VLM_PATH}" \
    --navsim-log-path "${PLANREG_NAVSIM_LOG_ROOT}/trainval" \
    --sensor-blobs-path "${PLANREG_SENSOR_BLOB_ROOT}/trainval"
  exit 0
fi

if [[ "${PLANREG_NUM_GPUS}" != "1" ]]; then
  echo "Real-data smoke is intentionally single-GPU; set CUDA_VISIBLE_DEVICES to one GPU" >&2
  exit 2
fi
if [[ -e "${smoke_root}" ]]; then
  echo "Refusing to reuse real-data smoke directory: ${smoke_root}" >&2
  exit 2
fi

PLANREG_OUTPUT_DIR="${training_dir}" SMOKE_SPLIT=1 \
  planreg_launch real_data_smoke "${seed}" \
    agent.world_model.enabled=true \
    agent.ema.enabled=true \
    agent.world_model.future_mode=correct

mapfile -t checkpoints < <(find "${training_dir}" -name last.ckpt -print)
if [[ "${#checkpoints[@]}" -ne 1 ]]; then
  echo "Expected exactly one last.ckpt under ${training_dir}, found ${#checkpoints[@]}" >&2
  exit 2
fi

"${PYTHON_BIN}" "${PLANREG_REPO_ROOT}/scripts/export_planreg_student_checkpoint.py" \
  "${checkpoints[0]}" "${student_checkpoint}" \
  --resolved-config "${training_dir}/run_metadata/resolved_hydra_config.yaml" \
  | tee "${smoke_root}/student_export.json"
"${PYTHON_BIN}" "${PLANREG_REPO_ROOT}/scripts/export_planreg_student_checkpoint.py" \
  --verify "${student_checkpoint}"
"${PYTHON_BIN}" "${PLANREG_REPO_ROOT}/scripts/smoke_planreg_student_inference.py" \
  "${student_checkpoint}" \
  --vlm-path "${PLANREG_VLM_PATH}" \
  --navsim-log-path "${PLANREG_NAVSIM_LOG_ROOT}/trainval" \
  --sensor-blobs-path "${PLANREG_SENSOR_BLOB_ROOT}/trainval" \
  | tee "${smoke_root}/student_inference.json"
