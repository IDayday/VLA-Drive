#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 base|driving_vqa STUDENT_CHECKPOINT" >&2
  exit 2
fi
variant="$1"
student_checkpoint="$2"
student_agent=episode_drive_planreg_wm_formal_student
if [[ "${PLANREG_PROTOCOL_VERSION:-v1}" == v1p1 ]]; then
  student_agent=episode_drive_planreg_wm_v1p1_student
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
fi
if [[ "${PLANREG_PROTOCOL_VERSION:-v1}" == task_future_lite ]]; then
  student_agent=episode_drive_task_future_lite_student
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
fi
if [[ "${variant}" != "base" && "${variant}" != "driving_vqa" ]]; then
  echo "Unknown formal variant: ${variant}" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
# shellcheck source=../load_env.sh
source "${repo_root}/load_env.sh"
# shellcheck source=formal_runtime.sh
source "${script_dir}/formal_runtime.sh"
planreg_formal_runtime_setup "${repo_root}"
python_bin="${PLANREG_FORMAL_PYTHON_BIN}"
[[ -f "${student_checkpoint}" ]] || { echo "Missing student checkpoint: ${student_checkpoint}" >&2; exit 2; }

if [[ "${variant}" == "base" ]]; then
  : "${PLANREG_BASE_VLM_PATH:?Set PLANREG_BASE_VLM_PATH}"
  vlm_path="${PLANREG_BASE_VLM_PATH}"
else
  : "${PLANREG_VQA_VLM_PATH:?Set PLANREG_VQA_VLM_PATH}"
  vlm_path="${PLANREG_VQA_VLM_PATH}"
fi
navsim_data="${OPENSCENE_DATA_ROOT:-/mnt/project/DriveDreamer-Policy/navsim_raw}"
maps_root="${NUPLAN_MAPS_ROOT:-/mnt/navsim/maps}"
metric_cache="${PLANREG_NAVTEST_METRIC_CACHE:-${METRIC_CACHE_PATH:-/mnt/project/DriveDreamer-Policy/navsim_exp/eval_v1_1/metric_cache_navtest}}"
evaluation_root="${PLANREG_FORMAL_EVAL_ROOT:-/mnt/project/DriveVLA-M0-formal-runs/evaluation}/${variant}"
jobs="${PLANREG_CANDIDATE_SCORE_JOBS:-32}"
for path in "${vlm_path}" "${navsim_data}/navsim_logs/test" "${navsim_data}/sensor_blobs/test" "${maps_root}" "${metric_cache}"; do
  [[ -d "${path}" ]] || { echo "Missing formal evaluation directory: ${path}" >&2; exit 2; }
done
if [[ -e "${evaluation_root}" ]]; then
  echo "Refusing to overwrite formal evaluation: ${evaluation_root}" >&2
  exit 2
fi

export PLANREG_STUDENT_CHECKPOINT="$(realpath "${student_checkpoint}")"
export PLANREG_FORMAL_VLM_PATH="$(realpath "${vlm_path}")"
export OPENSCENE_DATA_ROOT="${navsim_data}"
export NUPLAN_MAPS_ROOT="${maps_root}"
export NAVSIM_TRAIN_METRIC_CACHE="${metric_cache}"
export SUBSCORE_PATH="${evaluation_root}"
export DRIVEVLA_SCORE_RAY=0
export DRIVEVLA_SCORE_PROCESSES=0
export CUDA_VISIBLE_DEVICES="${PLANREG_EVAL_GPU:-0}"

"${python_bin}" "${repo_root}/scripts/export_planreg_student_checkpoint.py" \
  --verify "${student_checkpoint}"
mkdir -p "${evaluation_root}/run_metadata"
planreg_formal_runtime_audit_local \
  "${repo_root}" "${evaluation_root}/run_metadata/formal_runtime_node0.json"
printf '%s\n' 'BF16 VLM + FP32 action/scorer (not full FP32)' \
  > "${evaluation_root}/run_metadata/precision_contract.txt"
if [[ "${PLANREG_PROTOCOL_VERSION:-v1}" == v1p1 ]]; then
  printf '%s\n' 'FP32 VLM + FP32 action/scorer, current-only benchmark' \
    > "${evaluation_root}/run_metadata/precision_contract.txt"
fi
printf '%s\n' 'Current single front frame only; no future input or predictor/EMA construction.' \
  > "${evaluation_root}/run_metadata/inference_contract.txt"

run_inference_and_selected_pdms() {
  local label="$1"
  local max_scenes="$2"
  local output="${evaluation_root}/${label}"
  local bank="${output}/candidate_bank.npz"
  mkdir -p "${output}"
  local args=(
    train_test_split=navtest
    "agent=${student_agent}"
    "experiment_name=formal_${variant}_${label}"
    "output_dir=${output}"
    "metric_cache_path=${metric_cache}"
    "navsim_log_path=${navsim_data}/navsim_logs/test"
    "sensor_blobs_path=${navsim_data}/sensor_blobs/test"
    load_image_path=true
    dataloader.params.batch_size=1
    dataloader.params.num_workers=4
    +trainer.params.devices=1
    trainer.params.strategy=auto
    trainer.params.precision=32
    candidate_analysis=true
    "candidate_artifact_path=${bank}"
  )
  if [[ "${PLANREG_PROTOCOL_VERSION:-v1}" == v1p1 ]]; then
    args+=(agent.vlm_config.compute_dtype=float32)
  fi
  if [[ "${max_scenes}" != "all" ]]; then
    args+=("train_test_split.scene_filter.max_scenes=${max_scenes}")
  fi
  "${python_bin}" "${repo_root}/navsim/planning/script/run_pdm_score_multi_gpu.py" \
    "${args[@]}" 2>&1 | tee "${output}/evaluation.log"
}

# Public smoke gate is mandatory before spending a full Navtest pass.
run_inference_and_selected_pdms gate4 4
"${python_bin}" "${repo_root}/scripts/score_planreg_candidate_bank.py" \
  --candidate-bank "${evaluation_root}/gate4/candidate_bank.npz" \
  --metric-cache "${metric_cache}" \
  --work-dir "${evaluation_root}/gate4/candidate_scoring" \
  --output "${evaluation_root}/gate4/candidate_bank_scored.npz" --jobs 4
"${python_bin}" "${repo_root}/scripts/validate_planreg_formal_eval_gate.py" \
  --candidate-bank "${evaluation_root}/gate4/candidate_bank.npz" \
  --score-directory "${evaluation_root}/gate4" \
  | tee "${evaluation_root}/gate4/gate_report.json"

run_inference_and_selected_pdms navtest all
"${python_bin}" "${repo_root}/scripts/score_planreg_candidate_bank.py" \
  --candidate-bank "${evaluation_root}/navtest/candidate_bank.npz" \
  --metric-cache "${metric_cache}" \
  --work-dir "${evaluation_root}/navtest/candidate_scoring" \
  --output "${evaluation_root}/navtest/candidate_bank_scored.npz" \
  --jobs "${jobs}"
"${python_bin}" "${repo_root}/local_planreg_wm_v1/collect_candidate_metrics.py" \
  "${evaluation_root}/navtest/candidate_bank_scored.npz" \
  --output "${evaluation_root}/navtest/formal_candidate_metrics.json"
