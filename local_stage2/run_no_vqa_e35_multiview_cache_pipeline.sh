#!/usr/bin/env bash

# Export scorer-private current multiview tokens from the locally trained
# No-VQA M0 checkpoint.  Trainval and Navtest exports are physically separate;
# neither path reads proposals, factors, scores, future images or future labels
# as model inputs.  All eight local GPUs are used shard-wise and outputs are
# resumable at deterministic chunk boundaries.

set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
python_bin="${DRIVEVLA_PYTHON:-/mnt/project/DriveVLA-M0-env/bin/python}"
checkpoint="${NO_VQA_CHECKPOINT:-/mnt/project/DriveVLA-M0-no-vqa/runs/training/no_vqa_full_ft_seed0_e36/lightning_logs/version_0/checkpoints/best-epoch=35-step=232416.ckpt}"
resolved_config="${NO_VQA_RESOLVED_CONFIG:-/mnt/project/DriveVLA-M0-no-vqa/runs/training/no_vqa_full_ft_seed0_e36/code/hydra/config.yaml}"
vlm_path="${DRIVEVLA_VLM_DIR:-/mnt/project/DriveVLA-M0-models/InternVL3-2B-modelscope}"
feature_root="${NO_VQA_FEATURE_ROOT:-/root/scorer_pdms93_cache/no_vqa_e35_features_full_v1}"
navtest_proposals="${NO_VQA_NAVTEST_PROPOSALS:-/mnt/project/DriveVLA-M0-stage2/runs/ke_candidate_audit/no_vqa_e35_navtest_scorer_features_fp32_v1/proposal_predictions.pkl}"
train_log_path="${NO_VQA_TRAIN_LOG_PATH:-/mnt/project/DriveDreamer-Policy/navsim_raw/navsim_logs/trainval}"
train_sensor_root="${NO_VQA_TRAIN_SENSOR_ROOT:-/mnt/project/onevl_navsim_data/sensor_blobs/trainval}"
test_log_path="${NO_VQA_TEST_LOG_PATH:-/mnt/project/DriveDreamer-Policy/navsim_raw/navsim_logs/test}"
test_sensor_root="${NO_VQA_TEST_SENSOR_ROOT:-/mnt/project/onevl_navsim_data/sensor_blobs/test}"
train_output="${NO_VQA_MULTIVIEW_TRAIN_OUTPUT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_multiview_trainval_pool2_tiles4_v1_8shard}"
test_output="${NO_VQA_MULTIVIEW_TEST_OUTPUT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_multiview_navtest_pool2_tiles4_v1_8shard}"
log_root="${NO_VQA_MULTIVIEW_LOG_ROOT:-/root/scorer_pdms93_logs/no_vqa_e35_multiview_pool2_tiles4_v1}"
shard_count=8
resume="${NO_VQA_MULTIVIEW_RESUME:-0}"

for path in "${repo_root}" "${checkpoint}" "${resolved_config}" "${vlm_path}" "${feature_root}" "${navtest_proposals}" "${train_log_path}" "${train_sensor_root}" "${test_log_path}" "${test_sensor_root}"; do
  [[ -e "${path}" ]] || { echo "missing multiview input: ${path}" >&2; exit 2; }
done
if [[ "${resume}" != "0" && "${resume}" != "1" ]]; then
  echo "NO_VQA_MULTIVIEW_RESUME must be 0 or 1" >&2
  exit 2
fi
if [[ "${resume}" == "0" ]]; then
  for path in "${train_output}" "${test_output}" "${log_root}"; do
    [[ ! -e "${path}" ]] || { echo "refusing existing multiview output: ${path}" >&2; exit 2; }
  done
fi

mkdir -p "${train_output}" "${test_output}" "${log_root}"
export PYTHONPATH="${repo_root}:${repo_root}/nuplan-devkit${PYTHONPATH:+:${PYTHONPATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

export_split() {
  local split="$1"
  local output="$2"
  local log_path="$3"
  local sensor_root="$4"
  shift 4
  local inventory_args=("$@")
  local pids=()
  for shard in $(seq 0 $((shard_count - 1))); do
    (
      export CUDA_VISIBLE_DEVICES="${shard}"
      exec "${python_bin}" "${repo_root}/local_stage2/export_multiview_m0_observation_replay.py" \
        "${inventory_args[@]}" \
        --repo-root "${repo_root}" \
        --checkpoint "${checkpoint}" \
        --resolved-config "${resolved_config}" \
        --vlm-path "${vlm_path}" \
        --log-path "${log_path}" \
        --sensor-root "${sensor_root}" \
        --output-dir "${output}" \
        --max-dynamic-tiles 4 \
        --pool-height 2 \
        --pool-width 2 \
        --batch-size 8 \
        --image-workers 8 \
        --chunk-size 128 \
        --shard-count "${shard_count}" \
        --shard-index "${shard}"
    ) >"${log_root}/${split}_shard_${shard}.log" 2>&1 &
    pids+=("$!")
    echo "NO_VQA_MULTIVIEW_STARTED split=${split} shard=${shard} gpu=${shard} pid=$!"
  done
  local failure=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[${index}]}"; then
      echo "NO_VQA_MULTIVIEW_FAILED split=${split} shard=${index}" >&2
      failure=1
    fi
  done
  (( failure == 0 )) || exit 1
}

validate_cache() {
  local root="$1"
  local expected="$2"
  "${python_bin}" - "${root}" "${expected}" "${checkpoint}" <<'PY'
import hashlib
import sys
from pathlib import Path

from local_stage2.train_independent_scorer import load_private_observation_table

root = Path(sys.argv[1])
expected = int(sys.argv[2])
checkpoint = Path(sys.argv[3])
table = load_private_observation_table(root)
assert len(table.tokens) == expected, (len(table.tokens), expected)
assert len(set(table.tokens)) == expected
assert tuple(table.observation_tokens.shape[1:]) == (80, 1536)
digest = hashlib.sha256()
with checkpoint.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
        digest.update(chunk)
assert table.lineage["checkpoint_sha256"] == digest.hexdigest()
assert table.lineage["current_observation_only"] is True
assert table.lineage["future_or_evaluator_input"] is False
print({"root": str(root), "scene_count": expected, "status": "PASS"})
PY
}

if [[ ! -f "${train_output}/.complete" ]]; then
  export_split trainval "${train_output}" "${train_log_path}" "${train_sensor_root}" \
    --feature-root "${feature_root}"
  validate_cache "${train_output}" 103288 \
    >"${log_root}/validate_trainval.log" 2>&1
  touch "${train_output}/.complete"
else
  validate_cache "${train_output}" 103288 \
    >"${log_root}/validate_trainval.log" 2>&1
fi

if [[ ! -f "${test_output}/.complete" ]]; then
  export_split navtest "${test_output}" "${test_log_path}" "${test_sensor_root}" \
    --proposal-pickle "${navtest_proposals}"
  validate_cache "${test_output}" 12146 \
    >"${log_root}/validate_navtest.log" 2>&1
  touch "${test_output}/.complete"
else
  validate_cache "${test_output}" 12146 \
    >"${log_root}/validate_navtest.log" 2>&1
fi

echo "NO_VQA_MULTIVIEW_PIPELINE_COMPLETE train=${train_output} navtest=${test_output}"
