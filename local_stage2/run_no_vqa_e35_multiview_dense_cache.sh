#!/usr/bin/env bash

# Export a denser scorer-private current-observation cache from the frozen
# No-VQA E35 M0 vision encoder.  Shards are resumable and may be distributed
# across hosts by assigning disjoint DENSE_SHARD_IDS values.

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
split="${DENSE_SPLIT:-trainval}"
shard_count="${DENSE_SHARD_COUNT:-8}"
shard_ids_csv="${DENSE_SHARD_IDS:-0,1,2,3,4,5,6,7}"
gpu_ids_csv="${DENSE_GPU_IDS:-${shard_ids_csv}}"
pool_height="${DENSE_POOL_HEIGHT:-4}"
pool_width="${DENSE_POOL_WIDTH:-4}"
max_dynamic_tiles="${DENSE_MAX_DYNAMIC_TILES:-4}"
batch_size="${DENSE_BATCH_SIZE:-8}"

case "${split}" in
  trainval)
    inventory_args=(--feature-root "${feature_root}")
    log_path="${train_log_path}"
    sensor_root="${train_sensor_root}"
    output="${DENSE_OUTPUT_DIR:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_multiview_trainval_pool4_tiles4_v1_8shard}"
    expected_scenes=103288
    ;;
  navtest)
    inventory_args=(--proposal-pickle "${navtest_proposals}")
    log_path="${test_log_path}"
    sensor_root="${test_sensor_root}"
    output="${DENSE_OUTPUT_DIR:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_multiview_navtest_pool4_tiles4_v1_8shard}"
    expected_scenes=12146
    ;;
  *)
    echo "DENSE_SPLIT must be trainval or navtest" >&2
    exit 2
    ;;
esac
log_root="${DENSE_LOG_ROOT:-/root/scorer_pdms93_logs/no_vqa_e35_multiview_pool4_tiles4_v1/${split}}"

IFS=',' read -r -a shard_ids <<< "${shard_ids_csv}"
IFS=',' read -r -a gpu_ids <<< "${gpu_ids_csv}"
(( ${#shard_ids[@]} > 0 )) || { echo "DENSE_SHARD_IDS is empty" >&2; exit 2; }
(( ${#shard_ids[@]} == ${#gpu_ids[@]} )) || {
  echo "DENSE_SHARD_IDS and DENSE_GPU_IDS must have equal lengths" >&2
  exit 2
}
for path in "${repo_root}" "${checkpoint}" "${resolved_config}" "${vlm_path}" "${log_path}" "${sensor_root}"; do
  [[ -e "${path}" ]] || { echo "missing dense-cache input: ${path}" >&2; exit 2; }
done
for shard in "${shard_ids[@]}"; do
  [[ "${shard}" =~ ^[0-9]+$ ]] || { echo "invalid shard id: ${shard}" >&2; exit 2; }
  (( shard >= 0 && shard < shard_count )) || {
    echo "shard id ${shard} is outside [0, ${shard_count})" >&2
    exit 2
  }
done

mkdir -p "${output}" "${log_root}"
export PYTHONPATH="${repo_root}:${repo_root}/nuplan-devkit${PYTHONPATH:+:${PYTHONPATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

pids=()
for index in "${!shard_ids[@]}"; do
  shard="${shard_ids[${index}]}"
  gpu="${gpu_ids[${index}]}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    exec "${python_bin}" \
      "${repo_root}/local_stage2/export_multiview_m0_observation_replay.py" \
      "${inventory_args[@]}" \
      --repo-root "${repo_root}" \
      --checkpoint "${checkpoint}" \
      --resolved-config "${resolved_config}" \
      --vlm-path "${vlm_path}" \
      --log-path "${log_path}" \
      --sensor-root "${sensor_root}" \
      --output-dir "${output}" \
      --max-dynamic-tiles "${max_dynamic_tiles}" \
      --pool-height "${pool_height}" \
      --pool-width "${pool_width}" \
      --batch-size "${batch_size}" \
      --image-workers 8 \
      --chunk-size 128 \
      --shard-count "${shard_count}" \
      --shard-index "${shard}"
  ) >"${log_root}/shard_${shard}.log" 2>&1 &
  pids+=("$!")
  echo "NO_VQA_DENSE_CACHE_STARTED split=${split} shard=${shard} gpu=${gpu} pid=$!"
done

failure=0
for index in "${!pids[@]}"; do
  if wait "${pids[${index}]}"; then
    echo "NO_VQA_DENSE_CACHE_SHARD_COMPLETE split=${split} shard=${shard_ids[${index}]}"
  else
    status=$?
    echo "NO_VQA_DENSE_CACHE_SHARD_FAILED split=${split} shard=${shard_ids[${index}]} status=${status}" >&2
    failure=1
  fi
done
(( failure == 0 )) || exit 1

# A partial distributed invocation is successful without claiming the cache is
# complete.  The invocation that observes all shard manifests performs the
# immutable full-table validation and alone creates `.complete`.
manifest_count="$(find "${output}" -mindepth 2 -maxdepth 2 -name manifest.json -type f | wc -l)"
if (( manifest_count == shard_count )); then
  expected_tokens="$((4 * (max_dynamic_tiles + 1) * pool_height * pool_width))"
  "${python_bin}" - "${output}" "${expected_scenes}" "${expected_tokens}" "${checkpoint}" <<'PY'
import hashlib
import sys
from pathlib import Path

from local_stage2.train_independent_scorer import load_private_observation_table

root = Path(sys.argv[1])
expected_scenes = int(sys.argv[2])
expected_tokens = int(sys.argv[3])
checkpoint = Path(sys.argv[4])
table = load_private_observation_table(root)
assert len(table.tokens) == expected_scenes
assert len(set(table.tokens)) == expected_scenes
assert tuple(table.observation_tokens.shape[1:]) == (expected_tokens, 1536)
digest = hashlib.sha256()
with checkpoint.open("rb") as stream:
    while block := stream.read(8 * 1024 * 1024):
        digest.update(block)
assert table.lineage["checkpoint_sha256"] == digest.hexdigest()
assert table.lineage["current_observation_only"] is True
assert table.lineage["future_or_evaluator_input"] is False
print({"root": str(root), "scenes": expected_scenes, "tokens": expected_tokens})
PY
  touch "${output}/.complete"
  echo "NO_VQA_DENSE_CACHE_COMPLETE split=${split} root=${output}"
else
  echo "NO_VQA_DENSE_CACHE_PARTIAL split=${split} manifests=${manifest_count}/${shard_count} root=${output}"
fi
