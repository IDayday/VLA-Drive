#!/usr/bin/env bash

# Export the frozen No-VQA epoch-35 proposal/scorer representation on the
# complete legal Navtrain cache and attach PDM supervision in a physically
# separate CPU-only tree.  The job is resumable: immutable completed chunks are
# skipped by both stages.

set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${DRIVEVLA_PYTHON:-/mnt/project/DriveVLA-M0-env/bin/python}"
checkpoint="${NO_VQA_CHECKPOINT:-/mnt/project/DriveVLA-M0-no-vqa/runs/training/no_vqa_full_ft_seed0_e36/lightning_logs/version_0/checkpoints/best-epoch=35-step=232416.ckpt}"
resolved_config="${NO_VQA_RESOLVED_CONFIG:-/mnt/project/DriveVLA-M0-no-vqa/runs/training/no_vqa_full_ft_seed0_e36/code/hydra/config.yaml}"
feature_cache="${NO_VQA_FEATURE_CACHE:-/mnt/project/DriveVLA-M0-stage2/cache/feature_cache_navtrain_full}"
metric_cache="${NO_VQA_METRIC_CACHE:-/mnt/project/DriveVLA-M0-stage2/cache/metric_cache_navtrain_full}"
source_root="${NO_VQA_SOURCE_ROOT:-/root/scorer_pdms93_cache/no_vqa_e35_features_full_v1}"
label_root="${NO_VQA_LABEL_ROOT:-/root/scorer_pdms93_cache/no_vqa_e35_labels_full_v1}"
log_root="${NO_VQA_LOG_ROOT:-/root/scorer_pdms93_logs/no_vqa_e35_cache_full_v1}"
gpu_list="${NO_VQA_GPU_LIST:-0,1,2,3,4,5,6,7}"
label_shards="${NO_VQA_LABEL_SHARDS:-4}"
label_workers_per_shard="${NO_VQA_LABEL_WORKERS_PER_SHARD:-12}"

for required in \
  "${python_bin}" \
  "${checkpoint}" \
  "${resolved_config}" \
  "${feature_cache}" \
  "${metric_cache}"; do
  if [[ ! -e "${required}" ]]; then
    echo "NO_VQA_CACHE_PIPELINE_ERROR missing=${required}" >&2
    exit 2
  fi
done

mkdir -p "${source_root}" "${label_root}" "${log_root}"

export DRIVEVLA_VLM_CONFIG="${DRIVEVLA_VLM_CONFIG:-/mnt/project/DriveVLA-M0-models/InternVL3-2B-modelscope}"
export DRIVEVLA_SCORE_RAY=0
export DRIVEVLA_SCORE_PROCESSES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONPATH="${repo_root}:${repo_root}/nuplan-devkit${PYTHONPATH:+:${PYTHONPATH}}"

IFS=',' read -r -a gpus <<< "${gpu_list}"
gpu_shards="${#gpus[@]}"
if (( gpu_shards == 0 )); then
  echo "NO_VQA_CACHE_PIPELINE_ERROR no GPUs supplied" >&2
  exit 2
fi

export_pids=()
export_labels=()
for shard_index in "${!gpus[@]}"; do
  gpu="${gpus[${shard_index}]}"
  label="export_${shard_index}_of_${gpu_shards}_gpu_${gpu}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    exec "${python_bin}" "${repo_root}/local_stage2/export_public_base_scorer_cache.py" \
      --repo-root "${repo_root}" \
      --checkpoint "${checkpoint}" \
      --resolved-config "${resolved_config}" \
      --feature-cache "${feature_cache}" \
      --output-dir "${source_root}" \
      --split all \
      --shard-count "${gpu_shards}" \
      --shard-index "${shard_index}" \
      --batch-size 2 \
      --num-workers 2 \
      --chunk-size 128
  ) >"${log_root}/${label}.log" 2>&1 &
  export_pids+=("$!")
  export_labels+=("${label}")
  echo "NO_VQA_CACHE_EXPORT_STARTED label=${label} pid=$!"
done

label_pids=()
label_names=()
for (( shard_index=0; shard_index<label_shards; shard_index+=1 )); do
  label="pdm_label_${shard_index}_of_${label_shards}"
  (
    exec "${python_bin}" "${repo_root}/local_stage2/score_public_base_scorer_cache.py" \
      --source-root "${source_root}" \
      --output-root "${label_root}" \
      --metric-cache "${metric_cache}" \
      --num-workers "${label_workers_per_shard}" \
      --worker-shard-count "${label_shards}" \
      --worker-shard-index "${shard_index}" \
      --watch \
      --poll-seconds 10
  ) >"${log_root}/${label}.log" 2>&1 &
  label_pids+=("$!")
  label_names+=("${label}")
  echo "NO_VQA_PDM_LABEL_STARTED label=${label} pid=$!"
done

failure=0
for index in "${!export_pids[@]}"; do
  if wait "${export_pids[${index}]}"; then
    echo "NO_VQA_CACHE_EXPORT_COMPLETE label=${export_labels[${index}]}"
  else
    status=$?
    echo "NO_VQA_CACHE_EXPORT_FAILED label=${export_labels[${index}]} status=${status}" >&2
    failure=1
  fi
done

for index in "${!label_pids[@]}"; do
  if wait "${label_pids[${index}]}"; then
    echo "NO_VQA_PDM_LABEL_COMPLETE label=${label_names[${index}]}"
  else
    status=$?
    echo "NO_VQA_PDM_LABEL_FAILED label=${label_names[${index}]} status=${status}" >&2
    failure=1
  fi
done

if (( failure != 0 )); then
  echo "NO_VQA_CACHE_PIPELINE_FAILED source=${source_root} labels=${label_root}" >&2
  exit 1
fi

echo "NO_VQA_CACHE_PIPELINE_COMPLETE source=${source_root} labels=${label_root}"
