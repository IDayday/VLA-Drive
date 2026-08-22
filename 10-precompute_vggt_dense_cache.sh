#!/usr/bin/env bash
# Precompute uncompressed dense VGGT patch features for Spatial-Forcing style alignment.

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

export VGGT_MODEL="${VGGT_MODEL:-/mnt/LLM_weight/facebook/VGGT-1B}"

split="${SPLIT:-train}"
datalist="${DATALIST:-${NAVSIM_DATALIST_PATH:-${DRIVEDREAMER_ROOT}/${split}_meta.json}}"
layer_index="${VGGT_DENSE_LAYER_INDEX:--1}"
cache_root="${NAVSIM_VGGT_DENSE_CACHE_ROOT:-${DRIVEDREAMER_ROOT}/navsim_feature_cache/vggt_dense_${split}_layer${layer_index}}"

if [ -n "${NUM_PROCESSES:-}" ]; then
  num_processes="${NUM_PROCESSES}"
elif [ -n "${NPROC_PER_NODE:-}" ]; then
  num_processes="${NPROC_PER_NODE}"
else
  num_processes="$(python -c 'import torch; print(torch.cuda.device_count())')"
fi
if [ -z "$num_processes" ] || [ "$num_processes" = "0" ]; then
  echo "No CUDA device is visible; VGGT dense cache generation requires GPU." >&2
  exit 2
fi

map_size_gb="${MAP_SIZE_GB:-512}"
batch_size="${BATCH_SIZE:-1}"
image_size="${VGGT_IMAGE_SIZE:-518}"
max_samples_arg=()
if [ -n "${MAX_SAMPLES:-}" ]; then
  max_samples_arg=(--max-samples "${MAX_SAMPLES}")
fi
overwrite_arg=()
if [ "${OVERWRITE:-0}" = "1" ]; then
  overwrite_arg=(--overwrite)
fi

mkdir -p "$cache_root"

echo "VGGT_MODEL=${VGGT_MODEL}"
echo "cache_root=${cache_root}"
echo "datalist=${datalist}"
echo "data_root=${DATA_ROOT}"
echo "split=${split}"
echo "num_processes=${num_processes} batch_size=${batch_size} map_size_gb=${map_size_gb} layer_index=${layer_index} image_size=${image_size}"

set -x
torchrun --standalone --nnodes=1 --nproc-per-node="${num_processes}" \
  tools/precompute_vggt_dense_cache.py \
  --cache-root "${cache_root}" \
  --vggt-model "${VGGT_MODEL}" \
  --datalist "${datalist}" \
  --data-root "${DATA_ROOT}" \
  --split "${split}" \
  --layer-index "${layer_index}" \
  --image-size "${image_size}" \
  --batch-size "${batch_size}" \
  --map-size-gb "${map_size_gb}" \
  --commit-interval "${COMMIT_INTERVAL:-8}" \
  --log-interval "${LOG_INTERVAL:-100}" \
  "${max_samples_arg[@]}" \
  "${overwrite_arg[@]}"
