#!/usr/bin/env bash
# Distributed frozen-feature precomputation for formal NAVSIM v1 training.
# Formal contract: one node x 16 PPU processes, batch 4 per process.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

# Match the formal training input path exactly.  The MP4 alternative contains
# the same frames but is H.264-lossy, so it must never be mixed into an
# image-source feature cache used for reproducibility.
export NAVSIM_VIDEO_SOURCE="${NAVSIM_VIDEO_SOURCE:-images}"

actual_devices="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
formal_job=0
if [ -n "${NPROC_PER_NODE:-}" ] || [ -n "${PAI_JOB_ID:-}" ]; then
  formal_job=1
fi

num_nodes="${NUM_MACHINES:-${WORLD_SIZE:-1}}"
node_rank="${MACHINE_RANK:-${RANK:-0}}"
if (( formal_job == 1 )); then
  local_processes="${PRECACHE_LOCAL_PROCESSES:-16}"
else
  local_processes="${PRECACHE_LOCAL_PROCESSES:-$actual_devices}"
fi
batch_size="${PRECACHE_BATCH_SIZE:-4}"
cache_root="${PRECACHE_ROOT:-$NAVSIM_FEATURE_CACHE_ROOT}"
components="${PRECACHE_COMPONENTS:-${NAVSIM_CACHE_COMPONENTS:-wan,ppd}}"
map_size_gb="${PRECACHE_MAP_SIZE_GB:-256}"
max_samples="${PRECACHE_MAX_SAMPLES:-}"

for name in num_nodes node_rank local_processes batch_size map_size_gb actual_devices; do
  value="${!name}"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "[pre-cache] $name must be a non-negative integer, got: $value" >&2
    exit 2
  fi
done
if (( num_nodes != 1 || node_rank != 0 )); then
  echo "[pre-cache] This launcher requires exactly one node (num_nodes=$num_nodes node_rank=$node_rank)" >&2
  exit 2
fi
if (( local_processes < 1 || local_processes > actual_devices )); then
  echo "[pre-cache] Need $local_processes local devices, but only $actual_devices are visible" >&2
  exit 2
fi
if (( formal_job == 1 && local_processes != 16 )); then
  echo "[pre-cache] Formal cache generation is pinned to 16 local processes" >&2
  exit 2
fi
if (( batch_size < 1 || map_size_gb < 1 )); then
  echo "[pre-cache] Batch size and LMDB map size must be positive" >&2
  exit 2
fi
if [ -n "$max_samples" ] && ! [[ "$max_samples" =~ ^[1-9][0-9]*$ ]]; then
  echo "[pre-cache] PRECACHE_MAX_SAMPLES must be a positive integer" >&2
  exit 2
fi

if [ ! -d "$DRIVEDREAMER_SHARED_ROOT" ]; then
  echo "[pre-cache] Canonical shared project path is not mounted: $DRIVEDREAMER_SHARED_ROOT" >&2
  exit 2
fi
if [ ! "$project_root/pre_cache.sh" -ef "$DRIVEDREAMER_SHARED_ROOT/pre_cache.sh" ]; then
  echo "[pre-cache] Entrypoint and canonical shared project path are not the same repository" >&2
  echo "[pre-cache] entrypoint=$project_root canonical=$DRIVEDREAMER_SHARED_ROOT" >&2
  exit 2
fi

resolved_cache_root="$(readlink -m "$cache_root")"
resolved_cache_parent="$(readlink -m "$DRIVEDREAMER_SHARED_ROOT/navsim_feature_cache")"
case "$resolved_cache_root" in
  "$resolved_cache_parent"|"$resolved_cache_parent"/*) ;;
  *)
    echo "[pre-cache] Refusing cache path outside the dedicated project cache directory: $cache_root" >&2
    echo "[pre-cache] Use a path below $resolved_cache_parent" >&2
    exit 2
    ;;
esac
cache_root="$resolved_cache_root"

required_paths=(
  "$BASE_VLM/config.json"
  "$BASE_VLM/model.safetensors"
  "$VIDEO_MODEL/Wan2.1_VAE.pth"
  "$VIDEO_MODEL/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
  "$PPD_MODEL"
  "$DEPTH_ANYTHING_MODEL"
  "$DATA_ROOT/meta/train"
  "$NAVSIM_TRAINVAL_SENSOR_ROOT"
  "$NAVSIM_DATALIST_PATH"
)
for path in "${required_paths[@]}"; do
  if [ ! -e "$path" ]; then
    echo "[pre-cache] Missing required path: $path" >&2
    exit 2
  fi
done

mkdir -p "$cache_root/logs"
run_id="${PRECACHE_RUN_ID:-navsim-v1-cache-$(date +'%Y%m%d_%H%M%S')}"
log_path="$cache_root/logs/${run_id}.log"
exec > >(tee -a "$log_path") 2>&1

echo "[pre-cache] project_root=$project_root"
echo "[pre-cache] canonical_shared_project_root=$DRIVEDREAMER_SHARED_ROOT"
echo "[pre-cache] run_id=$run_id"
echo "[pre-cache] topology=nodes:1 node_rank:0 local_processes:$local_processes visible_devices:$actual_devices"
echo "[pre-cache] batch_per_device=$batch_size aggregate_batch=$((local_processes * batch_size))"
echo "[pre-cache] components=$components cache_root=$cache_root map_size_gb_per_rank=$map_size_gb"
echo "[pre-cache] data_root=$DATA_ROOT sensor_root=$NAVSIM_TRAINVAL_SENSOR_ROOT video_source=$NAVSIM_VIDEO_SOURCE log=$log_path"
echo "[pre-cache] script_sha256=$(sha256sum "$project_root/pre_cache.sh" | awk '{print $1}')"

if [ "${PRECACHE_TOPOLOGY_ONLY:-0}" = "1" ]; then
  echo "[pre-cache] PRECACHE_TOPOLOGY_ONLY=1; no cache was generated"
  exit 0
fi

sample_count="$(python - <<'PY'
import json
import os
with open(os.environ["NAVSIM_DATALIST_PATH"], "r", encoding="utf-8") as stream:
    print(len(json.load(stream)))
PY
)"
if [ -z "$max_samples" ] && [ "$sample_count" != "103288" ]; then
  echo "[pre-cache] Expected 103288 NAVSIM v1 samples, found $sample_count" >&2
  exit 2
fi

# Feature payloads are durable, but compiler/autotune state must be rank-local.
export TRITON_CACHE_DIR="${TRITON_LOCAL_CACHE_ROOT:-/tmp/drivedreamer-precache-triton}/${run_id}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_LOCAL_ROOT:-/tmp/drivedreamer-precache-extensions}/${run_id}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

common_args=(
  --cache-root "$cache_root"
  --config-yaml "${TRAIN_CONFIG_YAML:-$project_root/starVLA/config/training/cfg_yaw_1225.yaml}"
  --base-vlm "$BASE_VLM"
  --video-model "$VIDEO_MODEL"
  --ppd-model "$PPD_MODEL"
  --depth-model "$DEPTH_ANYTHING_MODEL"
  --video-config "${VIDEO_CONFIG:-$project_root/starVLA/model/modules/video_model/config/wan2.1/wan_civitai.yaml}"
  --datalist "$NAVSIM_DATALIST_PATH"
  --data-root "$DATA_ROOT"
  --split train
  --attn-implementation "$VLM_ATTN_IMPLEMENTATION"
  --batch-size "$batch_size"
  --map-size-gb "$map_size_gb"
)
if [ -n "$max_samples" ]; then
  common_args+=(--max-samples "$max_samples")
fi
if [ "${PRECACHE_OVERWRITE:-0}" = "1" ]; then
  common_args+=(--overwrite)
fi

IFS=',' read -r -a component_list <<< "$components"
for component in "${component_list[@]}"; do
  component="${component//[[:space:]]/}"
  case "$component" in
    qwen|wan|ppd) ;;
    *) echo "[pre-cache] Unknown component: $component" >&2; exit 2 ;;
  esac
  echo "[pre-cache] Starting component=$component"
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="$local_processes" \
    "$project_root/starVLA/cache/precompute_navsim.py" \
    --component "$component" \
    "${common_args[@]}"
done

echo "[pre-cache] ALL COMPONENTS COMPLETE cache_root=$cache_root"
