#!/usr/bin/env bash
# One-command SQ-3D-Mix gated cache preparation and formal training on PAI-DLC.
# Fixed default contract: 1 node x 16 PPU x batch 2, effective batch 32,
# 100K optimizer steps, real VGGT intervention, and no training smoke.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

usage() {
  echo "Usage: bash $project_root/run_sq3dmix_gated_dlc.sh"
}

if (( $# != 0 )); then
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi
  usage >&2
  exit 2
fi

phase="bootstrap"
on_error() {
  local status=$?
  echo "[sq3dmix-gated-dlc] FAILED phase=$phase exit_code=$status" >&2
  exit "$status"
}
trap on_error ERR

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[sq3dmix-gated-dlc] Missing required file: $1" >&2
    exit 2
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "[sq3dmix-gated-dlc] Missing required directory: $1" >&2
    exit 2
  fi
}

timestamp="$(date +'%Y%m%d_%H%M%S')"

export NUM_MACHINES=1
export MACHINE_RANK=0
export LOCAL_NUM_PROCESSES=16
export NUM_PROCESSES=16
export CUDA_VISIBLE_DEVICES="${SQ3DMIX_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export PER_DEVICE_BATCH_SIZE=2
export GRADIENT_ACCUMULATION_STEPS=1
export TARGET_EFFECTIVE_BATCH_SIZE=32
export NAVSIM_NUM_WORKERS="${NAVSIM_NUM_WORKERS:-3}"
export NAVSIM_PREFETCH_FACTOR="${NAVSIM_PREFETCH_FACTOR:-2}"
export NAVSIM_PIN_MEMORY="${NAVSIM_PIN_MEMORY:-1}"

export SQ3DMIX_FUSION_MODE=gated
export SQ3DMIX_INTERVENTION=real
export SQ3DMIX_SMOKE=0
export SQ3DMIX_DRY_RUN=0
export TRAINING_TOPOLOGY_ONLY=0

export MAX_TRAIN_STEPS=100000
export NUM_WARMUP_STEPS=5000
export SAVE_INTERVAL=10000
export TRAINING_LOGGING_FREQUENCY=50
export EXPECTED_TRAIN_SAMPLES=103288
export FM_REPEAT=8
export ACTION_HIDDEN_SIZE=1536
export ACTION_LAYERS=24

export BASE_LEARNING_RATE=1e-5
export ACTION_LEARNING_RATE=1e-5
export SCENE_QUERY_LEARNING_RATE=3e-5
export GATED_FUSION_LEARNING_RATE=1e-4
export OPTIMIZER_WEIGHT_DECAY=1e-3
export VLM_ATTN_IMPLEMENTATION=sdpa

export VGGT_DENSE_CACHE_NUM_PROCESSES=16
export VGGT_DENSE_CACHE_BATCH_SIZE="${VGGT_DENSE_CACHE_BATCH_SIZE:-1}"
export VGGT_DENSE_CACHE_MAP_SIZE_GB="${VGGT_DENSE_CACHE_MAP_SIZE_GB:-80}"
export VGGT_DENSE_CACHE_COMMIT_INTERVAL="${VGGT_DENSE_CACHE_COMMIT_INTERVAL:-8}"
export VGGT_DENSE_CACHE_FULL=1

export TRAIN_CONFIG_YAML="$project_root/starVLA/config/training/cfg_yaw_1225.yaml"
export MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-${MASTER_ADDR:-127.0.0.1}}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-${MASTER_PORT:-29694}}"
export RUN_ID="${RUN_ID:-sq3dmix-gated-real-${MAX_TRAIN_STEPS}-${PAI_JOB_ID:-dlc}-${timestamp}}"
export TRITON_CACHE_DIR="${TRITON_LOCAL_CACHE_ROOT:-/tmp/drivedreamer-triton}/${RUN_ID}/node0"

split="${SPLIT:-train}"
cache_manifest="$NAVSIM_VGGT_DENSE_CACHE_ROOT/vggt_dense/manifest.json"
launcher_log_dir="$NAVSIM_EXP_ROOT/launcher_logs"
launcher_log="$launcher_log_dir/${RUN_ID}.pipeline.log"
mkdir -p "$launcher_log_dir" "$TRITON_CACHE_DIR"
exec > >(tee -a "$launcher_log") 2>&1

echo "[sq3dmix-gated-dlc] project_root=$DRIVEDREAMER_ROOT"
echo "[sq3dmix-gated-dlc] run_id=$RUN_ID"
echo "[sq3dmix-gated-dlc] topology=1x16 effective_batch=32"
echo "[sq3dmix-gated-dlc] algorithm=gated intervention=real steps=$MAX_TRAIN_STEPS smoke=disabled"
echo "[sq3dmix-gated-dlc] lr=base:$BASE_LEARNING_RATE action:$ACTION_LEARNING_RATE scene:$SCENE_QUERY_LEARNING_RATE fusion:$GATED_FUSION_LEARNING_RATE"
echo "[sq3dmix-gated-dlc] data_root=$DATA_ROOT"
echo "[sq3dmix-gated-dlc] datalist=$NAVSIM_DATALIST_PATH"
echo "[sq3dmix-gated-dlc] dense_cache_root=$NAVSIM_VGGT_DENSE_CACHE_ROOT"
echo "[sq3dmix-gated-dlc] launcher_log=$launcher_log"
if [[ -n "${ACTION_ONLY_CHECKPOINT:-}" ]]; then
  echo "[sq3dmix-gated-dlc] initialization=action-only checkpoint=$ACTION_ONLY_CHECKPOINT"
else
  echo "[sq3dmix-gated-dlc] initialization=base-vlm checkpoint=none"
fi

phase="dense-cache"
require_file "$NAVSIM_DATALIST_PATH"
require_dir "$DATA_ROOT/meta/$split"
if [[ ! -f "$cache_manifest" ]]; then
  require_dir "$VGGT_REPO"
  require_file "$VGGT_CHECKPOINT"
  require_dir "$NAVSIM_TRAINVAL_SENSOR_ROOT"
  mkdir -p "$NAVSIM_VGGT_DENSE_CACHE_ROOT"
  bash "$DRIVEDREAMER_ROOT/11-precompute_vggt_dense_cache.sh"
else
  echo "[sq3dmix-gated-dlc] Dense VGGT manifest exists; cache generation skipped."
fi
require_file "$cache_manifest"

phase="formal-training"
require_file "$BASE_VLM/config.json"
require_file "$TRAIN_CONFIG_YAML"
require_file "$DRIVEDREAMER_ROOT/starVLA/config/training/sq_3d_mix.yaml"
if [[ -n "${ACTION_ONLY_CHECKPOINT:-}" ]]; then
  require_file "$ACTION_ONLY_CHECKPOINT"
fi

bash "$DRIVEDREAMER_ROOT/14-train_sq_3d_mix.sh" \
  --trainer.num_warmup_steps "$NUM_WARMUP_STEPS" \
  --trainer.save_interval "$SAVE_INTERVAL" \
  --trainer.logging_frequency "$TRAINING_LOGGING_FREQUENCY" \
  --datasets.vla_data.expected_sample_count "$EXPECTED_TRAIN_SAMPLES"

echo "[sq3dmix-gated-dlc] COMPLETE run_dir=$NAVSIM_EXP_ROOT/$RUN_ID"
