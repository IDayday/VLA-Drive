#!/usr/bin/env bash
set -euo pipefail

GPU_LIST="3,5,6,7"
SCENES_PER_LOG=2
SEED=20260828
OUTPUT_DIR=reports/shared_future_candidate_consequence_gate_c
CACHE_DIR=outputs/shared_future_candidate_consequence_gate_c/all
PYTHON_BIN="${PYTHON_BIN:-/mnt/project/DriveVLA-M0-env/bin/python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --scenes-per-log) SCENES_PER_LOG="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

IFS=',' read -r -a GPUS <<<"${GPU_LIST}"
NUM_SHARDS=${#GPUS[@]}
if (( NUM_SHARDS == 0 )); then
  echo "At least one GPU is required" >&2
  exit 2
fi

mkdir -p "${CACHE_DIR}/model_candidates/export_logs"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export DRIVEVLA_SCORE_RAY=0
export DRIVEVLA_SCORE_PROCESSES=0

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu=${GPUS[$shard]}
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
    -m tools.shared_future_candidate_consequence.export_episode_drive_candidates \
    --mode export \
    --split trainval \
    --scenes-per-log "${SCENES_PER_LOG}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${shard}" \
    --determinism-scenes 1 \
    --seed "${SEED}" \
    --output-dir "${OUTPUT_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    >"${CACHE_DIR}/model_candidates/export_logs/shard_${shard}_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failures=$((failures + 1))
  fi
done
if (( failures > 0 )); then
  echo "${failures} EpisodeDrive export shard(s) failed" >&2
  exit 1
fi

"${PYTHON_BIN}" -m tools.shared_future_candidate_consequence.export_episode_drive_candidates \
  --mode aggregate \
  --split trainval \
  --scenes-per-log "${SCENES_PER_LOG}" \
  --seed "${SEED}" \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir "${CACHE_DIR}"
