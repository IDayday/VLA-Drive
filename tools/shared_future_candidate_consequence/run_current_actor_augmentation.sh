#!/usr/bin/env bash
set -euo pipefail

NUM_SHARDS=32
MAX_PARALLEL=8
ACTOR_SLOTS=16
OUTPUT_DIR=reports/shared_future_candidate_consequence_gate_c
CACHE_DIR=outputs/shared_future_candidate_consequence_gate_c/all
DEFAULT_NAVSIM_PYTHON=python
if [[ -x /root/miniconda3/envs/navsim/bin/python ]]; then
  DEFAULT_NAVSIM_PYTHON=/root/miniconda3/envs/navsim/bin/python
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_NAVSIM_PYTHON}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-shards) NUM_SHARDS="$2"; shift 2 ;;
    --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --actor-slots) ACTOR_SLOTS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${CACHE_DIR}/current_actor_augmentation/logs"
run_one() {
  local shard="$1"
  local padded
  padded=$(printf '%04d' "${shard}")
  "${PYTHON_BIN}" -m tools.shared_future_candidate_consequence.augment_current_actor_targets \
    --split trainval \
    --actor-slots "${ACTOR_SLOTS}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${shard}" \
    --output-dir "${OUTPUT_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    >"${CACHE_DIR}/current_actor_augmentation/logs/shard_${padded}.log" 2>&1
}

failures=0
running=0
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  run_one "${shard}" &
  running=$((running + 1))
  if (( running >= MAX_PARALLEL )); then
    if ! wait -n; then failures=$((failures + 1)); fi
    running=$((running - 1))
  fi
done
while (( running > 0 )); do
  if ! wait -n; then failures=$((failures + 1)); fi
  running=$((running - 1))
done
if (( failures > 0 )); then
  echo "${failures} current-actor augmentation shard(s) failed" >&2
  exit 1
fi

"${PYTHON_BIN}" -m tools.shared_future_candidate_consequence.augment_current_actor_targets \
  --mode aggregate \
  --actor-slots "${ACTOR_SLOTS}" \
  --num-shards "${NUM_SHARDS}" \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir "${CACHE_DIR}"
