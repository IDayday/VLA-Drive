#!/usr/bin/env bash
set -euo pipefail

START_SHARD=0
END_SHARD=0
NUM_SHARDS=32
MAX_PARALLEL=4
SPLIT=trainval
NUM_CANDIDATES=16
ACTOR_SLOTS=16
SEED=20260828
OUTPUT_DIR=reports/shared_future_candidate_consequence_gate_c
CACHE_DIR=outputs/shared_future_candidate_consequence_gate_c/all
DEFAULT_NAVSIM_PYTHON=python
if [[ -x /root/miniconda3/envs/navsim/bin/python ]]; then
  DEFAULT_NAVSIM_PYTHON=/root/miniconda3/envs/navsim/bin/python
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_NAVSIM_PYTHON}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start-shard) START_SHARD="$2"; shift 2 ;;
    --end-shard) END_SHARD="$2"; shift 2 ;;
    --num-shards) NUM_SHARDS="$2"; shift 2 ;;
    --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --num-candidates) NUM_CANDIDATES="$2"; shift 2 ;;
    --actor-slots) ACTOR_SLOTS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if (( START_SHARD < 0 || END_SHARD < START_SHARD || END_SHARD >= NUM_SHARDS )); then
  echo "Invalid shard range ${START_SHARD}..${END_SHARD} for ${NUM_SHARDS} shards" >&2
  exit 2
fi
if (( MAX_PARALLEL < 1 )); then
  echo "--max-parallel must be positive" >&2
  exit 2
fi

mkdir -p "${CACHE_DIR}/job_logs"

run_one() {
  local shard="$1"
  local padded
  padded=$(printf '%04d' "${shard}")
  "${PYTHON_BIN}" -m tools.shared_future_candidate_consequence.run_all_log_pipeline \
    --split "${SPLIT}" \
    --num-candidates "${NUM_CANDIDATES}" \
    --actor-slots "${ACTOR_SLOTS}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${shard}" \
    --seed "${SEED}" \
    --output-dir "${OUTPUT_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    >"${CACHE_DIR}/job_logs/shard_${padded}.log" 2>&1
}

failures=0
running=0
for shard in $(seq "${START_SHARD}" "${END_SHARD}"); do
  run_one "${shard}" &
  running=$((running + 1))
  if (( running >= MAX_PARALLEL )); then
    if ! wait -n; then
      failures=$((failures + 1))
    fi
    running=$((running - 1))
  fi
done

while (( running > 0 )); do
  if ! wait -n; then
    failures=$((failures + 1))
  fi
  running=$((running - 1))
done

if (( failures > 0 )); then
  echo "${failures} shard process(es) failed; inspect ${CACHE_DIR}/job_logs" >&2
  exit 1
fi
echo "Completed shards ${START_SHARD}..${END_SHARD}"
