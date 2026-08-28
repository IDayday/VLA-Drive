#!/usr/bin/env bash
set -euo pipefail

GPU_LIST="3,5,6,7"
# The 2k-scene convergence audit showed that eight epochs under-fit the MLP
# relative to fifteen; formal Gate C1 therefore uses the converged budget.
EPOCHS=15
BATCH_SCENES=512
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
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-scenes) BATCH_SCENES="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

IFS=',' read -r -a GPUS <<<"${GPU_LIST}"
if (( ${#GPUS[@]} != 4 )); then
  echo "The audited schedule requires exactly four comma-separated GPUs" >&2
  exit 2
fi
if (( EPOCHS < 1 || BATCH_SCENES < 1 )); then
  echo "epochs and batch-scenes must be positive" >&2
  exit 2
fi

mkdir -p "${CACHE_DIR}/oracle_job_logs"

# One sequential queue per GPU prevents two jobs from accidentally sharing a
# card when feature groups have different runtimes. O3 owns the held-out-family
# audit; its internal audit also evaluates O8 without duplicating O8 fold rows.
GROUP_QUEUES=(
  "O0 O4 O8 O12"
  "O1 O5 O9 O13"
  "O3 O6 O10"
  "O2 O7 O11"
)

run_queue() {
  local queue_index="$1"
  local gpu="${GPUS[$queue_index]}"
  local group
  for group in ${GROUP_QUEUES[$queue_index]}; do
    if [[ -f "${CACHE_DIR}/oracle_jobs/formal_${group}/job_summary.json" \
       && -f "${CACHE_DIR}/oracle_jobs/formal_${group}/oracle_factor_calibration.csv" ]]; then
      echo "Reusing completed formal_${group} on GPU ${gpu}"
      continue
    fi
    local extra=()
    if [[ "${group}" == "O3" ]]; then
      extra+=(--heldout)
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
      -m tools.shared_future_candidate_consequence.run_oracle_decomposition \
      --mode full \
      --groups "${group}" \
      --models linear,mlp \
      --epochs "${EPOCHS}" \
      --batch-scenes "${BATCH_SCENES}" \
      --device cuda \
      --seed "${SEED}" \
      --job-name "formal_${group}" \
      --output-dir "${OUTPUT_DIR}" \
      --cache-dir "${CACHE_DIR}" \
      "${extra[@]}" \
      >"${CACHE_DIR}/oracle_job_logs/formal_${group}_gpu_${gpu}.log" 2>&1
  done
}

pids=()
for queue_index in 0 1 2 3; do
  run_queue "${queue_index}" &
  pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failures=$((failures + 1))
  fi
done
if (( failures > 0 )); then
  echo "${failures} GPU queue(s) failed; inspect ${CACHE_DIR}/oracle_job_logs" >&2
  exit 1
fi

"${PYTHON_BIN}" -m tools.shared_future_candidate_consequence.aggregate_oracle_jobs \
  --num-folds 5 \
  --seed "${SEED}" \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir "${CACHE_DIR}"
