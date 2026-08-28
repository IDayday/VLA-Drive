#!/usr/bin/env bash
set -euo pipefail

GPU=3
EPOCHS=15
OUTPUT_DIR=reports/shared_future_candidate_consequence_gate_c
CACHE_DIR=outputs/shared_future_candidate_consequence_gate_c/all
DEFAULT_NAVSIM_PYTHON=python
if [[ -x /root/miniconda3/envs/navsim/bin/python ]]; then
  DEFAULT_NAVSIM_PYTHON=/root/miniconda3/envs/navsim/bin/python
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_NAVSIM_PYTHON}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

JOB_ROOT="${CACHE_DIR}/model_candidates/oracle_jobs"
STORE_DIR="${CACHE_DIR}/model_candidate_oracle_store"
if [[ ! -f "${JOB_ROOT}/model_oracle/job_summary.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" \
    -m tools.shared_future_candidate_consequence.run_oracle_decomposition \
    --mode full \
    --groups O3,O4,O5,O8,O9,O10,O11,O12,O13 \
    --models linear,mlp \
    --epochs "${EPOCHS}" \
    --batch-scenes 256 \
    --device cuda \
    --job-name model_oracle \
    --store-dir "${STORE_DIR}" \
    --job-root "${JOB_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --cache-dir "${CACHE_DIR}"
else
  echo "Reusing completed model-candidate oracle job: ${JOB_ROOT}/model_oracle"
fi

"${PYTHON_BIN}" -m tools.shared_future_candidate_consequence.summarize_model_candidate_oracle \
  --job-dir "${JOB_ROOT}/model_oracle" \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir "${CACHE_DIR}"
