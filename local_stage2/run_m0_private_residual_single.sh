#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 {hybrid|factor|direct} GPU OUTPUT_DIR" >&2
  exit 2
fi

score_mode="$1"
gpu="$2"
output_dir="$3"
if [[ "${score_mode}" != "hybrid" && "${score_mode}" != "factor" && "${score_mode}" != "direct" ]]; then
  echo "Unsupported residual score mode: ${score_mode}" >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/project/DriveVLA-M0-env/bin/python}"
RUN_ROOT="${RUN_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93}"
TRAIN_CACHE_ROOT="${TRAIN_CACHE_ROOT:-${RUN_ROOT}/m0_native_multiview_trainval_pool2_tiles4_v1_4shard}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-/mnt/project/DriveVLA-M0-scorer-pdms93/reports/scorer_pdms93/OFFICIAL_SCORER_SPLIT.json}"

if [[ "$(find "${TRAIN_CACHE_ROOT}" -name manifest.json -type f 2>/dev/null | wc -l)" -ne 4 ]]; then
  echo "M0-native trainval observation cache is incomplete" >&2
  exit 1
fi
test ! -e "${output_dir}"

export CUDA_VISIBLE_DEVICES="${gpu}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/navsim/planning/script${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON_BIN}" "${REPO_ROOT}/local_stage2/train_m0_private_residual_scorer.py" \
  --source public_base "${RUN_ROOT}/public_base_features_full_v1" "${RUN_ROOT}/public_base_labels_full_v1" \
  --private-observation-root "${TRAIN_CACHE_ROOT}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --selection-source public_base \
  --seed 2 \
  --epochs 8 \
  --batch-size 12 \
  --eval-batch-size 24 \
  --num-workers 0 \
  --bootstrap-replicates 1000 \
  --model-dim 256 \
  --dynamic-queries 16 \
  --private-layers 2 \
  --trajectory-layers 2 \
  --candidate-layers 1 \
  --fine-layers 2 \
  --private-fine-top-k 16 \
  --residual-layers 2 \
  --residual-top-k 64 \
  --score-mode "${score_mode}" \
  --max-residual 0.5 \
  --minimum-pair-delta 0.02 \
  --factor-rank-minimum-delta 0.05 \
  --pairwise-weight 1 \
  --base-pairwise-weight 1 \
  --listwise-weight 0.1 \
  --top-set-weight 0.5 \
  --expected-regret-weight 1 \
  --factor-weight 1 \
  --private-factor-weight 0.25 \
  --factor-rank-weight 0.5 \
  --relative-safety-weight 0.5 \
  --residual-l2-weight 0.01 \
  --safety-negative-weight 1 \
  --output-dir "${output_dir}"
