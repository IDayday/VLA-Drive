#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 {hybrid|factor|direct} SHARED_FUTURE_WEIGHT GPU OUTPUT_DIR [auxiliary_only|factorized|factorized_cv_residual]" >&2
  exit 2
fi

score_mode="$1"
shared_future_weight="$2"
gpu="$3"
output_dir="$4"
future_mode="${5:-auxiliary_only}"
if [[ "${score_mode}" != "hybrid" && "${score_mode}" != "factor" && "${score_mode}" != "direct" ]]; then
  echo "Unsupported residual score mode: ${score_mode}" >&2
  exit 2
fi
if [[ "${future_mode}" != "auxiliary_only" && "${future_mode}" != "factorized" && "${future_mode}" != "factorized_cv_residual" ]]; then
  echo "Unsupported shared-future mode: ${future_mode}" >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/project/DriveVLA-M0-env/bin/python}"
RUN_ROOT="${RUN_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93}"
TRAIN_CACHE_ROOT="${TRAIN_CACHE_ROOT:-${RUN_ROOT}/m0_native_multiview_trainval_pool2_tiles4_v1_4shard}"
SHARED_FUTURE_TARGET_ROOT="${SHARED_FUTURE_TARGET_ROOT:-${RUN_ROOT}/shared_future_target_table_v1}"
CURRENT_ACTOR_TARGET_ROOT="${CURRENT_ACTOR_TARGET_ROOT:-/mnt/project/DriveVLA-M0-gate-c/outputs/shared_future_candidate_consequence_gate_c/all/oracle_store}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-/mnt/project/DriveVLA-M0-scorer-pdms93/reports/scorer_pdms93/OFFICIAL_SCORER_SPLIT.json}"
SAFETY_NEGATIVE_WEIGHT="${SAFETY_NEGATIVE_WEIGHT:-1}"
CANDIDATE_RELATIVE_WEIGHT="${CANDIDATE_RELATIVE_WEIGHT:-0}"
CURRENT_ACTOR_WEIGHT="${CURRENT_ACTOR_WEIGHT:-1}"

if [[ "$(find "${TRAIN_CACHE_ROOT}" -name manifest.json -type f 2>/dev/null | wc -l)" -ne 4 ]]; then
  echo "M0-native trainval observation cache is incomplete" >&2
  exit 1
fi
test -f "${SHARED_FUTURE_TARGET_ROOT}/manifest.json"
"${PYTHON_BIN}" - "${SHARED_FUTURE_TARGET_ROOT}/manifest.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
assert payload["scene_count"] == 45378
assert payload["valid_scene_count"] == 45377
assert payload["state_shape"] == [45378, 8, 16, 8]
assert payload["mask_shape"] == [45378, 8, 16]
assert payload["depends_on_logged_future"] is True
assert payload["training_only_target"] is True
assert payload["available_as_model_input_at_inference"] is False
PY
test ! -e "${output_dir}"

export CUDA_VISIBLE_DEVICES="${gpu}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/navsim/planning/script${PYTHONPATH:+:${PYTHONPATH}}"

future_args=()
if [[ "${future_mode}" == "factorized" ]]; then
  future_args+=(--shared-future-relabeling)
elif [[ "${future_mode}" == "factorized_cv_residual" ]]; then
  test -f "${CURRENT_ACTOR_TARGET_ROOT}/current.npy"
  future_args+=(
    --shared-future-relabeling
    --shared-future-constant-velocity-residual
    --current-actor-target-root "${CURRENT_ACTOR_TARGET_ROOT}"
    --current-actor-weight "${CURRENT_ACTOR_WEIGHT}"
  )
fi

exec "${PYTHON_BIN}" "${REPO_ROOT}/local_stage2/train_m0_private_residual_scorer.py" \
  --source public_base "${RUN_ROOT}/public_base_features_full_v1" "${RUN_ROOT}/public_base_labels_full_v1" \
  --private-observation-root "${TRAIN_CACHE_ROOT}" \
  --shared-future-target-root "${SHARED_FUTURE_TARGET_ROOT}" \
  --shared-future-weight "${shared_future_weight}" \
  --candidate-relative-weight "${CANDIDATE_RELATIVE_WEIGHT}" \
  "${future_args[@]}" \
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
  --safety-negative-weight "${SAFETY_NEGATIVE_WEIGHT}" \
  --output-dir "${output_dir}"
