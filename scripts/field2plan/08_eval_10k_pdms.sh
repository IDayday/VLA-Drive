#!/usr/bin/env bash
# Evaluate every available Phase-2 step-10k checkpoint with NAVSIM v1.1 PDMS.
#
# The launcher intentionally regenerates predictions with one fixed flow seed.
# This makes per-token comparisons between supervision/access controls paired.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/env.sh"

MODEL_ITER="${MODEL_ITER:-10000}"
INFER_SEED="${INFER_SEED:-20260808}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
EVAL_THREADS="${EVAL_THREADS:-3}"

SHARED_ROOT="${DRIVEDREAMER_SHARED_ROOT:-/mnt/zhangt_workspace/project/DriveDreamer-Policy}"
PRED_ROOT="${PRED_ROOT:-${SHARED_ROOT}/navsim_planning_results/field2plan_step10k_seed20260808}"
EVAL_ROOT="${EVAL_ROOT:-${SHARED_ROOT}/navsim_exp/eval_field2plan_step10k_pdms_v1_1}"
LOG_ROOT="${LOG_ROOT:-${SHARED_ROOT}/navsim_exp/field2plan_step10k_pdms_logs}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${REPO_ROOT}/navsim_exp/metric_cache_navtest_v1_1}"

EXPERIMENTS=(
  field2plan-p2_00_nosup_noaccess-steps100000-seed42
  field2plan-p2_01_nosup_access-steps100000-seed42
  field2plan-p2_10_sup_noaccess_da3-steps100000-seed42
  field2plan-p2_11_sup_access_da3-steps100000-seed42
  field2plan-p2_11_sup_access_vggt-steps100000-seed42
  field2plan-p2_random_access_da3-steps100000-seed42
)

mkdir -p "${PRED_ROOT}" "${EVAL_ROOT}" "${LOG_ROOT}"
test -d "${METRIC_CACHE_PATH}/metadata"

# The repository cache only contains navtrain. Inference must consume raw
# navtest inputs instead of silently looking up incompatible train entries.
unset NAVSIM_FEATURE_CACHE_ROOT

validate_predictions() {
  local prediction_dir="$1"
  python - "${REPO_ROOT}/test_meta.json" "${prediction_dir}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

datalist_path = Path(sys.argv[1])
prediction_dir = Path(sys.argv[2])
tokens = json.loads(datalist_path.read_text(encoding="utf-8"))
expected = set(tokens)
actual = {path.stem for path in prediction_dir.glob("*.npy")}
diagnostics = {path.stem for path in (prediction_dir / "diagnostics").glob("*.npz")}
if actual != expected:
    raise SystemExit(
        f"prediction token mismatch: missing={len(expected - actual)}, "
        f"extra={len(actual - expected)}"
    )
if diagnostics != expected:
    raise SystemExit(
        f"diagnostic token mismatch: missing={len(expected - diagnostics)}, "
        f"extra={len(diagnostics - expected)}"
    )
for token in tokens:
    action = np.load(prediction_dir / f"{token}.npy", allow_pickle=False)
    if action.shape != (8, 3) or action.dtype != np.float32 or not np.isfinite(action).all():
        raise SystemExit(
            f"invalid prediction {token}: shape={action.shape}, "
            f"dtype={action.dtype}, finite={np.isfinite(action).all()}"
        )
print(f"validated {len(tokens)} predictions in {prediction_dir}")
PY
}

run_pdms_v1_1() {
  local short_name="$1"
  local run_root="$2"
  local log_path="${LOG_ROOT}/${short_name}.pdms_v1_1.log"

  (
    export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
    export NAVSIM_EXP_ROOT="${EVAL_ROOT}"
    export NAVSIM_DEVKIT_ROOT="${REPO_ROOT}/navsim_v1.1/navsim"
    export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"
    python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score.py" \
      train_test_split=navtest \
      agent=human_agent \
      experiment_name="${short_name}" \
      metric_cache_path="${METRIC_CACHE_PATH}" \
      pred_dir="${run_root}" \
      split=test \
      worker.threads_per_node="${EVAL_THREADS}" \
      worker.log_to_driver=false
  ) >"${log_path}" 2>&1
}

eval_pid=""
eval_name=""
for experiment in "${EXPERIMENTS[@]}"; do
  checkpoint_dir="${REPO_ROOT}/navsim_exp/${experiment}"
  checkpoint="${checkpoint_dir}/checkpoints/steps_${MODEL_ITER}_pytorch_model.pt"
  short_name="${experiment#field2plan-}"
  short_name="${short_name%-steps100000-seed42}"
  run_root="${PRED_ROOT}/${experiment}-step${MODEL_ITER}"
  prediction_dir="${run_root}/test"

  if [[ ! -s "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi

  echo "[infer] ${experiment}"
  infer_pids=()
  for rank in 0 1; do
    CUDA_VISIBLE_DEVICES="${rank}" python "${REPO_ROOT}/infer.py" \
      --ckpt_dir "${checkpoint_dir}" \
      --model_iter "${MODEL_ITER}" \
      --datalist_path "${REPO_ROOT}/test_meta.json" \
      --data_root "${DATA_ROOT}" \
      --out_dir "${PRED_ROOT}" \
      --split test \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --rank "${rank}" \
      --world_size 2 \
      --seed "${INFER_SEED}" \
      --qwen_forward_mode auto \
      --smooth 0 \
      --save_diagnostics \
      --overwrite \
      >"${LOG_ROOT}/${short_name}.rank${rank}.infer.log" 2>&1 &
    infer_pids+=("$!")
  done

  infer_failed=0
  for infer_pid in "${infer_pids[@]}"; do
    if ! wait "${infer_pid}"; then
      infer_failed=1
    fi
  done
  if [[ "${infer_failed}" -ne 0 ]]; then
    echo "Inference failed for ${experiment}; inspect ${LOG_ROOT}/${short_name}.rank*.infer.log" >&2
    exit 1
  fi
  validate_predictions "${prediction_dir}"

  # Keep at most one CPU evaluator beside the next two-PPU inference job.
  if [[ -n "${eval_pid}" ]]; then
    echo "[wait-pdms] ${eval_name}"
    wait "${eval_pid}"
  fi
  echo "[pdms-v1.1] ${experiment}"
  run_pdms_v1_1 "${short_name}" "${run_root}" &
  eval_pid="$!"
  eval_name="${short_name}"
done

if [[ -n "${eval_pid}" ]]; then
  echo "[wait-pdms] ${eval_name}"
  wait "${eval_pid}"
fi

echo "All step-${MODEL_ITER} NAVSIM v1.1 evaluations completed."
