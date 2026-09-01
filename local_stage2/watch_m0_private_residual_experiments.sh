#!/usr/bin/env bash
set -euo pipefail

# Queue Base-calibrated, M0-native residual scorers behind the two direct
# ranking controls on rl-zt3 GPUs 5/6. The residual heads are zero initialized,
# so an untrained artifact exactly preserves released-M0 selection.

REPO_ROOT="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/project/DriveVLA-M0-env/bin/python}"
RUN_ROOT="${RUN_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93}"
TRAIN_CACHE_ROOT="${TRAIN_CACHE_ROOT:-${RUN_ROOT}/m0_native_multiview_trainval_pool2_tiles4_v1_4shard}"
DIRECT_PAIRWISE="${RUN_ROOT}/m0_native_multiview_direct_pairwise_all64_seed2_v1"
DIRECT_REGRET="${RUN_ROOT}/m0_native_multiview_direct_listwise_regret_all64_seed2_v1"
HYBRID_OUTPUT="${RUN_ROOT}/m0_native_multiview_basecalibrated_hybrid_residual_all64_seed2_v1"
FACTOR_OUTPUT="${RUN_ROOT}/m0_native_multiview_basecalibrated_factor_residual_all64_seed2_v1"
LOG_ROOT="${LOG_ROOT:-/root/scorer_pdms93_logs/m0_native_basecalibrated_residual_v1}"
SPLIT_MANIFEST="/mnt/project/DriveVLA-M0-scorer-pdms93/reports/scorer_pdms93/OFFICIAL_SCORER_SPLIT.json"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-960}"

mkdir -p "${LOG_ROOT}"

run_complete() {
  local summary="$1/training_summary.json"
  [[ -f "${summary}" ]] || return 1
  "${PYTHON_BIN}" - "${summary}" <<'PY' >/dev/null
import json
import sys

payload = json.load(open(sys.argv[1]))
if len(payload.get("history", [])) != 8:
    raise SystemExit(1)
PY
}

for ((attempt = 0; attempt < WAIT_ATTEMPTS; attempt++)); do
  if run_complete "${DIRECT_PAIRWISE}" && run_complete "${DIRECT_REGRET}"; then
    break
  fi
  sleep 30
done
if ! run_complete "${DIRECT_PAIRWISE}" || ! run_complete "${DIRECT_REGRET}"; then
  echo "Timed out before both M0-native direct-ranking controls completed" >&2
  exit 1
fi

test ! -e "${HYBRID_OUTPUT}"
test ! -e "${FACTOR_OUTPUT}"

COMMON_ARGS=(
  --source public_base "${RUN_ROOT}/public_base_features_full_v1" "${RUN_ROOT}/public_base_labels_full_v1"
  --private-observation-root "${TRAIN_CACHE_ROOT}"
  --split-manifest "${SPLIT_MANIFEST}"
  --selection-source public_base
  --seed 2
  --epochs 8
  --batch-size 12
  --eval-batch-size 24
  --num-workers 0
  --bootstrap-replicates 1000
  --model-dim 256
  --dynamic-queries 16
  --private-layers 2
  --trajectory-layers 2
  --candidate-layers 1
  --fine-layers 2
  --private-fine-top-k 16
  --residual-layers 2
  --residual-top-k 64
  --max-residual 0.5
  --minimum-pair-delta 0.02
  --factor-rank-minimum-delta 0.05
  --pairwise-weight 1
  --base-pairwise-weight 1
  --listwise-weight 0.1
  --top-set-weight 0.5
  --expected-regret-weight 1
  --factor-weight 1
  --private-factor-weight 0.25
  --factor-rank-weight 0.5
  --relative-safety-weight 0.5
  --residual-l2-weight 0.01
  --safety-negative-weight 1
)

env CUDA_VISIBLE_DEVICES=5 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/navsim/planning/script" \
  "${PYTHON_BIN}" "${REPO_ROOT}/local_stage2/train_m0_private_residual_scorer.py" \
  "${COMMON_ARGS[@]}" \
  --score-mode hybrid \
  --output-dir "${HYBRID_OUTPUT}" \
  >"${LOG_ROOT}/hybrid.log" 2>&1 &
hybrid_pid=$!

env CUDA_VISIBLE_DEVICES=6 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/navsim/planning/script" \
  "${PYTHON_BIN}" "${REPO_ROOT}/local_stage2/train_m0_private_residual_scorer.py" \
  "${COMMON_ARGS[@]}" \
  --score-mode factor \
  --output-dir "${FACTOR_OUTPUT}" \
  >"${LOG_ROOT}/factor.log" 2>&1 &
factor_pid=$!

printf 'hybrid_residual_pid=%s factor_residual_pid=%s\n' \
  "${hybrid_pid}" "${factor_pid}"

wait "${hybrid_pid}"
wait "${factor_pid}"
echo "M0-private Base-calibrated residual experiments completed"
