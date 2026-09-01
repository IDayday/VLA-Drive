#!/usr/bin/env bash
set -euo pipefail

# Launch two direct-ranking objectives once the immutable M0-native trainval
# observation cache is complete. This watcher consumes no GPU while waiting.

REPO_ROOT="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/project/DriveVLA-M0-env/bin/python}"
TRAIN_CACHE_ROOT="${TRAIN_CACHE_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/m0_native_multiview_trainval_pool2_tiles4_v1_4shard}"
RUN_ROOT="${RUN_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93}"
LOG_ROOT="${LOG_ROOT:-/root/scorer_pdms93_logs/m0_native_direct_ranking_v1}"
SOURCE_FEATURE_ROOT="${RUN_ROOT}/public_base_features_full_v1"
SOURCE_LABEL_ROOT="${RUN_ROOT}/public_base_labels_full_v1"
SPLIT_MANIFEST="/mnt/project/DriveVLA-M0-scorer-pdms93/reports/scorer_pdms93/OFFICIAL_SCORER_SPLIT.json"
PAIRWISE_OUTPUT="${RUN_ROOT}/m0_native_multiview_direct_pairwise_all64_seed2_v1"
REGRET_OUTPUT="${RUN_ROOT}/m0_native_multiview_direct_listwise_regret_all64_seed2_v1"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-480}"

mkdir -p "${LOG_ROOT}"

manifest_count() {
  find "${TRAIN_CACHE_ROOT}" -name manifest.json -type f 2>/dev/null | wc -l
}

for ((attempt = 0; attempt < WAIT_ATTEMPTS; attempt++)); do
  if [[ "$(manifest_count)" -eq 4 ]]; then
    break
  fi
  sleep 30
done
if [[ "$(manifest_count)" -ne 4 ]]; then
  echo "Timed out before all four M0-native cache manifests appeared" >&2
  exit 1
fi

env PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/navsim/planning/script" \
  "${PYTHON_BIN}" - "${TRAIN_CACHE_ROOT}" <<'PY'
import sys
from pathlib import Path

from local_stage2.train_independent_scorer import load_private_observation_table

table = load_private_observation_table(Path(sys.argv[1]))
assert len(table.tokens) == 103_288, len(table.tokens)
assert len(set(table.tokens)) == 103_288
assert tuple(table.observation_tokens.shape[1:]) == (80, 1536)
assert tuple(table.status_features.shape[1:]) == (11,)
assert table.lineage["current_observation_only"] is True
assert table.lineage["future_or_evaluator_input"] is False
assert table.lineage["official_score_or_factor_input"] is False
assert table.lineage["proposal_input"] is False
print(
    {
        "scene_count": len(table.tokens),
        "observation_shape": list(table.observation_tokens.shape),
        "checkpoint_sha256": table.lineage["checkpoint_sha256"],
    },
    flush=True,
)
PY

test ! -e "${PAIRWISE_OUTPUT}"
test ! -e "${REGRET_OUTPUT}"

COMMON_ARGS=(
  --source public_base "${SOURCE_FEATURE_ROOT}" "${SOURCE_LABEL_ROOT}"
  --private-observation-root "${TRAIN_CACHE_ROOT}"
  --split-manifest "${SPLIT_MANIFEST}"
  --selection-source public_base
  --seed 2
  --epochs 8
  --batch-size 16
  --eval-batch-size 32
  --num-workers 0
  --bootstrap-replicates 1000
  --candidate-keep-count 64
  --fine-top-k 64
  --dynamic-queries 16
  --factor-weight 1
  --factor-loss-mode episode_drive_bce
  --progress-regression-weight 0
  --factor-rank-weight 1
  --consequence-weight 0
  --confidence-weight 0
  --safety-negative-weight 1
)

env CUDA_VISIBLE_DEVICES=5 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/navsim/planning/script" \
  "${PYTHON_BIN}" "${REPO_ROOT}/local_stage2/train_independent_scorer.py" \
  "${COMMON_ARGS[@]}" \
  --pairwise-weight 1 \
  --hard-pairwise-weight 1 \
  --listwise-weight 0 \
  --top-set-weight 0 \
  --expected-regret-weight 0 \
  --top-regret-weight 0 \
  --coarse-loss-weight 1 \
  --output-dir "${PAIRWISE_OUTPUT}" \
  >"${LOG_ROOT}/pairwise.log" 2>&1 &
pairwise_pid=$!

env CUDA_VISIBLE_DEVICES=6 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/navsim/planning/script" \
  "${PYTHON_BIN}" "${REPO_ROOT}/local_stage2/train_independent_scorer.py" \
  "${COMMON_ARGS[@]}" \
  --pairwise-weight 0.5 \
  --hard-pairwise-weight 0.5 \
  --listwise-weight 0.1 \
  --top-set-weight 0.5 \
  --expected-regret-weight 1 \
  --top-regret-weight 1 \
  --coarse-loss-weight 1 \
  --output-dir "${REGRET_OUTPUT}" \
  >"${LOG_ROOT}/listwise_regret.log" 2>&1 &
regret_pid=$!

printf 'pairwise_pid=%s listwise_regret_pid=%s\n' \
  "${pairwise_pid}" "${regret_pid}"

wait "${pairwise_pid}"
wait "${regret_pid}"
echo "M0-native direct-ranking experiments completed"
