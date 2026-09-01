#!/usr/bin/env bash
set -euo pipefail

# Continue the M0-native scorer campaign after the four trainval cache shards
# finish.  Paths are intentionally explicit: output directories are immutable
# experiment identities and are never reused.

REPO_ROOT="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation-bb573a7}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/project/DriveVLA-M0-env/bin/python}"
TRAIN_CACHE_ROOT="${TRAIN_CACHE_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/m0_native_multiview_trainval_pool2_tiles4_v1_4shard}"
TRAIN_CACHE_PRODUCER_PIDS="${TRAIN_CACHE_PRODUCER_PIDS:-1503919 1503920 1503921 1503922}"
RUN_ROOT="/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93"
LOG_ROOT="/root/scorer_pdms93_logs/m0_native_multiview_followup_unweighted_v1"
SOURCE_FEATURE_ROOT="${RUN_ROOT}/public_base_features_full_v1"
SOURCE_LABEL_ROOT="${RUN_ROOT}/public_base_labels_full_v1"
ACTOR_TARGET_ROOT="/mnt/project/DriveVLA-M0-gate-c/outputs/shared_future_candidate_consequence_gate_c/all/oracle_store"
SPLIT_MANIFEST="/mnt/project/DriveVLA-M0-scorer-pdms93/reports/scorer_pdms93/OFFICIAL_SCORER_SPLIT.json"
CONTROL_OUTPUT="${RUN_ROOT}/m0_native_multiview_factoronly_unweighted_rank0_all64_dq16_seed2_v1"
ACTOR_OUTPUT="${RUN_ROOT}/m0_native_multiview_currentactor_aux_w05_unweighted_rank0_all64_dq16_seed2_v1"
NAVTEST_CACHE_ROOT="${RUN_ROOT}/m0_native_multiview_navtest_pool2_tiles4_v1_2shard"

mkdir -p "${LOG_ROOT}"

manifest_count() {
  find "${TRAIN_CACHE_ROOT}" -name manifest.json -type f 2>/dev/null | wc -l
}

while [[ "$(manifest_count)" -lt 4 ]]; do
  live=0
  for pid in ${TRAIN_CACHE_PRODUCER_PIDS}; do
    if kill -0 "${pid}" 2>/dev/null; then
      live=$((live + 1))
    fi
  done
  if [[ "${live}" -eq 0 ]]; then
    echo "All train-cache producers exited before four manifests appeared" >&2
    exit 1
  fi
  sleep 30
done

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
print(
    {
        "trainval_scene_count": len(table.tokens),
        "observation_shape": list(table.observation_tokens.shape),
        "status_shape": list(table.status_features.shape),
        "checkpoint_sha256": table.lineage["checkpoint_sha256"],
    },
    flush=True,
)
PY

test ! -e "${CONTROL_OUTPUT}"
test ! -e "${ACTOR_OUTPUT}"
test ! -e "${NAVTEST_CACHE_ROOT}"

COMMON_TRAIN_ARGS=(
  --source public_base "${SOURCE_FEATURE_ROOT}" "${SOURCE_LABEL_ROOT}"
  --private-observation-root "${TRAIN_CACHE_ROOT}"
  --split-manifest "${SPLIT_MANIFEST}"
  --selection-source public_base
  --seed 2
  --epochs 8
  --batch-size 20
  --eval-batch-size 40
  --num-workers 0
  --bootstrap-replicates 1000
  --candidate-keep-count 64
  --fine-top-k 16
  --dynamic-queries 16
  --pairwise-weight 0
  --hard-pairwise-weight 0
  --listwise-weight 0
  --top-set-weight 0
  --expected-regret-weight 0
  --top-regret-weight 0
  --coarse-loss-weight 0
  --factor-weight 1
  --factor-rank-weight 0
  --consequence-weight 0
  --confidence-weight 0
  --safety-negative-weight 1
)

env CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/navsim/planning/script" \
  "${PYTHON_BIN}" "${REPO_ROOT}/local_stage2/train_independent_scorer.py" \
  "${COMMON_TRAIN_ARGS[@]}" \
  --output-dir "${CONTROL_OUTPUT}" \
  >"${LOG_ROOT}/control.log" 2>&1 &
control_pid=$!

env CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/navsim/planning/script" \
  "${PYTHON_BIN}" "${REPO_ROOT}/local_stage2/train_independent_scorer.py" \
  "${COMMON_TRAIN_ARGS[@]}" \
  --current-actor-target-root "${ACTOR_TARGET_ROOT}" \
  --current-actor-weight 0.5 \
  --output-dir "${ACTOR_OUTPUT}" \
  >"${LOG_ROOT}/current_actor.log" 2>&1 &
actor_pid=$!

COMMON_EXPORT_ARGS=(
  --proposal-pickle /mnt/project/DriveVLA-M0-stage2/runs/ke_candidate_audit/public_base_navtest_proposal_full_fp32/proposal_predictions.pkl
  --repo-root "${REPO_ROOT}"
  --checkpoint /mnt/project/DriveVLA-M0-modelscope/best-epoch_26-step_174312.server_merged.ckpt
  --vlm-path /mnt/project/DriveVLA-M0-models/InternVL3-2B-modelscope
  --log-path /mnt/navsim/test_navsim_logs/test
  --sensor-root /mnt/navsim/test_sensor_blobs/test
  --output-dir "${NAVTEST_CACHE_ROOT}"
  --max-dynamic-tiles 4
  --pool-height 2
  --pool-width 2
  --batch-size 8
  --image-workers 8
  --chunk-size 128
  --shard-count 2
)

env CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/navsim/planning/script" \
  "${PYTHON_BIN}" "${REPO_ROOT}/local_stage2/export_multiview_m0_observation_replay.py" \
  "${COMMON_EXPORT_ARGS[@]}" --shard-index 0 \
  >"${LOG_ROOT}/navtest_shard_0.log" 2>&1 &
navtest_zero_pid=$!

env CUDA_VISIBLE_DEVICES=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/navsim/planning/script" \
  "${PYTHON_BIN}" "${REPO_ROOT}/local_stage2/export_multiview_m0_observation_replay.py" \
  "${COMMON_EXPORT_ARGS[@]}" --shard-index 1 \
  >"${LOG_ROOT}/navtest_shard_1.log" 2>&1 &
navtest_one_pid=$!

printf 'control_pid=%s actor_pid=%s navtest_pids=%s,%s\n' \
  "${control_pid}" "${actor_pid}" "${navtest_zero_pid}" "${navtest_one_pid}"

wait "${navtest_zero_pid}"
wait "${navtest_one_pid}"
env PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/navsim/planning/script" \
  "${PYTHON_BIN}" - "${NAVTEST_CACHE_ROOT}" <<'PY'
import sys
from pathlib import Path

from local_stage2.train_independent_scorer import load_private_observation_table

table = load_private_observation_table(Path(sys.argv[1]))
assert len(table.tokens) == 12_146, len(table.tokens)
assert len(set(table.tokens)) == 12_146
assert tuple(table.observation_tokens.shape[1:]) == (80, 1536)
print({"navtest_scene_count": len(table.tokens), "cache_valid": True}, flush=True)
PY

wait "${control_pid}"
wait "${actor_pid}"
echo "M0 native multiview follow-up completed" 
