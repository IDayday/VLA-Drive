#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 RUN_DIR GPU [OUTPUT_JSON]" >&2
  exit 2
fi

run_dir="$1"
gpu="$2"
output_json="${3:-${run_dir}/shared_future_validation.json}"
REPO_ROOT="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/project/DriveVLA-M0-env/bin/python}"
RUN_ROOT="${RUN_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-/mnt/project/DriveVLA-M0-scorer-pdms93/reports/scorer_pdms93/OFFICIAL_SCORER_SPLIT.json}"
CURRENT_ACTOR_TARGET_ROOT="${CURRENT_ACTOR_TARGET_ROOT:-/mnt/project/DriveVLA-M0-gate-c/outputs/shared_future_candidate_consequence_gate_c/all/oracle_store}"
PRIVATE_OBSERVATION_ROOT="${PRIVATE_OBSERVATION_ROOT:-${RUN_ROOT}/m0_native_multiview_trainval_pool2_tiles4_v1_4shard}"
LOCK_PATH="${LOCK_PATH:-/tmp/m0_native_navtest_gpu.lock}"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-1440}"

run_complete() {
  local summary="${run_dir}/training_summary.json"
  [[ -f "${summary}" ]] || return 1
  "${PYTHON_BIN}" - "${summary}" <<'PY' >/dev/null
import json
import sys

payload = json.load(open(sys.argv[1]))
history = payload.get("history", [])
fold = json.load(open(sys.argv[1].replace("training_summary.json", "fold_manifest.json")))
if not history or len(history) != int(fold["args"]["epochs"]):
    raise SystemExit(1)
PY
}

for ((attempt = 0; attempt < WAIT_ATTEMPTS; attempt++)); do
  if run_complete; then
    break
  fi
  sleep 30
done
if ! run_complete; then
  echo "Timed out before shared-future scorer completed: ${run_dir}" >&2
  exit 1
fi
artifact="${run_dir}/best_m0_private_residual_scorer.pt"
test -f "${artifact}"
test ! -e "${output_json}"

exec 9>"${LOCK_PATH}"
flock -x 9
export CUDA_VISIBLE_DEVICES="${gpu}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/navsim/planning/script${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON_BIN}" "${REPO_ROOT}/local_stage2/evaluate_shared_future_prediction.py" \
  --artifact "${artifact}" \
  --source-name public_base \
  --feature-root "${RUN_ROOT}/public_base_features_full_v1" \
  --label-root "${RUN_ROOT}/public_base_labels_full_v1" \
  --private-observation-root "${PRIVATE_OBSERVATION_ROOT}" \
  --shared-future-target-root "${RUN_ROOT}/shared_future_target_table_v1" \
  --current-actor-target-root "${CURRENT_ACTOR_TARGET_ROOT}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --output "${output_json}" \
  --device cuda \
  --batch-size 32
