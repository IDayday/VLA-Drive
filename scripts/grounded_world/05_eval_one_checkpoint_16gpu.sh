#!/usr/bin/env bash
# Evaluate one GroundedWorld checkpoint on NAVSIM-v1.1 PDMS with 16 shards.
set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
source "$project_root/env.sh"

: "${MODEL_DIR:?GroundedWorld run/checkpoint directory is required}"
: "${MODEL_ITER:?checkpoint step is required, e.g. 30000}"
metric_cache_path="${METRIC_CACHE_PATH:-$project_root/navsim_exp/metric_cache_navtest_v1_1}"
test_meta="${TEST_META:-$project_root/test_meta.json}"
prediction_root="${PRED_ROOT:-$project_root/navsim_planning_results/groundedworld}"
eval_root="${EVAL_ROOT:-$project_root/navsim_exp/groundedworld_eval}"
run_name="${EVAL_NAME:-$(basename "$MODEL_DIR")-step${MODEL_ITER}}"
run_root="$prediction_root/$run_name"
mkdir -p "$run_root/logs" "$eval_root"

if [ ! -e "$metric_cache_path/metadata" ]; then
  echo "[groundedworld-eval] missing metric cache: $metric_cache_path" >&2
  exit 2
fi

pids=()
for rank in $(seq 0 15); do
  (
    GPU="$rank" RANK="$rank" WORLD_SIZE=16 \
    MODEL_DIR="$MODEL_DIR" MODEL_ITER="$MODEL_ITER" \
    SPLIT=test DATALIST="$test_meta" OUT_DIR="$run_root" \
    BATCH_SIZE="${BATCH_SIZE:-16}" NUM_WORKERS="${NUM_WORKERS:-2}" \
    INFER_SEED="${INFER_SEED:-20260808}" \
    bash "$project_root/4-infer.sh"
  ) >"$run_root/logs/infer-rank${rank}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if [ "$failed" != "0" ]; then
  echo "[groundedworld-eval] inference failed; inspect $run_root/logs" >&2
  exit 3
fi

prediction_count="$(find "$run_root/test" -maxdepth 1 -type f -name '*.npy' | wc -l)"
expected_count="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$test_meta")"
if [ "$prediction_count" != "$expected_count" ]; then
  echo "[groundedworld-eval] prediction count $prediction_count != $expected_count" >&2
  exit 3
fi

export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NAVSIM_EXP_ROOT="$eval_root"
export NAVSIM_DEVKIT_ROOT="$project_root/navsim_v1.1/navsim"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
  train_test_split=navtest \
  agent=human_agent \
  experiment_name="$run_name" \
  metric_cache_path="$metric_cache_path" \
  pred_dir="$run_root" \
  split=test \
  worker=single_machine_thread_pool \
  worker.max_workers="${EVAL_THREADS:-16}"
