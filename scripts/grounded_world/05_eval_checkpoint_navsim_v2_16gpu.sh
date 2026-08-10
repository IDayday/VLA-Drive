#!/usr/bin/env bash
# Infer and score one checkpoint on NAVSIM-v2 navtest or two-stage navhard.

set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
source "$project_root/env.sh"

: "${MODEL_DIR:?GroundedWorld run directory is required}"
: "${MODEL_ITER:?checkpoint step is required, e.g. 30000}"
suite="${EVAL_SUITE:-navtest}"
case "$suite" in
  navtest)
    infer_split=test
    datalist="${TEST_META:-$project_root/test_meta.json}"
    scorer="$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py"
    ;;
  navhard_two_stage)
    infer_split=navhard_two_stage
    datalist="${NAVHARD_DATALIST:-$project_root/navhard_two_stage_meta.json}"
    scorer="$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py"
    for path in \
      "$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs" \
      "$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles" \
      "$OPENSCENE_DATA_ROOT/meta/navhard_two_stage"; do
      if [ ! -d "$path" ]; then
        echo "[groundedworld-eval] missing navhard input: $path" >&2
        exit 2
      fi
    done
    ;;
  *) echo "[groundedworld-eval] EVAL_SUITE must be navtest or navhard_two_stage" >&2; exit 2 ;;
esac

metric_cache="${METRIC_CACHE_PATH:-$DRIVEDREAMER_SHARED_ROOT/groundedworld_cache/navsim_v2_metric_${suite}}"
prediction_root="${PRED_ROOT:-$project_root/navsim_planning_results/groundedworld_v2}"
eval_root="${EVAL_ROOT:-$project_root/navsim_exp/groundedworld_eval_v2}"
intervention="${GROUNDEDWORLD_INFERENCE_INTERVENTION:-none}"
run_name="${EVAL_NAME:-$(basename "$MODEL_DIR")-step${MODEL_ITER}-${suite}-${intervention}}"
run_parent="$prediction_root/$run_name"
inference_name="$(basename "$MODEL_DIR")-step${MODEL_ITER}"
run_root="$run_parent/$inference_name"
score_experiment="$run_name-score"

for path in "$datalist" "$metric_cache/metadata"; do
  if [ ! -e "$path" ]; then
    echo "[groundedworld-eval] missing required input: $path" >&2
    exit 2
  fi
done
checkpoint="$MODEL_DIR/checkpoints/steps_${MODEL_ITER}_pytorch_model.pt"
if [ ! -f "$checkpoint" ]; then
  echo "[groundedworld-eval] missing checkpoint: $checkpoint" >&2
  exit 2
fi
actual_devices="$(python -c 'import torch; print(torch.cuda.device_count())')"
if [ "${EVAL_TOPOLOGY_ONLY:-0}" != "1" ] && (( actual_devices < 16 )); then
  echo "[groundedworld-eval] need 16 visible accelerators, found $actual_devices" >&2
  exit 2
fi
if [ "${EVAL_TOPOLOGY_ONLY:-0}" = "1" ]; then
  echo "[groundedworld-eval] preflight suite=$suite datalist=$datalist cache=$metric_cache"
  exit 0
fi

mkdir -p "$run_parent/logs" "$eval_root"
pids=()
for rank in $(seq 0 15); do
  (
    GPU="$rank" RANK="$rank" WORLD_SIZE=16 \
    MODEL_DIR="$MODEL_DIR" MODEL_ITER="$MODEL_ITER" \
    SPLIT="$infer_split" DATALIST="$datalist" OUT_DIR="$run_parent" \
    BATCH_SIZE="${BATCH_SIZE:-16}" NUM_WORKERS="${NUM_WORKERS:-2}" \
    INFER_SEED="${INFER_SEED:-20260808}" OVERWRITE="${OVERWRITE:-0}" \
    GROUNDEDWORLD_INFERENCE_INTERVENTION="$intervention" \
    bash "$project_root/4-infer.sh"
  ) >"$run_parent/logs/infer-rank${rank}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if [ "$failed" != "0" ]; then
  echo "[groundedworld-eval] inference failed; inspect $run_parent/logs" >&2
  exit 3
fi

prediction_count="$(find "$run_root/$infer_split" -maxdepth 1 -type f -name '*.npy' | wc -l)"
expected_count="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$datalist")"
if [ "$prediction_count" != "$expected_count" ]; then
  echo "[groundedworld-eval] predictions=$prediction_count expected=$expected_count" >&2
  exit 3
fi

export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NAVSIM_EXP_ROOT="$eval_root"
export NAVSIM_DEVKIT_ROOT="$project_root/navsim"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
python "$scorer" \
  train_test_split="$suite" \
  agent=human_agent \
  experiment_name="$score_experiment" \
  metric_cache_path="$metric_cache" \
  pred_dir="$run_root" \
  split="$infer_split" \
  worker=single_machine_thread_pool \
  worker.max_workers="${EVAL_THREADS:-16}" \
  worker.use_process_pool=true

result_csv="$(find "$eval_root/$score_experiment" -type f -name '*.csv' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
if [ -z "$result_csv" ] || [ ! -f "$result_csv" ]; then
  echo "[groundedworld-eval] scorer did not produce a CSV" >&2
  exit 3
fi
python tools/grounded_world/validate_navsim_v2_results.py \
  --csv "$result_csv" \
  --suite "$suite" \
  --expected-scenarios "$expected_count" \
  --output "$run_root/${suite}_summary.json"
