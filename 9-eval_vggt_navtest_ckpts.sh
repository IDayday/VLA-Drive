#!/usr/bin/env bash
# Evaluate several checkpoints from one VGGT-query run on NAVSIM v1.1 navtest.
#
# The inference phase is sharded across EVAL_DEVICE_COUNT accelerators.  The
# checkpoints are evaluated sequentially to avoid loading three copies per
# accelerator and saturating the shared dataset mount.  Existing predictions
# are resumed only when their manifests name the same checkpoint.
#
# Typical DLC invocation:
#   EVAL_DEVICE_COUNT=8 CHECKPOINT_STEPS="10000 20000 30000" \
#     bash 9-eval_vggt_navtest_ckpts.sh

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

run_dir="${MODEL_DIR:-$NAVSIM_EXP_ROOT/vggt-query-full-fixed-20260811_130159}"
checkpoint_steps="${CHECKPOINT_STEPS:-10000 20000 30000}"
device_count="${EVAL_DEVICE_COUNT:-8}"
device_ids="${EVAL_DEVICE_IDS:-}"
batch_size="${BATCH_SIZE:-8}"
num_workers="${NUM_WORKERS:-4}"
overwrite="${OVERWRITE:-0}"
datalist="${DATALIST:-$DRIVEDREAMER_ROOT/test_meta.json}"
metric_cache="${METRIC_CACHE_PATH:-$NAVSIM_EXP_ROOT/metric_cache_navtest_v1_1}"
output_root="${OUT_DIR:-$NAVSIM_EVAL_ROOT/vggt-navtest-ckpt-comparison/predictions}"
evaluation_root="${EVALUATION_ROOT:-$NAVSIM_EVAL_ROOT/vggt-navtest-ckpt-comparison/pdms}"
log_root="${LOG_ROOT:-$NAVSIM_EVAL_ROOT/vggt-navtest-ckpt-comparison/logs}"
preflight_only="${EVAL_PREFLIGHT_ONLY:-0}"

if [[ ! "$device_count" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVAL_DEVICE_COUNT must be a positive integer, got: $device_count" >&2
  exit 1
fi

if [[ -z "$device_ids" ]]; then
  device_ids="$(seq -s, 0 $((device_count - 1)))"
fi
IFS=',' read -r -a devices <<< "$device_ids"
if [[ "${#devices[@]}" -ne "$device_count" ]]; then
  echo "EVAL_DEVICE_IDS has ${#devices[@]} entries but EVAL_DEVICE_COUNT=$device_count" >&2
  exit 1
fi

required_paths=(
  "$run_dir/config.yaml"
  "$datalist"
  "$DATA_ROOT/meta/test"
  "$VGGT_BASE_VLM"
  "$metric_cache/metadata"
  "$NUPLAN_MAPS_ROOT"
  "$NAVSIM_TEST_LOG_ROOT"
  "$NAVSIM_TEST_SENSOR_ROOT"
)
for required_path in "${required_paths[@]}"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required evaluation asset is missing: $required_path" >&2
    exit 1
  fi
done

read -r -a steps <<< "$checkpoint_steps"
if [[ "${#steps[@]}" -eq 0 ]]; then
  echo "CHECKPOINT_STEPS must contain at least one step" >&2
  exit 1
fi
for step in "${steps[@]}"; do
  if [[ ! "$step" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid checkpoint step: $step" >&2
    exit 1
  fi
  checkpoint="$run_dir/checkpoints/steps_${step}_pytorch_model.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Checkpoint is missing: $checkpoint" >&2
    exit 1
  fi
done

visible_count="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if [[ "$visible_count" -lt "$device_count" ]]; then
  echo "Requested $device_count accelerators, but torch sees only $visible_count" >&2
  exit 1
fi

expected_count="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$datalist")"
run_name="$(basename -- "$run_dir")"

echo "[eval] repository:       $DRIVEDREAMER_ROOT"
echo "[eval] run:              $run_dir"
echo "[eval] checkpoints:      ${steps[*]}"
echo "[eval] devices:          ${devices[*]}"
echo "[eval] navtest samples:  $expected_count"
echo "[eval] metric cache:     $metric_cache"
echo "[eval] prediction root:  $output_root"
echo "[eval] PDMS root:        $evaluation_root"
echo "[eval] base VLM:         $VGGT_BASE_VLM"

if [[ "$preflight_only" == "1" ]]; then
  echo "[eval] preflight passed; no inference or scoring was started"
  exit 0
fi

mkdir -p "$output_root" "$evaluation_root" "$log_root"
summary_tmp="$(mktemp "$evaluation_root/summary.csv.tmp.XXXXXX")"
printf '%s\n' \
  'step,pdms,no_at_fault_collisions,drivable_area_compliance,ego_progress,time_to_collision_within_bound,comfort,driving_direction_compliance,result_csv' \
  > "$summary_tmp"

for step in "${steps[@]}"; do
  checkpoint_name="steps_${step}_pytorch_model.pt"
  prediction_run="$output_root/${run_name}-step${step}"
  prediction_dir="$prediction_run/test"
  step_log_root="$log_root/step${step}"
  mkdir -p "$step_log_root"

  existing_count=0
  if [[ -d "$prediction_dir" ]]; then
    existing_count="$(find "$prediction_dir" -maxdepth 1 -type f -name '*.npy' | wc -l)"
  fi
  if [[ "$existing_count" -gt 0 && "$overwrite" != "1" ]]; then
    manifest_count="$(find "$prediction_dir" -maxdepth 1 -type f -name 'inference_manifest.rank*.json' 2>/dev/null | wc -l)"
    if [[ "$manifest_count" -eq 0 ]]; then
      echo "Refusing an unverifiable resume: $prediction_dir has predictions but no manifests" >&2
      echo "Set OVERWRITE=1 to regenerate them explicitly." >&2
      exit 1
    fi
    python - "$prediction_dir" "$step" "$checkpoint_name" <<'PY'
import json
import pathlib
import sys

prediction_dir = pathlib.Path(sys.argv[1])
expected_step = int(sys.argv[2])
expected_name = sys.argv[3]
for manifest_path in prediction_dir.glob("inference_manifest.rank*.json"):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_iter") != expected_step:
        raise SystemExit(f"Resume manifest step mismatch: {manifest_path}")
    if pathlib.Path(manifest.get("checkpoint_file", "")).name != expected_name:
        raise SystemExit(f"Resume checkpoint mismatch: {manifest_path}")
PY
  fi

  echo "[eval][step $step] starting $device_count inference shards"
  pids=()
  for ((rank = 0; rank < device_count; rank++)); do
    device="${devices[$rank]}"
    rank_log="$step_log_root/infer.rank${rank}.log"
    (
      MODEL_DIR="$run_dir" \
      MODEL_ITER="$step" \
      SPLIT=test \
      DATALIST="$datalist" \
      OUT_DIR="$output_root" \
      BATCH_SIZE="$batch_size" \
      NUM_WORKERS="$num_workers" \
      GPU="$device" \
      RANK="$rank" \
      WORLD_SIZE="$device_count" \
      OVERWRITE="$overwrite" \
      INFER_USE_FEATURE_CACHE=0 \
      VLM_ATTN_IMPLEMENTATION="${VLM_ATTN_IMPLEMENTATION:-sdpa}" \
        bash "$project_root/4-infer.sh"
    ) >"$rank_log" 2>&1 &
    pids+=("$!")
  done

  shard_failure=0
  for ((rank = 0; rank < device_count; rank++)); do
    if ! wait "${pids[$rank]}"; then
      echo "[eval][step $step] inference rank $rank failed; see $step_log_root/infer.rank${rank}.log" >&2
      shard_failure=1
    fi
  done
  if [[ "$shard_failure" -ne 0 ]]; then
    exit 1
  fi

  prediction_count="$(find "$prediction_dir" -maxdepth 1 -type f -name '*.npy' | wc -l)"
  if [[ "$prediction_count" -ne "$expected_count" ]]; then
    echo "[eval][step $step] prediction count mismatch: $prediction_count != $expected_count" >&2
    exit 1
  fi
  echo "[eval][step $step] inference complete: $prediction_count predictions"

  step_evaluation_root="$evaluation_root/step${step}"
  mkdir -p "$step_evaluation_root"
  PRED_DIR="$prediction_run" \
  SPLIT=test \
  DATALIST="$datalist" \
  METRIC_CACHE_PATH="$metric_cache" \
  NAVSIM_EVAL_ROOT="$step_evaluation_root" \
    bash "$project_root/5-eval_v1.sh" >"$step_log_root/pdms.log" 2>&1

  result_csv="$(find "$step_evaluation_root" -type f -name '*.csv' -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
  if [[ -z "$result_csv" || ! -f "$result_csv" ]]; then
    echo "[eval][step $step] PDMS result CSV was not produced" >&2
    exit 1
  fi
  python - "$step" "$result_csv" >> "$summary_tmp" <<'PY'
import csv
import sys

step, result_path = sys.argv[1:]
with open(result_path, newline="", encoding="utf-8") as result_file:
    average = next(
        row for row in csv.DictReader(result_file) if row.get("token") == "average"
    )
columns = (
    "score",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
)
writer = csv.writer(sys.stdout, lineterminator="\n")
writer.writerow([step, *(average[column] for column in columns), result_path])
PY
  echo "[eval][step $step] PDMS complete: $result_csv"
done

mv -f "$summary_tmp" "$evaluation_root/summary.csv"
echo "[eval] all checkpoints complete"
echo "[eval] summary: $evaluation_root/summary.csv"
column -s, -t "$evaluation_root/summary.csv" 2>/dev/null || cat "$evaluation_root/summary.csv"
