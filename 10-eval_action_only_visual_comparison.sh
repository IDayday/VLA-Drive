#!/usr/bin/env bash
# Reproducible 8-accelerator comparison of frozen-visual and trainable-visual
# action-only checkpoints. NAVTEST uses all 12,146 scenes; train diagnostics
# use one fixed sampled subset and report physical ADE/FDE/heading error.

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

frozen_run="${FROZEN_ACTION_ONLY_RUN_DIR:-$NAVSIM_EXP_ROOT/frozen-visual-action-only-20260804}"
visual_run="${VISUAL_ACTION_ONLY_RUN_DIR:-$NAVSIM_EXP_ROOT/qwen-visual-action-only-20260812_223838}"
comparison_root="${ACTION_ONLY_COMPARISON_ROOT:-$NAVSIM_EVAL_ROOT/action-only-visual-comparison}"
checkpoint_steps="${ACTION_ONLY_COMPARISON_STEPS:-30000 20000 10000}"
device_count="${EVAL_DEVICE_COUNT:-8}"
device_ids="${EVAL_DEVICE_IDS:-0,1,2,3,4,5,6,7}"
batch_size="${BATCH_SIZE:-8}"
num_workers="${NUM_WORKERS:-4}"
inference_seed="${INFER_SEED:-42}"
train_subset_size="${TRAIN_SUBSET_SIZE:-512}"
train_subset_seed="${TRAIN_SUBSET_SEED:-20260813}"
overwrite="${OVERWRITE:-0}"
restart_partial="${RESTART_PARTIAL:-1}"
preflight_only="${COMPARISON_PREFLIGHT_ONLY:-0}"
datalist="${DATALIST:-$DRIVEDREAMER_ROOT/test_meta.json}"
train_datalist="${TRAIN_DATALIST:-$DRIVEDREAMER_ROOT/train_meta.json}"
metric_cache="${METRIC_CACHE_PATH:-$NAVSIM_EXP_ROOT/metric_cache_navtest_v1_1}"
helper="$project_root/tools/evaluate_action_only_visual_comparison.py"

for value_name in device_count batch_size num_workers train_subset_size inference_seed train_subset_seed; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "$value_name must be a non-negative integer, got: $value" >&2
    exit 1
  fi
done
if [[ "$device_count" -lt 1 || "$batch_size" -lt 1 ]]; then
  echo "EVAL_DEVICE_COUNT and BATCH_SIZE must be positive" >&2
  exit 1
fi
if [[ "$overwrite" != "0" && "$overwrite" != "1" ]]; then
  echo "OVERWRITE must be 0 or 1" >&2
  exit 1
fi
if [[ "$restart_partial" != "0" && "$restart_partial" != "1" ]]; then
  echo "RESTART_PARTIAL must be 0 or 1" >&2
  exit 1
fi

read -r -a steps <<< "$checkpoint_steps"
if [[ "${#steps[@]}" -eq 0 ]]; then
  echo "ACTION_ONLY_COMPARISON_STEPS must not be empty" >&2
  exit 1
fi
declare -A seen_steps=()
for step in "${steps[@]}"; do
  if [[ ! "$step" =~ ^[1-9][0-9]*$ || -n "${seen_steps[$step]:-}" ]]; then
    echo "Steps must be unique positive integers, got: $checkpoint_steps" >&2
    exit 1
  fi
  seen_steps[$step]=1
done

frozen_run="$(readlink -m "$frozen_run")"
visual_run="$(readlink -m "$visual_run")"
comparison_root="$(readlink -m "$comparison_root")"
if [[ "$frozen_run" == "$visual_run" ]]; then
  echo "Frozen and visual run directories must be different" >&2
  exit 1
fi
if [[ "$(basename -- "$frozen_run")" == "$(basename -- "$visual_run")" ]]; then
  echo "Run directory basenames must differ because they namespace predictions" >&2
  exit 1
fi

required_paths=(
  "$helper"
  "$frozen_run/config.yaml"
  "$visual_run/config.yaml"
  "$datalist"
  "$DATA_ROOT/meta/test"
  "$BASE_VLM"
  "$metric_cache/metadata"
  "$NUPLAN_MAPS_ROOT"
  "$NAVSIM_TEST_LOG_ROOT"
  "$NAVSIM_TEST_SENSOR_ROOT"
)
if [[ "$train_subset_size" -gt 0 ]]; then
  required_paths+=("$train_datalist" "$DATA_ROOT/meta/train")
fi
for required_path in "${required_paths[@]}"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required action-only comparison asset is missing: $required_path" >&2
    exit 1
  fi
done
for arm_run in "$frozen_run" "$visual_run"; do
  for step in "${steps[@]}"; do
    checkpoint="$arm_run/checkpoints/steps_${step}_pytorch_model.pt"
    if [[ ! -f "$checkpoint" ]]; then
      echo "Required comparison checkpoint is missing: $checkpoint" >&2
      exit 1
    fi
  done
done

python - "$frozen_run/config.yaml" "$visual_run/config.yaml" <<'PY'
import sys
from omegaconf import OmegaConf

for path in sys.argv[1:]:
    config = OmegaConf.load(path)
    framework = str(OmegaConf.select(config, "framework.name", default=""))
    prompt = str(OmegaConf.select(config, "framework.action_prompt_mode", default=""))
    if framework != "QwenOFT" or prompt != "minimal":
        raise SystemExit(
            f"Expected QwenOFT/minimal action-only config, got {framework}/{prompt}: {path}"
        )
PY

echo "[comparison] repository:       $DRIVEDREAMER_ROOT"
echo "[comparison] frozen run:      $frozen_run"
echo "[comparison] visual run:      $visual_run"
echo "[comparison] steps:           ${steps[*]}"
echo "[comparison] devices:         $device_ids"
echo "[comparison] batch/rank:      $batch_size"
echo "[comparison] inference seed:  $inference_seed"
echo "[comparison] train subset:    $train_subset_size (seed $train_subset_seed)"
echo "[comparison] output:          $comparison_root"

# Let the existing evaluator verify accelerator visibility and all NAVTEST
# dependencies without creating output files.
for arm_run in "$frozen_run" "$visual_run"; do
  MODEL_DIR="$arm_run" \
  CHECKPOINT_STEPS="${steps[0]}" \
  EVAL_DEVICE_COUNT="$device_count" \
  EVAL_DEVICE_IDS="$device_ids" \
  BATCH_SIZE="$batch_size" \
  NUM_WORKERS="$num_workers" \
  INFER_SEED="$inference_seed" \
  DATALIST="$datalist" \
  METRIC_CACHE_PATH="$metric_cache" \
  EVAL_PREFLIGHT_ONLY=1 \
  VLM_ATTN_IMPLEMENTATION="${VLM_ATTN_IMPLEMENTATION:-sdpa}" \
    bash "$project_root/9-eval_vggt_navtest_ckpts.sh"
done
if [[ "$preflight_only" == "1" ]]; then
  echo "[comparison] preflight passed; no predictions or scores were written"
  exit 0
fi

mkdir -p "$comparison_root"
exec > >(tee -a "$comparison_root/pipeline.log") 2>&1

python "$helper" build-manifest \
  --frozen-run "$frozen_run" \
  --visual-run "$visual_run" \
  --steps "${steps[@]}" \
  --seed "$inference_seed" \
  --world-size "$device_count" \
  --batch-size "$batch_size" \
  --output "$comparison_root/evaluation_manifest.json"

if [[ "$train_subset_size" -gt 0 ]]; then
  python "$helper" make-train-subset \
    --datalist "$train_datalist" \
    --data-root "$DATA_ROOT" \
    --output-dir "$comparison_root/train_subset" \
    --size "$train_subset_size" \
    --seed "$train_subset_seed"
fi

expected_test_count="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$datalist")"

choose_overwrite() {
  local prediction_dir="$1"
  local expected_count="$2"
  local count=0
  if [[ -d "$prediction_dir" ]]; then
    count="$(find "$prediction_dir" -maxdepth 1 -type f -name '*.npy' | wc -l)"
  fi
  if [[ "$overwrite" == "1" ]]; then
    echo 1
  elif [[ "$count" -eq 0 || "$count" -eq "$expected_count" ]]; then
    echo 0
  elif [[ "$restart_partial" == "1" ]]; then
    echo "[comparison] partial prediction set $prediction_dir ($count/$expected_count); restarting it" >&2
    echo 1
  else
    echo "Partial prediction set requires OVERWRITE=1 or RESTART_PARTIAL=1: $prediction_dir" >&2
    return 1
  fi
}

validate_train_manifests() {
  local prediction_dir="$1"
  python - "$prediction_dir" "$device_count" "$inference_seed" <<'PY'
import json
import pathlib
import sys

directory = pathlib.Path(sys.argv[1])
world_size = int(sys.argv[2])
seed = int(sys.argv[3])
paths = sorted(directory.glob("inference_manifest.rank*.json"))
if len(paths) != world_size:
    raise SystemExit(f"Expected {world_size} train manifests in {directory}, found {len(paths)}")
ranks = set()
for path in paths:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rank = manifest.get("rank")
    if manifest.get("split") != "train" or manifest.get("world_size") != world_size:
        raise SystemExit(f"Train manifest topology mismatch: {path}")
    if manifest.get("seed") != seed or manifest.get("effective_seed") != seed + rank:
        raise SystemExit(f"Train manifest seed mismatch: {path}")
    ranks.add(rank)
if ranks != set(range(world_size)):
    raise SystemExit(f"Train manifest rank coverage mismatch: {sorted(ranks)}")
PY
}

run_train_subset() {
  local arm="$1"
  local arm_run="$2"
  local step="$3"
  local run_name
  run_name="$(basename -- "$arm_run")"
  local prediction_dir="$comparison_root/predictions/${run_name}-step${step}/train"
  local arm_overwrite
  arm_overwrite="$(choose_overwrite "$prediction_dir" "$train_subset_size")"
  if [[ "$arm_overwrite" == "0" && -d "$prediction_dir" ]]; then
    local count
    count="$(find "$prediction_dir" -maxdepth 1 -type f -name '*.npy' | wc -l)"
    if [[ "$count" -eq "$train_subset_size" ]]; then
      validate_train_manifests "$prediction_dir"
    fi
  fi

  local log_dir="$comparison_root/logs/train/$arm/step$step"
  mkdir -p "$log_dir"
  IFS=',' read -r -a devices <<< "$device_ids"
  local pids=()
  for ((rank = 0; rank < device_count; rank++)); do
    (
      MODEL_DIR="$arm_run" \
      MODEL_ITER="$step" \
      SPLIT=train \
      DATALIST="$comparison_root/train_subset/train_subset.json" \
      OUT_DIR="$comparison_root/predictions" \
      BATCH_SIZE="$batch_size" \
      NUM_WORKERS="$num_workers" \
      GPU="${devices[$rank]}" \
      RANK="$rank" \
      WORLD_SIZE="$device_count" \
      OVERWRITE="$arm_overwrite" \
      INFER_SEED="$inference_seed" \
      INFER_USE_FEATURE_CACHE=0 \
      VLM_ATTN_IMPLEMENTATION="${VLM_ATTN_IMPLEMENTATION:-sdpa}" \
        bash "$project_root/4-infer.sh"
    ) >"$log_dir/infer.rank${rank}.log" 2>&1 &
    pids+=("$!")
  done
  local failure=0
  for ((rank = 0; rank < device_count; rank++)); do
    if ! wait "${pids[$rank]}"; then
      echo "[comparison][$arm][step $step] train inference rank $rank failed: $log_dir/infer.rank${rank}.log" >&2
      failure=1
    fi
  done
  [[ "$failure" -eq 0 ]] || return 1
  local prediction_count
  prediction_count="$(find "$prediction_dir" -maxdepth 1 -type f -name '*.npy' | wc -l)"
  if [[ "$prediction_count" -ne "$train_subset_size" ]]; then
    echo "Train prediction count mismatch: $prediction_count != $train_subset_size" >&2
    return 1
  fi
  validate_train_manifests "$prediction_dir"
  python "$helper" score-train \
    --ground-truth "$comparison_root/train_subset/train_subset_ground_truth.npz" \
    --prediction-dir "$prediction_dir" \
    --arm "$arm" \
    --step "$step" \
    --output "$comparison_root/train_metrics/$arm/step${step}.json"
}

for step in "${steps[@]}"; do
  echo "[comparison][priority] evaluating paired step $step"
  for arm in frozen visual; do
    if [[ "$arm" == "frozen" ]]; then
      arm_run="$frozen_run"
    else
      arm_run="$visual_run"
    fi
    run_name="$(basename -- "$arm_run")"
    prediction_dir="$comparison_root/predictions/${run_name}-step${step}/test"
    arm_overwrite="$(choose_overwrite "$prediction_dir" "$expected_test_count")"
    MODEL_DIR="$arm_run" \
    CHECKPOINT_STEPS="$step" \
    EVAL_DEVICE_COUNT="$device_count" \
    EVAL_DEVICE_IDS="$device_ids" \
    BATCH_SIZE="$batch_size" \
    NUM_WORKERS="$num_workers" \
    OVERWRITE="$arm_overwrite" \
    INFER_SEED="$inference_seed" \
    DATALIST="$datalist" \
    METRIC_CACHE_PATH="$metric_cache" \
    OUT_DIR="$comparison_root/predictions" \
    EVALUATION_ROOT="$comparison_root/pdms/$arm/step$step" \
    LOG_ROOT="$comparison_root/logs/navtest/$arm/step$step" \
    VLM_ATTN_IMPLEMENTATION="${VLM_ATTN_IMPLEMENTATION:-sdpa}" \
      bash "$project_root/9-eval_vggt_navtest_ckpts.sh"
  done

  if [[ "$train_subset_size" -gt 0 ]]; then
    run_train_subset frozen "$frozen_run" "$step"
    run_train_subset visual "$visual_run" "$step"
  fi
done

python "$helper" summarize --root "$comparison_root" --steps "${steps[@]}"
echo "[comparison] complete"
echo "[comparison] CSV:    $comparison_root/paired_summary.csv"
echo "[comparison] report: $comparison_root/REPORT.md"
