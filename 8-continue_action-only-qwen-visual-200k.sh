#!/usr/bin/env bash
# Continue the completed 100k Qwen-visual action-only model to global step 200k.
# The source run contains model weights only, so Adam moments are necessarily
# fresh. Endpoint learning rates are kept constant to avoid an LR restart.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

source_run="${QWEN_VISUAL_SOURCE_RUN:-$NAVSIM_EXP_ROOT/qwen-visual-action-only-20260814_001706}"
source_step="${QWEN_VISUAL_SOURCE_STEP:-100000}"
target_step="${MAX_TRAIN_STEPS:-200000}"
expected_train_samples="${EXPECTED_TRAIN_SAMPLES:-103288}"
datalist="${NAVSIM_DATALIST_PATH:-$DRIVEDREAMER_ROOT/train_meta.json}"
qwen_learning_rate="${QWEN_LEARNING_RATE:-5e-7}"
visual_learning_rate="${VISUAL_LEARNING_RATE:-1e-7}"
action_learning_rate="${ACTION_LEARNING_RATE:-5e-7}"
timestamp="$(date +'%Y%m%d_%H%M%S')"
run_id="${RUN_ID:-qwen-visual-action-only-100k-to-200k-${PAI_JOB_ID:-$timestamp}}"

for pair in \
  "QWEN_VISUAL_SOURCE_STEP:$source_step" \
  "MAX_TRAIN_STEPS:$target_step" \
  "EXPECTED_TRAIN_SAMPLES:$expected_train_samples"; do
  variable="${pair%%:*}"
  value="${pair#*:}"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$variable must be a positive integer, got: $value" >&2
    exit 2
  fi
done
if (( source_step >= target_step )); then
  echo "QWEN_VISUAL_SOURCE_STEP must be smaller than MAX_TRAIN_STEPS" >&2
  exit 2
fi

source_run="$(readlink -m "$source_run")"
datalist="$(readlink -m "$datalist")"
source_checkpoint="$source_run/checkpoints/steps_${source_step}_pytorch_model.pt"
source_config="$source_run/config.yaml"

for required_path in "$source_checkpoint" "$source_config" "$datalist"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Missing Qwen visual continuation asset: $required_path" >&2
    exit 2
  fi
done

python - "$source_config" "$datalist" "$source_step" "$expected_train_samples" <<'PY'
import json
import sys

from omegaconf import OmegaConf

config_path, datalist_path, source_step, expected_count = sys.argv[1:]
source_step = int(source_step)
expected_count = int(expected_count)
config = OmegaConf.load(config_path)

framework = str(OmegaConf.select(config, "framework.name", default=""))
prompt_mode = str(OmegaConf.select(config, "framework.action_prompt_mode", default=""))
freeze_visual = bool(
    OmegaConf.select(config, "framework.qwenvl.freeze_visual", default=True)
)
trained_steps = int(OmegaConf.select(config, "trainer.max_train_steps", default=0))
if framework != "QwenOFT" or prompt_mode != "minimal" or freeze_visual:
    raise SystemExit(
        "Source must be the QwenOFT/minimal action-only experiment with "
        f"freeze_visual=false, got {framework}/{prompt_mode}/freeze_visual={freeze_visual}"
    )
if trained_steps < source_step:
    raise SystemExit(
        f"Source config trained only {trained_steps} steps, cannot continue from {source_step}"
    )

with open(datalist_path, encoding="utf-8") as stream:
    records = json.load(stream)
if not isinstance(records, list):
    raise SystemExit("Full NAVSIM training datalist must be a JSON list")
if len(records) != expected_count:
    raise SystemExit(
        f"Full NAVSIM training datalist count mismatch: {len(records)} != {expected_count}"
    )
if len(set(records)) != expected_count:
    raise SystemExit("Full NAVSIM training datalist contains duplicate tokens")
if not all(isinstance(record, str) and record for record in records):
    raise SystemExit("Full NAVSIM training datalist must contain non-empty token strings")
print(f"[qwen-visual-continue] validated full training set: {len(records)} unique samples")
PY

echo "[qwen-visual-continue] source=$source_checkpoint"
echo "[qwen-visual-continue] global_steps=$source_step->$target_step"
echo "[qwen-visual-continue] datalist=$datalist samples=$expected_train_samples"
echo "[qwen-visual-continue] run_id=$run_id"
echo "[qwen-visual-continue] optimizer=fresh lr=qwen:$qwen_learning_rate visual:$visual_learning_rate action:$action_learning_rate"

exec env \
  RUN_ID="$run_id" \
  NAVSIM_DATALIST_PATH="$datalist" \
  EXPECTED_TRAIN_SAMPLES="$expected_train_samples" \
  MAX_TRAIN_STEPS="$target_step" \
  NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-0}" \
  SAVE_INTERVAL="${SAVE_INTERVAL:-10000}" \
  QWEN_LEARNING_RATE="$qwen_learning_rate" \
  VISUAL_LEARNING_RATE="$visual_learning_rate" \
  ACTION_LEARNING_RATE="$action_learning_rate" \
  QWEN_VISUAL_RESUME_CKPT="$source_checkpoint" \
  QWEN_VISUAL_INITIAL_STEP="$source_step" \
  QWEN_VISUAL_RESUME_STRICT=1 \
  QWEN_VISUAL_RUN_SMOKE_BEFORE_FORMAL=0 \
  bash "$project_root/8-train_action-only-qwen-visual.sh"
