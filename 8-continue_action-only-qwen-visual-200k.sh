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
expected_ppus="${QWEN_VISUAL_EXPECTED_PPU_COUNT:-16}"
num_machines="${NUM_MACHINES:-1}"
machine_rank="${MACHINE_RANK:-0}"
local_processes="${LOCAL_NUM_PROCESSES:-${NPROC_PER_NODE:-$expected_ppus}}"
num_processes="${NUM_PROCESSES:-$((num_machines * local_processes))}"
per_device_batch="${PER_DEVICE_BATCH_SIZE:-2}"
gradient_accumulation="${GRADIENT_ACCUMULATION_STEPS:-1}"
target_effective_batch="${TARGET_EFFECTIVE_BATCH_SIZE:-32}"
dlc_dry_run="${QWEN_VISUAL_DLC_DRY_RUN:-0}"
dlc_preflight="${QWEN_VISUAL_DLC_PREFLIGHT:-1}"
dlc_preflight_only="${QWEN_VISUAL_DLC_PREFLIGHT_ONLY:-0}"
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
for pair in \
  "QWEN_VISUAL_EXPECTED_PPU_COUNT:$expected_ppus" \
  "NUM_MACHINES:$num_machines" \
  "MACHINE_RANK:$machine_rank" \
  "LOCAL_NUM_PROCESSES:$local_processes" \
  "NUM_PROCESSES:$num_processes" \
  "PER_DEVICE_BATCH_SIZE:$per_device_batch" \
  "GRADIENT_ACCUMULATION_STEPS:$gradient_accumulation" \
  "TARGET_EFFECTIVE_BATCH_SIZE:$target_effective_batch"; do
  variable="${pair%%:*}"
  value="${pair#*:}"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "$variable must be a non-negative integer, got: $value" >&2
    exit 2
  fi
done
for pair in \
  "QWEN_VISUAL_DLC_DRY_RUN:$dlc_dry_run" \
  "QWEN_VISUAL_DLC_PREFLIGHT:$dlc_preflight" \
  "QWEN_VISUAL_DLC_PREFLIGHT_ONLY:$dlc_preflight_only"; do
  variable="${pair%%:*}"
  value="${pair#*:}"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "$variable must be 0 or 1, got: $value" >&2
    exit 2
  fi
done
if (( num_machines != 1 || machine_rank != 0 )); then
  echo "This launcher requires one DLC node: nodes=$num_machines rank=$machine_rank" >&2
  exit 2
fi
if (( expected_ppus < 1 || local_processes != expected_ppus || num_processes != expected_ppus )); then
  echo "This launcher requires $expected_ppus local/global processes: local=$local_processes global=$num_processes" >&2
  exit 2
fi
effective_batch=$((num_processes * per_device_batch * gradient_accumulation))
if (( effective_batch != target_effective_batch )); then
  echo "Refusing effective batch $effective_batch; expected $target_effective_batch" >&2
  echo "Formula: $num_processes x $per_device_batch x $gradient_accumulation" >&2
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

python - \
  "$source_config" \
  "$datalist" \
  "$source_step" \
  "$expected_train_samples" \
  "$qwen_learning_rate" \
  "$visual_learning_rate" \
  "$action_learning_rate" <<'PY'
import json
import math
import sys

from omegaconf import OmegaConf

(
    config_path,
    datalist_path,
    source_step,
    expected_count,
    qwen_learning_rate,
    visual_learning_rate,
    action_learning_rate,
) = sys.argv[1:]
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

scheduler_type = str(
    OmegaConf.select(config, "trainer.lr_scheduler_type", default="")
)
source_base_lr = float(
    OmegaConf.select(config, "trainer.learning_rate.base", default=0.0)
)
source_min_lr = float(
    OmegaConf.select(
        config,
        "trainer.scheduler_specific_kwargs.min_lr",
        default=0.0,
    )
)
if scheduler_type != "cosine_with_min_lr" or source_base_lr <= 0 or source_min_lr <= 0:
    raise SystemExit(
        "Source must use cosine_with_min_lr with positive base/min learning rates"
    )
endpoint_scale = source_min_lr / source_base_lr
source_rates = {
    "qwen": float(
        OmegaConf.select(
            config,
            "trainer.learning_rate.qwen_vl_interface",
            default=0.0,
        )
    ),
    "visual": float(
        OmegaConf.select(config, "trainer.learning_rate.qwen_visual", default=0.0)
    ),
    "action": float(
        OmegaConf.select(config, "trainer.learning_rate.action_model", default=0.0)
    ),
}
requested_rates = {
    "qwen": float(qwen_learning_rate),
    "visual": float(visual_learning_rate),
    "action": float(action_learning_rate),
}
endpoint_rates = {
    name: source_rate * endpoint_scale
    for name, source_rate in source_rates.items()
}
for name, expected_rate in endpoint_rates.items():
    requested_rate = requested_rates[name]
    if not math.isclose(requested_rate, expected_rate, rel_tol=1e-9, abs_tol=0.0):
        raise SystemExit(
            f"{name} continuation LR must equal source endpoint: "
            f"requested={requested_rate:g} expected={expected_rate:g}"
        )
print(
    "[qwen-visual-continue] validated endpoint learning rates: "
    + " ".join(f"{name}={rate:g}" for name, rate in endpoint_rates.items())
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
echo "[qwen-visual-continue] topology=nodes:$num_machines rank:$machine_rank local:$local_processes global:$num_processes"
echo "[qwen-visual-continue] effective_batch=$effective_batch (per_device=$per_device_batch accumulation=$gradient_accumulation)"
echo "[qwen-visual-continue] optimizer=fresh lr=qwen:$qwen_learning_rate visual:$visual_learning_rate action:$action_learning_rate"

exec env \
  RUN_ID="$run_id" \
  NUM_MACHINES="$num_machines" \
  MACHINE_RANK="$machine_rank" \
  LOCAL_NUM_PROCESSES="$local_processes" \
  NUM_PROCESSES="$num_processes" \
  PER_DEVICE_BATCH_SIZE="$per_device_batch" \
  GRADIENT_ACCUMULATION_STEPS="$gradient_accumulation" \
  TARGET_EFFECTIVE_BATCH_SIZE="$target_effective_batch" \
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
  QWEN_VISUAL_CONSTANT_LEARNING_RATE=1 \
  QWEN_VISUAL_EXPECTED_PPU_COUNT="$expected_ppus" \
  QWEN_VISUAL_DLC_DRY_RUN="$dlc_dry_run" \
  QWEN_VISUAL_DLC_PREFLIGHT="$dlc_preflight" \
  QWEN_VISUAL_DLC_PREFLIGHT_ONLY="$dlc_preflight_only" \
  QWEN_VISUAL_RUN_SMOKE_BEFORE_FORMAL=0 \
  bash "$project_root/8-train_action-only-qwen-visual.sh"
