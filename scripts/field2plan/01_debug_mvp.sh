#!/usr/bin/env bash
# One-optimizer-step Field2Plan Phase 1 smoke test for the development machine.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
source "$project_root/env.sh"

export FIELD2PLAN_BASELINE_CHECKPOINT="${FIELD2PLAN_BASELINE_CHECKPOINT:-$project_root/navsim_exp/navsim-v1-formal-16gpu-bz2-ga1-flashattn-20260801_193514/final_model/pytorch_model.pt}"
export FIELD2PLAN_DATALIST_PATH="${FIELD2PLAN_DATALIST_PATH:-$project_root/mini_meta.json}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$project_root/navsim_exp}"
export VLM_ATTN_IMPLEMENTATION="${VLM_ATTN_IMPLEMENTATION:-flash_attention_2}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export NAVSIM_NUM_WORKERS="${NAVSIM_NUM_WORKERS:-1}"
export NAVSIM_PREFETCH_FACTOR="${NAVSIM_PREFETCH_FACTOR:-2}"
export NAVSIM_PIN_MEMORY="${NAVSIM_PIN_MEMORY:-0}"
export TRAINING_SKIP_FINAL_SAVE="${TRAINING_SKIP_FINAL_SAVE:-1}"

# The old Qwen token cache does not contain spatial feature maps. Phase 1 must
# read current images and run exactly one visual forward.
unset NAVSIM_FEATURE_CACHE_ROOT NAVSIM_CACHE_COMPONENTS

for required_path in \
  "$BASE_VLM/config.json" \
  "$FIELD2PLAN_BASELINE_CHECKPOINT" \
  "$FIELD2PLAN_DATALIST_PATH" \
  "$DATA_ROOT/meta/mini"; do
  if [ ! -e "$required_path" ]; then
    echo "[field2plan-debug] missing required path: $required_path" >&2
    exit 2
  fi
done

visible_devices="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if ! [[ "$visible_devices" =~ ^[1-9][0-9]*$ ]]; then
  echo "[field2plan-debug] no visible accelerator" >&2
  exit 2
fi
num_processes="${FIELD2PLAN_DEBUG_NUM_PROCESSES:-$visible_devices}"
if (( num_processes > 2 )); then
  num_processes=2
fi
per_device_batch="${FIELD2PLAN_DEBUG_BATCH_SIZE:-2}"
target_effective_batch=32
micro_global=$((num_processes * per_device_batch))
if (( micro_global < 1 || target_effective_batch % micro_global != 0 )); then
  echo "[field2plan-debug] cannot preserve effective batch 32: processes=$num_processes batch=$per_device_batch" >&2
  exit 2
fi
gradient_accumulation=$((target_effective_batch / micro_global))
if (( num_processes > visible_devices )); then
  echo "[field2plan-debug] requested $num_processes processes, visible=$visible_devices" >&2
  exit 2
fi

run_id="${RUN_ID:-field2plan-mvp-debug-$(date +'%Y%m%d_%H%M%S')}"
log_dir="$NAVSIM_EXP_ROOT/launcher_logs"
mkdir -p "$log_dir"
launcher_log="$log_dir/$run_id.log"
export TRITON_CACHE_DIR="${TRITON_LOCAL_CACHE_ROOT:-/tmp/field2plan-triton}/$run_id/node0"
mkdir -p "$TRITON_CACHE_DIR"
exec > >(tee -a "$launcher_log") 2>&1

echo "[field2plan-debug] project_root=$project_root"
echo "[field2plan-debug] run_id=$run_id"
echo "[field2plan-debug] topology=nodes:1 local_processes:$num_processes global_processes:$num_processes visible:$visible_devices"
echo "[field2plan-debug] per_device_batch=$per_device_batch gradient_accumulation=$gradient_accumulation effective_batch=$((micro_global * gradient_accumulation)) target=32"
echo "[field2plan-debug] baseline_checkpoint=$FIELD2PLAN_BASELINE_CHECKPOINT"
echo "[field2plan-debug] data_root=$DATA_ROOT datalist=$FIELD2PLAN_DATALIST_PATH output=$NAVSIM_EXP_ROOT/$run_id"

if [ "${FIELD2PLAN_TOPOLOGY_ONLY:-0}" = "1" ]; then
  exit 0
fi

coordinate_dir="$NAVSIM_EXP_ROOT/$run_id/coordinate_sanity"
python tools/field2plan/visualize_coordinates.py \
  --config starVLA/config/training/cfg_field2plan_mvp_debug.yaml \
  --data-root "$DATA_ROOT" \
  --datalist-path "$FIELD2PLAN_DATALIST_PATH" \
  --split mini \
  --index 0 \
  --output-dir "$coordinate_dir"

python - <<'PY'
import accelerate
import deepspeed
import flash_attn
import torch
from omegaconf import OmegaConf
from starVLA.model.framework.QwenOFT_Field2Plan import Qwenvl_OFT_Field2Plan

cfg = OmegaConf.load("starVLA/config/training/cfg_field2plan_mvp_debug.yaml")
OmegaConf.resolve(cfg)
print(
    "[field2plan-debug] preflight OK:",
    f"torch={torch.__version__}",
    f"accelerate={accelerate.__version__}",
    f"deepspeed={deepspeed.__version__}",
    f"flash_attn={flash_attn.__version__}",
    f"visible={torch.cuda.device_count()}",
    f"framework={Qwenvl_OFT_Field2Plan.__name__}",
)
PY

if [ "${FIELD2PLAN_PREFLIGHT_ONLY:-0}" = "1" ]; then
  exit 0
fi

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "$num_processes" \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_port "${MAIN_PROCESS_PORT:-29691}" \
  --mixed_precision bf16 \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/cfg_field2plan_mvp_debug.yaml \
  --run_id "$run_id" \
  --datasets.vla_data.per_device_batch_size "$per_device_batch" \
  --trainer.gradient_accumulation_steps "$gradient_accumulation" \
  --trainer.max_train_steps 1 \
  --trainer.logging_frequency 1
