#!/usr/bin/env bash
# Two-PPU, one-optimizer-step Phase-3 end-to-end smoke using offline caches.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
source "$project_root/env.sh"

export FIELD2PLAN_BASELINE_CHECKPOINT="${FIELD2PLAN_BASELINE_CHECKPOINT:-$project_root/navsim_exp/navsim-v1-formal-16gpu-bz2-ga1-flashattn-20260801_193514/final_model/pytorch_model.pt}"
export FIELD2PLAN_DATALIST_PATH="${FIELD2PLAN_DATALIST_PATH:-$project_root/mini_meta.json}"
export FIELD2PLAN_DEBUG_SPLIT="${FIELD2PLAN_DEBUG_SPLIT:-mini}"
export FIELD2PLAN_DRAFT_CACHE="${FIELD2PLAN_DRAFT_CACHE:-$DRIVEDREAMER_SHARED_ROOT/field2plan_cache/debug_mini_2tokens}"
export FIELD2PLAN_GEOMETRY_CACHE="${FIELD2PLAN_GEOMETRY_CACHE:-$DRIVEDREAMER_SHARED_ROOT/field2plan_cache/debug_geometry_da3_mini}"
export FIELD2PLAN_DYNAMICS_CACHE="${FIELD2PLAN_DYNAMICS_CACHE:-$DRIVEDREAMER_SHARED_ROOT/field2plan_cache/debug_dynamics_vjepa2_1_vitl384_mini_2tokens}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$project_root/navsim_exp}"
export VLM_ATTN_IMPLEMENTATION="${VLM_ATTN_IMPLEMENTATION:-flash_attention_2}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export NAVSIM_NUM_WORKERS="${NAVSIM_NUM_WORKERS:-1}"
export NAVSIM_PREFETCH_FACTOR="${NAVSIM_PREFETCH_FACTOR:-2}"
export NAVSIM_PIN_MEMORY="${NAVSIM_PIN_MEMORY:-0}"
export TRAINING_SKIP_FINAL_SAVE="${TRAINING_SKIP_FINAL_SAVE:-1}"
unset NAVSIM_FEATURE_CACHE_ROOT NAVSIM_CACHE_COMPONENTS

required=(
  "$BASE_VLM/config.json"
  "$FIELD2PLAN_BASELINE_CHECKPOINT"
  "$FIELD2PLAN_DATALIST_PATH"
  "$FIELD2PLAN_DRAFT_CACHE/manifest.json"
  "$FIELD2PLAN_GEOMETRY_CACHE/manifest.json"
  "$FIELD2PLAN_DYNAMICS_CACHE/manifest.json"
  "$DATA_ROOT/meta/$FIELD2PLAN_DEBUG_SPLIT"
)
for path in "${required[@]}"; do
  if [ ! -e "$path" ]; then
    echo "[field2plan-phase3-debug] missing required path: $path" >&2
    exit 2
  fi
done

visible_devices="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if ! [[ "$visible_devices" =~ ^[1-9][0-9]*$ ]]; then
  echo "[field2plan-phase3-debug] no visible accelerator" >&2
  exit 2
fi
num_processes="${FIELD2PLAN_DEBUG_NUM_PROCESSES:-$visible_devices}"
if (( num_processes > 2 )); then num_processes=2; fi
per_device_batch="${FIELD2PLAN_DEBUG_BATCH_SIZE:-2}"
target_effective_batch=32
micro_global=$((num_processes * per_device_batch))
if (( micro_global < 1 || target_effective_batch % micro_global != 0 )); then
  echo "[field2plan-phase3-debug] cannot preserve effective batch 32" >&2
  exit 2
fi
gradient_accumulation=$((target_effective_batch / micro_global))

run_id="${RUN_ID:-field2plan-phase3-debug-$(date +'%Y%m%d_%H%M%S')}"
log_dir="$NAVSIM_EXP_ROOT/launcher_logs"
mkdir -p "$log_dir"
launcher_log="$log_dir/$run_id.log"
export TRITON_CACHE_DIR="${TRITON_LOCAL_CACHE_ROOT:-/tmp/field2plan-triton}/$run_id/node0"
mkdir -p "$TRITON_CACHE_DIR"
exec > >(tee -a "$launcher_log") 2>&1

echo "[field2plan-phase3-debug] topology=1x$num_processes batch=$per_device_batch accumulation=$gradient_accumulation effective_batch=32"
echo "[field2plan-phase3-debug] teacher_runtime=offline_cache_only dynamics_cache=$FIELD2PLAN_DYNAMICS_CACHE"
if [ "${FIELD2PLAN_TOPOLOGY_ONLY:-0}" = "1" ]; then
  exit 0
fi

export FIELD2PLAN_DRAFT_MANIFEST_SHA256="$(sha256sum "$FIELD2PLAN_DRAFT_CACHE/manifest.json" | awk '{print $1}')"
export FIELD2PLAN_GEOMETRY_MANIFEST_SHA256="$(sha256sum "$FIELD2PLAN_GEOMETRY_CACHE/manifest.json" | awk '{print $1}')"
export FIELD2PLAN_DYNAMICS_MANIFEST_SHA256="$(sha256sum "$FIELD2PLAN_DYNAMICS_CACHE/manifest.json" | awk '{print $1}')"

python - <<'PY'
import json
import os
from pathlib import Path
import torch
from starVLA.dataloader.field2plan_cache import (
    DraftCacheReader,
    DynamicsCacheReader,
    GeometryCacheReader,
)

datalist = Path(os.environ["FIELD2PLAN_DATALIST_PATH"])
tokens = json.loads(datalist.read_text(encoding="utf-8"))[:2]
split = os.environ["FIELD2PLAN_DEBUG_SPLIT"]
readers = (
    DraftCacheReader(
        os.environ["FIELD2PLAN_DRAFT_CACHE"],
        split,
        os.environ["FIELD2PLAN_DRAFT_MANIFEST_SHA256"],
    ),
    GeometryCacheReader(
        os.environ["FIELD2PLAN_GEOMETRY_CACHE"],
        split,
        os.environ["FIELD2PLAN_GEOMETRY_MANIFEST_SHA256"],
    ),
    DynamicsCacheReader(
        os.environ["FIELD2PLAN_DYNAMICS_CACHE"],
        split,
        os.environ["FIELD2PLAN_DYNAMICS_MANIFEST_SHA256"],
    ),
)
for reader in readers:
    reader.validate_dataset_binding(tokens, str(datalist))
    for token in tokens:
        reader.load(token)
print(
    "[field2plan-phase3-debug] cache preflight OK",
    f"torch={torch.__version__}",
    "tokens=2",
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
  --main_process_port "${MAIN_PROCESS_PORT:-29702}" \
  --mixed_precision bf16 \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/cfg_field2plan_phase3_debug.yaml \
  --run_id "$run_id" \
  --field2plan.proposal.cache_dir "$FIELD2PLAN_DRAFT_CACHE" \
  --field2plan.proposal.manifest_sha256 "$FIELD2PLAN_DRAFT_MANIFEST_SHA256" \
  --field2plan.geometry.cache_dir "$FIELD2PLAN_GEOMETRY_CACHE" \
  --field2plan.geometry.manifest_sha256 "$FIELD2PLAN_GEOMETRY_MANIFEST_SHA256" \
  --field2plan.dynamics.teacher.cache_dir "$FIELD2PLAN_DYNAMICS_CACHE" \
  --field2plan.dynamics.teacher.manifest_sha256 "$FIELD2PLAN_DYNAMICS_MANIFEST_SHA256" \
  --datasets.vla_data.split "$FIELD2PLAN_DEBUG_SPLIT" \
  --datasets.vla_data.max_samples 2 \
  --datasets.vla_data.per_device_batch_size "$per_device_batch" \
  --trainer.gradient_accumulation_steps "$gradient_accumulation" \
  --trainer.max_train_steps 1 \
  --trainer.logging_frequency 1
