#!/usr/bin/env bash
# One-optimizer-step Phase-2 DA3 geometry smoke on up to two local PPUs.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
source "$project_root/env.sh"

export FIELD2PLAN_BASELINE_CHECKPOINT="${FIELD2PLAN_BASELINE_CHECKPOINT:-$project_root/navsim_exp/navsim-v1-formal-16gpu-bz2-ga1-flashattn-20260801_193514/final_model/pytorch_model.pt}"
export FIELD2PLAN_DATALIST_PATH="${FIELD2PLAN_DATALIST_PATH:-$project_root/mini_meta.json}"
export FIELD2PLAN_DEBUG_SPLIT="${FIELD2PLAN_DEBUG_SPLIT:-mini}"
export FIELD2PLAN_GEOMETRY_CACHE="${FIELD2PLAN_GEOMETRY_CACHE:-$DRIVEDREAMER_SHARED_ROOT/field2plan_cache/debug_geometry_da3_mini}"
export FIELD2PLAN_GEOMETRY_TEACHER_TYPE="${FIELD2PLAN_GEOMETRY_TEACHER_TYPE:-da3}"
export FIELD2PLAN_DEBUG_DRAFT_CACHE="${FIELD2PLAN_DEBUG_DRAFT_CACHE:-$DRIVEDREAMER_SHARED_ROOT/field2plan_cache/debug_mini_2tokens}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$project_root/navsim_exp}"
export VLM_ATTN_IMPLEMENTATION="${VLM_ATTN_IMPLEMENTATION:-flash_attention_2}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export NAVSIM_NUM_WORKERS="${NAVSIM_NUM_WORKERS:-1}"
export NAVSIM_PREFETCH_FACTOR="${NAVSIM_PREFETCH_FACTOR:-2}"
export NAVSIM_PIN_MEMORY="${NAVSIM_PIN_MEMORY:-0}"
export TRAINING_SKIP_FINAL_SAVE="${TRAINING_SKIP_FINAL_SAVE:-1}"
unset NAVSIM_FEATURE_CACHE_ROOT NAVSIM_CACHE_COMPONENTS

debug_split="$FIELD2PLAN_DEBUG_SPLIT"
use_cached_draft="${FIELD2PLAN_DEBUG_USE_DRAFT_CACHE:-0}"
if [ "$use_cached_draft" != "0" ] && [ "$use_cached_draft" != "1" ]; then
  echo "[field2plan-geometry-debug] FIELD2PLAN_DEBUG_USE_DRAFT_CACHE must be 0 or 1" >&2
  exit 2
fi
if [ "$FIELD2PLAN_GEOMETRY_TEACHER_TYPE" != "da3" ] && [ "$FIELD2PLAN_GEOMETRY_TEACHER_TYPE" != "vggt" ]; then
  echo "[field2plan-geometry-debug] teacher type must be da3 or vggt" >&2
  exit 2
fi

for required_path in \
  "$BASE_VLM/config.json" \
  "$FIELD2PLAN_BASELINE_CHECKPOINT" \
  "$FIELD2PLAN_DATALIST_PATH" \
  "$FIELD2PLAN_GEOMETRY_CACHE/manifest.json" \
  "$DATA_ROOT/meta/$debug_split"; do
  if [ ! -e "$required_path" ]; then
    echo "[field2plan-geometry-debug] missing required path: $required_path" >&2
    exit 2
  fi
done
if [ "$use_cached_draft" = "1" ] && [ ! -f "$FIELD2PLAN_DEBUG_DRAFT_CACHE/manifest.json" ]; then
  echo "[field2plan-geometry-debug] missing draft manifest: $FIELD2PLAN_DEBUG_DRAFT_CACHE/manifest.json" >&2
  exit 2
fi

visible_devices="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if ! [[ "$visible_devices" =~ ^[1-9][0-9]*$ ]]; then
  echo "[field2plan-geometry-debug] no visible accelerator" >&2
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
  echo "[field2plan-geometry-debug] cannot preserve effective batch 32: processes=$num_processes batch=$per_device_batch" >&2
  exit 2
fi
gradient_accumulation=$((target_effective_batch / micro_global))
if (( num_processes > visible_devices )); then
  echo "[field2plan-geometry-debug] requested=$num_processes visible=$visible_devices" >&2
  exit 2
fi

run_id="${RUN_ID:-field2plan-geometry-debug-$(date +'%Y%m%d_%H%M%S')}"
log_dir="$NAVSIM_EXP_ROOT/launcher_logs"
mkdir -p "$log_dir"
launcher_log="$log_dir/$run_id.log"
export TRITON_CACHE_DIR="${TRITON_LOCAL_CACHE_ROOT:-/tmp/field2plan-triton}/$run_id/node0"
mkdir -p "$TRITON_CACHE_DIR"
exec > >(tee -a "$launcher_log") 2>&1

echo "[field2plan-geometry-debug] project_root=$project_root run_id=$run_id"
echo "[field2plan-geometry-debug] topology=nodes:1 local_processes:$num_processes global_processes:$num_processes visible:$visible_devices"
echo "[field2plan-geometry-debug] per_device_batch=$per_device_batch gradient_accumulation=$gradient_accumulation effective_batch=$((micro_global * gradient_accumulation)) target=32"
proposal_args=(
  --field2plan.proposal.source online_debug
  --field2plan.proposal.online_fallback true
  --datasets.vla_data.max_samples null
)
proposal_label="online_debug"
if [ "$use_cached_draft" = "1" ]; then
  proposal_args=(
    --field2plan.proposal.source cache
    --field2plan.proposal.cache_dir "$FIELD2PLAN_DEBUG_DRAFT_CACHE"
    --field2plan.proposal.cache_splits "[$debug_split]"
    --field2plan.proposal.online_fallback false
    --datasets.vla_data.max_samples "${FIELD2PLAN_DEBUG_MAX_SAMPLES:-2}"
  )
  proposal_label="cached"
fi
echo "[field2plan-geometry-debug] proposal=$proposal_label teacher=$FIELD2PLAN_GEOMETRY_TEACHER_TYPE cache=$FIELD2PLAN_GEOMETRY_CACHE"

if [ "${FIELD2PLAN_TOPOLOGY_ONLY:-0}" = "1" ]; then
  exit 0
fi

python - <<'PY'
import json
from pathlib import Path
import accelerate
import deepspeed
import flash_attn
import torch
from starVLA.dataloader.field2plan_cache import GeometryCacheReader

datalist = Path(__import__('os').environ['FIELD2PLAN_DATALIST_PATH'])
tokens = json.loads(datalist.read_text(encoding='utf-8'))
max_samples = int(__import__('os').environ.get('FIELD2PLAN_DEBUG_MAX_SAMPLES', '0'))
if max_samples > 0:
    tokens = tokens[:max_samples]
split = __import__('os').environ['FIELD2PLAN_DEBUG_SPLIT']
reader = GeometryCacheReader(
    __import__('os').environ['FIELD2PLAN_GEOMETRY_CACHE'],
    split,
)
for index in sorted({0, len(tokens) // 2, len(tokens) - 1}):
    reader.load(tokens[index])
print(
    '[field2plan-geometry-debug] preflight OK',
    f'torch={torch.__version__}',
    f'accelerate={accelerate.__version__}',
    f'deepspeed={deepspeed.__version__}',
    f'flash_attn={flash_attn.__version__}',
    f'geometry_entries={reader.manifest["splits"][split]["entry_count"]}',
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
  --main_process_port "${MAIN_PROCESS_PORT:-29692}" \
  --mixed_precision bf16 \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/cfg_field2plan_mvp_debug.yaml \
  --run_id "$run_id" \
  --field2plan.geometry.teacher_type "$FIELD2PLAN_GEOMETRY_TEACHER_TYPE" \
  --field2plan.geometry.cache_dir "$FIELD2PLAN_GEOMETRY_CACHE" \
  --field2plan.geometry.cache_splits "[$debug_split]" \
  --field2plan.geometry.supervision.enabled true \
  --field2plan.geometry.supervision.build_head true \
  "${proposal_args[@]}" \
  --datasets.vla_data.split "$debug_split" \
  --trainer.loss_weights.geometry_depth 0.0002 \
  --trainer.loss_weights.geometry_occupancy 0.002 \
  --trainer.loss_weights.geometry_free_space 0.002 \
  --trainer.loss_weights.geometry_relative 0.002 \
  --datasets.vla_data.per_device_batch_size "$per_device_batch" \
  --trainer.gradient_accumulation_steps "$gradient_accumulation" \
  --trainer.max_train_steps 1 \
  --trainer.logging_frequency 1
