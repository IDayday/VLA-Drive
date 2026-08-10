#!/usr/bin/env bash
# Formal Phase-3 one-node/16-PPU action-free dynamics training launcher.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
source "$project_root/env.sh"

export FIELD2PLAN_BASELINE_CHECKPOINT="${FIELD2PLAN_BASELINE_CHECKPOINT:-$project_root/navsim_exp/navsim-v1-formal-16gpu-bz2-ga1-flashattn-20260801_193514/final_model/pytorch_model.pt}"
export FIELD2PLAN_DATALIST_PATH="${FIELD2PLAN_DATALIST_PATH:-$project_root/train_meta.json}"
export FIELD2PLAN_DRAFT_CACHE="${FIELD2PLAN_DRAFT_CACHE:-$DRIVEDREAMER_SHARED_ROOT/field2plan_cache/baseline_drafts_193514_seed20260808_steps10}"
export FIELD2PLAN_GEOMETRY_CACHE="${FIELD2PLAN_GEOMETRY_CACHE:-$DRIVEDREAMER_SHARED_ROOT/field2plan_cache/geometry_da3_metric_v1}"
export FIELD2PLAN_DYNAMICS_CACHE="${FIELD2PLAN_DYNAMICS_CACHE:-$DRIVEDREAMER_SHARED_ROOT/field2plan_cache/dynamics_vjepa2_1_vitl384_c96_16_v1}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$DRIVEDREAMER_SHARED_ROOT/navsim_exp}"
export VLM_ATTN_IMPLEMENTATION="${VLM_ATTN_IMPLEMENTATION:-flash_attention_2}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TRAINING_SKIP_FINAL_SAVE=0
unset NAVSIM_FEATURE_CACHE_ROOT NAVSIM_CACHE_COMPONENTS

geometry_supervision="${FIELD2PLAN_GEOMETRY_SUPERVISION:-1}"
geometry_access="${FIELD2PLAN_GEOMETRY_ACCESS:-1}"
dynamics_supervision="${FIELD2PLAN_DYNAMICS_SUPERVISION:-1}"
dynamics_access="${FIELD2PLAN_DYNAMICS_ACCESS:-1}"
dynamics_teacher_mode="${FIELD2PLAN_DYNAMICS_TEACHER_MODE:-real}"
run_seed="${FIELD2PLAN_RUN_SEED:-42}"
export FIELD2PLAN_GEOMETRY_SUPERVISION="$geometry_supervision"
export FIELD2PLAN_GEOMETRY_ACCESS="$geometry_access"
export FIELD2PLAN_DYNAMICS_SUPERVISION="$dynamics_supervision"
export FIELD2PLAN_DYNAMICS_ACCESS="$dynamics_access"
for flag_name in geometry_supervision geometry_access dynamics_supervision dynamics_access; do
  flag_value="${!flag_name}"
  if [ "$flag_value" != "0" ] && [ "$flag_value" != "1" ]; then
    echo "[field2plan-phase3] $flag_name must be 0 or 1" >&2
    exit 2
  fi
done
if ! [[ "$run_seed" =~ ^[0-9]+$ ]]; then
  echo "[field2plan-phase3] FIELD2PLAN_RUN_SEED must be non-negative" >&2
  exit 2
fi
case "$dynamics_teacher_mode" in
  real|batch_shuffled|temporal_shuffled)
    if [ "$dynamics_supervision" != "1" ]; then
      echo "[field2plan-phase3] teacher mode $dynamics_teacher_mode requires dynamics supervision" >&2
      exit 2
    fi
    ;;
  equal_capacity)
    if [ "$dynamics_supervision" != "0" ]; then
      echo "[field2plan-phase3] equal_capacity requires dynamics supervision=0" >&2
      exit 2
    fi
    ;;
  *)
    echo "[field2plan-phase3] unsupported dynamics teacher mode: $dynamics_teacher_mode" >&2
    exit 2
    ;;
esac

actual_devices="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
num_machines="${NUM_MACHINES:-${WORLD_SIZE:-1}}"
machine_rank="${MACHINE_RANK:-${RANK:-0}}"
local_processes="${LOCAL_NUM_PROCESSES:-16}"
global_processes=$((num_machines * local_processes))
per_device_batch="${PER_DEVICE_BATCH_SIZE:-2}"
gradient_accumulation="${GRADIENT_ACCUMULATION_STEPS:-1}"
target_effective_batch=32
effective_batch=$((global_processes * per_device_batch * gradient_accumulation))
if (( num_machines != 1 || machine_rank != 0 || local_processes != 16 || global_processes != 16 )); then
  echo "[field2plan-phase3] formal topology must be one node x 16 processes; nodes=$num_machines rank=$machine_rank local=$local_processes global=$global_processes" >&2
  exit 2
fi
if (( effective_batch != target_effective_batch )); then
  echo "[field2plan-phase3] effective batch mismatch: $global_processes x $per_device_batch x $gradient_accumulation = $effective_batch, expected 32" >&2
  exit 2
fi
if [ "${FIELD2PLAN_TOPOLOGY_ONLY:-0}" != "1" ] && (( actual_devices < local_processes )); then
  echo "[field2plan-phase3] need 16 visible accelerators, found $actual_devices" >&2
  exit 2
fi

host_cpus="$(getconf _NPROCESSORS_ONLN)"
export NAVSIM_NUM_WORKERS="${NAVSIM_NUM_WORKERS:-6}"
export NAVSIM_PREFETCH_FACTOR="${NAVSIM_PREFETCH_FACTOR:-4}"
export NAVSIM_PIN_MEMORY="${NAVSIM_PIN_MEMORY:-1}"
if [ "${FIELD2PLAN_TOPOLOGY_ONLY:-0}" != "1" ] && (( NAVSIM_NUM_WORKERS * local_processes + local_processes > host_cpus )); then
  echo "[field2plan-phase3] CPU oversubscription: workers=$NAVSIM_NUM_WORKERS ranks=$local_processes cpus=$host_cpus" >&2
  exit 2
fi
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

job_tag="${PAI_JOB_ID:-$(date +'%Y%m%d_%H%M%S')}"
run_id="${RUN_ID:-field2plan-phase3-g${geometry_supervision}${geometry_access}-d${dynamics_supervision}${dynamics_access}-${dynamics_teacher_mode}-${job_tag}}"
run_dir="$NAVSIM_EXP_ROOT/$run_id"
launcher_log_dir="$NAVSIM_EXP_ROOT/launcher_logs"
mkdir -p "$launcher_log_dir"
launcher_log="$launcher_log_dir/$run_id.log"
export TRITON_CACHE_DIR="${TRITON_LOCAL_CACHE_ROOT:-/tmp/field2plan-triton}/$run_id/node0"
mkdir -p "$TRITON_CACHE_DIR"
exec > >(tee -a "$launcher_log") 2>&1

echo "[field2plan-phase3] project_root=$project_root run_id=$run_id"
echo "[field2plan-phase3] topology=1x16 batch=$per_device_batch accumulation=$gradient_accumulation effective_batch=$effective_batch"
echo "[field2plan-phase3] geometry=supervision:$geometry_supervision,access:$geometry_access"
echo "[field2plan-phase3] dynamics=supervision:$dynamics_supervision,access:$dynamics_access,teacher:$dynamics_teacher_mode"
echo "[field2plan-phase3] teacher_runtime=offline_cache_only seed=$run_seed"
echo "[field2plan-phase3] draft=$FIELD2PLAN_DRAFT_CACHE dynamics=$FIELD2PLAN_DYNAMICS_CACHE"

if [ "${FIELD2PLAN_TOPOLOGY_ONLY:-0}" = "1" ]; then
  exit 0
fi

required=(
  "$BASE_VLM/config.json"
  "$FIELD2PLAN_BASELINE_CHECKPOINT"
  "$FIELD2PLAN_DATALIST_PATH"
  "$FIELD2PLAN_DRAFT_CACHE/manifest.json"
  "$DATA_ROOT/meta/train"
)
if [ "$geometry_supervision" = "1" ]; then
  required+=("$FIELD2PLAN_GEOMETRY_CACHE/manifest.json")
fi
if [ "$dynamics_supervision" = "1" ]; then
  required+=("$FIELD2PLAN_DYNAMICS_CACHE/manifest.json")
fi
for path in "${required[@]}"; do
  if [ ! -e "$path" ]; then
    echo "[field2plan-phase3] missing required path: $path" >&2
    exit 2
  fi
done

export FIELD2PLAN_DRAFT_MANIFEST_SHA256="${FIELD2PLAN_DRAFT_MANIFEST_SHA256:-$(sha256sum "$FIELD2PLAN_DRAFT_CACHE/manifest.json" | awk '{print $1}')}"
geometry_args=()
if [ "$geometry_supervision" = "1" ]; then
  export FIELD2PLAN_GEOMETRY_MANIFEST_SHA256="${FIELD2PLAN_GEOMETRY_MANIFEST_SHA256:-$(sha256sum "$FIELD2PLAN_GEOMETRY_CACHE/manifest.json" | awk '{print $1}')}"
  geometry_args=(--field2plan.geometry.manifest_sha256 "$FIELD2PLAN_GEOMETRY_MANIFEST_SHA256")
fi
dynamics_args=()
if [ "$dynamics_supervision" = "1" ]; then
  export FIELD2PLAN_DYNAMICS_MANIFEST_SHA256="${FIELD2PLAN_DYNAMICS_MANIFEST_SHA256:-$(sha256sum "$FIELD2PLAN_DYNAMICS_CACHE/manifest.json" | awk '{print $1}')}"
  dynamics_args=(--field2plan.dynamics.teacher.manifest_sha256 "$FIELD2PLAN_DYNAMICS_MANIFEST_SHA256")
fi

python - <<'PY'
import json
import os
from pathlib import Path
from starVLA.dataloader.field2plan_cache import (
    DraftCacheReader,
    DynamicsCacheReader,
    GeometryCacheReader,
    sha256_file,
)

datalist = Path(os.environ["FIELD2PLAN_DATALIST_PATH"])
tokens = json.loads(datalist.read_text(encoding="utf-8"))
if len(tokens) != 103288:
    raise RuntimeError(f"expected 103288 train tokens, found {len(tokens)}")
draft = DraftCacheReader(
    os.environ["FIELD2PLAN_DRAFT_CACHE"],
    "train",
    expected_manifest_sha256=os.environ["FIELD2PLAN_DRAFT_MANIFEST_SHA256"],
)
draft.validate_dataset_binding(tokens, str(datalist))
checkpoint = Path(os.environ["FIELD2PLAN_BASELINE_CHECKPOINT"])
if checkpoint.is_dir():
    checkpoint = checkpoint / "final_model" / "pytorch_model.pt"
if draft.manifest["checkpoint"]["sha256"] != sha256_file(checkpoint):
    raise RuntimeError("draft cache checkpoint checksum mismatch")
readers = [draft]
if os.environ["FIELD2PLAN_GEOMETRY_SUPERVISION"] == "1":
    geometry = GeometryCacheReader(
        os.environ["FIELD2PLAN_GEOMETRY_CACHE"],
        "train",
        expected_manifest_sha256=os.environ["FIELD2PLAN_GEOMETRY_MANIFEST_SHA256"],
    )
    geometry.validate_dataset_binding(tokens, str(datalist))
    if geometry.manifest["teacher"]["name"] != "depth_anything_3_metric_depth":
        raise RuntimeError("Phase-3 geometry cache must be DA3 metric depth")
    readers.append(geometry)
if os.environ["FIELD2PLAN_DYNAMICS_SUPERVISION"] == "1":
    dynamics = DynamicsCacheReader(
        os.environ["FIELD2PLAN_DYNAMICS_CACHE"],
        "train",
        expected_manifest_sha256=os.environ["FIELD2PLAN_DYNAMICS_MANIFEST_SHA256"],
    )
    dynamics.validate_dataset_binding(tokens, str(datalist))
    if dynamics.manifest["teacher"]["name"] != "vjepa2_1":
        raise RuntimeError("Phase-3 dynamics cache must be V-JEPA 2.1")
    readers.append(dynamics)
for index in sorted({0, len(tokens) // 2, len(tokens) - 1}):
    for reader in readers:
        reader.load(tokens[index])
print("[field2plan-phase3] cache preflight OK sampled_tokens=3")
PY

if [ "${FIELD2PLAN_PREFLIGHT_ONLY:-0}" = "1" ]; then
  exit 0
fi
if [ -f "$run_dir/.field2plan_complete" ]; then
  echo "[field2plan-phase3] run already complete: $run_dir"
  exit 0
fi
if [ -f "$run_dir/.field2plan_started" ] && [ "${FIELD2PLAN_ALLOW_RESTART_FROM_SCRATCH:-0}" != "1" ]; then
  echo "[field2plan-phase3] prior incomplete launch detected; refusing implicit restart from scratch" >&2
  exit 3
fi
mkdir -p "$run_dir"
touch "$run_dir/.field2plan_started"

dynamics_batch_shuffle=false
dynamics_temporal_shuffle=false
case "$dynamics_teacher_mode" in
  batch_shuffled) dynamics_batch_shuffle=true ;;
  temporal_shuffled) dynamics_temporal_shuffle=true ;;
esac
geometry_build_head="$geometry_supervision"
dynamics_build_head=true

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 16 \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_port "${MAIN_PROCESS_PORT:-29703}" \
  --mixed_precision bf16 \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/cfg_field2plan_phase3.yaml \
  --run_id "$run_id" \
  --seed "$run_seed" \
  --field2plan.geometry.teacher_type da3 \
  --field2plan.geometry.cache_dir "$FIELD2PLAN_GEOMETRY_CACHE" \
  "${geometry_args[@]}" \
  --field2plan.geometry.access_enabled "$geometry_access" \
  --field2plan.geometry.supervision.enabled "$geometry_supervision" \
  --field2plan.geometry.supervision.build_head "$geometry_build_head" \
  --field2plan.dynamics.teacher.cache_dir "$FIELD2PLAN_DYNAMICS_CACHE" \
  "${dynamics_args[@]}" \
  --field2plan.dynamics.access_enabled "$dynamics_access" \
  --field2plan.dynamics.supervision.enabled "$dynamics_supervision" \
  --field2plan.dynamics.supervision.build_head "$dynamics_build_head" \
  --field2plan.controls.dynamics_shuffle_teacher_across_batch "$dynamics_batch_shuffle" \
  --field2plan.controls.temporal_shuffle_teacher "$dynamics_temporal_shuffle" \
  --field2plan.controls.teacher_seed "$run_seed" \
  --datasets.vla_data.per_device_batch_size "$per_device_batch" \
  --trainer.gradient_accumulation_steps "$gradient_accumulation" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS:-100000}"

touch "$run_dir/.field2plan_complete"
