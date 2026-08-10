#!/usr/bin/env bash
# Formal Phase-2 one-node/16-PPU training launcher with 2x2/control switches.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
source "$project_root/env.sh"

export FIELD2PLAN_BASELINE_CHECKPOINT="${FIELD2PLAN_BASELINE_CHECKPOINT:-$project_root/navsim_exp/navsim-v1-formal-16gpu-bz2-ga1-flashattn-20260801_193514/final_model/pytorch_model.pt}"
export FIELD2PLAN_DATALIST_PATH="${FIELD2PLAN_DATALIST_PATH:-$project_root/train_meta.json}"
export FIELD2PLAN_DRAFT_CACHE="${FIELD2PLAN_DRAFT_CACHE:-$DRIVEDREAMER_SHARED_ROOT/field2plan_cache/baseline_drafts_193514_seed20260808_steps10}"
export FIELD2PLAN_GEOMETRY_TEACHER_TYPE="${FIELD2PLAN_GEOMETRY_TEACHER_TYPE:-da3}"
case "$FIELD2PLAN_GEOMETRY_TEACHER_TYPE" in
  da3)
    default_geometry_cache="$DRIVEDREAMER_SHARED_ROOT/field2plan_cache/geometry_da3_metric_v1"
    ;;
  vggt)
    default_geometry_cache="$DRIVEDREAMER_SHARED_ROOT/field2plan_cache/geometry_vggt_1b_da3_anchor_v1"
    ;;
  *)
    echo "[field2plan-geometry] FIELD2PLAN_GEOMETRY_TEACHER_TYPE must be da3 or vggt" >&2
    exit 2
    ;;
esac
export FIELD2PLAN_GEOMETRY_CACHE="${FIELD2PLAN_GEOMETRY_CACHE:-$default_geometry_cache}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$DRIVEDREAMER_SHARED_ROOT/navsim_exp}"
export VLM_ATTN_IMPLEMENTATION="${VLM_ATTN_IMPLEMENTATION:-flash_attention_2}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TRAINING_SKIP_FINAL_SAVE=0
unset NAVSIM_FEATURE_CACHE_ROOT NAVSIM_CACHE_COMPONENTS

supervision="${FIELD2PLAN_GEOMETRY_SUPERVISION:-1}"
disable_access="${FIELD2PLAN_DISABLE_ACCESS:-0}"
teacher_mode="${FIELD2PLAN_TEACHER_MODE:-real}"
run_seed="${FIELD2PLAN_RUN_SEED:-42}"
export FIELD2PLAN_GEOMETRY_SUPERVISION="$supervision"
if ! [[ "$run_seed" =~ ^[0-9]+$ ]]; then
  echo "[field2plan-geometry] FIELD2PLAN_RUN_SEED must be a non-negative integer" >&2
  exit 2
fi
for flag_name in supervision disable_access; do
  flag_value="${!flag_name}"
  if [ "$flag_value" != "0" ] && [ "$flag_value" != "1" ]; then
    echo "[field2plan-geometry] $flag_name must be 0 or 1" >&2
    exit 2
  fi
done
case "$teacher_mode" in
  real|random|shuffled)
    if [ "$supervision" != "1" ]; then
      echo "[field2plan-geometry] teacher_mode=$teacher_mode requires supervision=1" >&2
      exit 2
    fi
    ;;
  equal_capacity)
    if [ "$supervision" != "0" ]; then
      echo "[field2plan-geometry] equal_capacity requires supervision=0" >&2
      exit 2
    fi
    ;;
  gt_mlp)
    if [ "$supervision" != "0" ]; then
      echo "[field2plan-geometry] gt_mlp requires supervision=0" >&2
      exit 2
    fi
    ;;
  *)
    echo "[field2plan-geometry] unsupported FIELD2PLAN_TEACHER_MODE=$teacher_mode" >&2
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
  echo "[field2plan-geometry] formal topology must be one node x 16 processes; nodes=$num_machines rank=$machine_rank local=$local_processes global=$global_processes" >&2
  exit 2
fi
if (( effective_batch != target_effective_batch )); then
  echo "[field2plan-geometry] effective batch mismatch: 16 x $per_device_batch x $gradient_accumulation = $effective_batch, expected 32" >&2
  exit 2
fi
if [ "${FIELD2PLAN_TOPOLOGY_ONLY:-0}" != "1" ] && (( actual_devices < local_processes )); then
  echo "[field2plan-geometry] need 16 visible accelerators, found $actual_devices" >&2
  exit 2
fi

host_cpus="$(getconf _NPROCESSORS_ONLN)"
export NAVSIM_NUM_WORKERS="${NAVSIM_NUM_WORKERS:-6}"
export NAVSIM_PREFETCH_FACTOR="${NAVSIM_PREFETCH_FACTOR:-4}"
export NAVSIM_PIN_MEMORY="${NAVSIM_PIN_MEMORY:-1}"
if [ "${FIELD2PLAN_TOPOLOGY_ONLY:-0}" != "1" ] && (( NAVSIM_NUM_WORKERS * local_processes + local_processes > host_cpus )); then
  echo "[field2plan-geometry] CPU oversubscription: workers=$NAVSIM_NUM_WORKERS ranks=$local_processes cpus=$host_cpus" >&2
  exit 2
fi
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

job_tag="${PAI_JOB_ID:-$(date +'%Y%m%d_%H%M%S')}"
access_tag="access"
if [ "$disable_access" = "1" ]; then access_tag="noaccess"; fi
supervision_tag="nosup"
if [ "$supervision" = "1" ]; then supervision_tag="sup"; fi
run_id="${RUN_ID:-field2plan-geometry-${FIELD2PLAN_GEOMETRY_TEACHER_TYPE}-${supervision_tag}-${access_tag}-${teacher_mode}-${job_tag}}"
run_dir="$NAVSIM_EXP_ROOT/$run_id"
launcher_log_dir="$NAVSIM_EXP_ROOT/launcher_logs"
mkdir -p "$launcher_log_dir"
launcher_log="$launcher_log_dir/$run_id.log"
export TRITON_CACHE_DIR="${TRITON_LOCAL_CACHE_ROOT:-/tmp/field2plan-triton}/$run_id/node0"
mkdir -p "$TRITON_CACHE_DIR"
exec > >(tee -a "$launcher_log") 2>&1

echo "[field2plan-geometry] project_root=$project_root run_id=$run_id"
echo "[field2plan-geometry] topology=nodes:1 rank:0 local_processes:16 global_processes:16 visible:$actual_devices"
echo "[field2plan-geometry] per_device_batch=$per_device_batch gradient_accumulation=$gradient_accumulation effective_batch=$effective_batch target=32"
echo "[field2plan-geometry] supervision=$supervision access=$((1-disable_access)) teacher_type=$FIELD2PLAN_GEOMETRY_TEACHER_TYPE teacher_mode=$teacher_mode"
echo "[field2plan-geometry] run_seed=$run_seed"
echo "[field2plan-geometry] draft_cache=$FIELD2PLAN_DRAFT_CACHE geometry_cache=$FIELD2PLAN_GEOMETRY_CACHE"
echo "[field2plan-geometry] dataloader=cpus:$host_cpus workers_per_rank:$NAVSIM_NUM_WORKERS prefetch:$NAVSIM_PREFETCH_FACTOR"

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
if [ "$supervision" = "1" ]; then
  required+=("$FIELD2PLAN_GEOMETRY_CACHE/manifest.json")
fi
for path in "${required[@]}"; do
  if [ ! -e "$path" ]; then
    echo "[field2plan-geometry] missing required path: $path" >&2
    exit 2
  fi
done

export FIELD2PLAN_DRAFT_MANIFEST_SHA256="${FIELD2PLAN_DRAFT_MANIFEST_SHA256:-$(sha256sum "$FIELD2PLAN_DRAFT_CACHE/manifest.json" | awk '{print $1}')}"
geometry_manifest_args=()
if [ "$supervision" = "1" ]; then
  export FIELD2PLAN_GEOMETRY_MANIFEST_SHA256="${FIELD2PLAN_GEOMETRY_MANIFEST_SHA256:-$(sha256sum "$FIELD2PLAN_GEOMETRY_CACHE/manifest.json" | awk '{print $1}')}"
  geometry_manifest_args=(
    --field2plan.geometry.manifest_sha256
    "$FIELD2PLAN_GEOMETRY_MANIFEST_SHA256"
  )
fi

python - <<'PY'
import json
import os
from pathlib import Path
from starVLA.dataloader.field2plan_cache import (
    DraftCacheReader,
    GeometryCacheReader,
    sha256_file,
)

datalist = Path(os.environ['FIELD2PLAN_DATALIST_PATH'])
tokens = json.loads(datalist.read_text(encoding='utf-8'))
if len(tokens) != 103288:
    raise RuntimeError(f'expected 103288 train tokens, found {len(tokens)}')
draft = DraftCacheReader(
    os.environ['FIELD2PLAN_DRAFT_CACHE'],
    'train',
    expected_manifest_sha256=os.environ['FIELD2PLAN_DRAFT_MANIFEST_SHA256'],
)
if draft.manifest['splits']['train']['entry_count'] != len(tokens):
    raise RuntimeError('draft cache entry_count mismatch')
if draft.manifest['splits']['train']['datalist_sha256'] != sha256_file(datalist):
    raise RuntimeError('draft cache datalist checksum mismatch')
checkpoint = Path(os.environ['FIELD2PLAN_BASELINE_CHECKPOINT'])
if checkpoint.is_dir():
    checkpoint = checkpoint / 'final_model' / 'pytorch_model.pt'
if draft.manifest['checkpoint']['sha256'] != sha256_file(checkpoint):
    raise RuntimeError('draft cache checkpoint checksum mismatch')
readers = [draft]
if os.environ['FIELD2PLAN_GEOMETRY_SUPERVISION'] == '1':
    geometry = GeometryCacheReader(
        os.environ['FIELD2PLAN_GEOMETRY_CACHE'],
        'train',
        expected_manifest_sha256=os.environ[
            'FIELD2PLAN_GEOMETRY_MANIFEST_SHA256'
        ],
    )
    if geometry.manifest['splits']['train']['entry_count'] != len(tokens):
        raise RuntimeError('geometry cache entry_count mismatch')
    if geometry.manifest['splits']['train']['datalist_sha256'] != sha256_file(datalist):
        raise RuntimeError('geometry cache datalist checksum mismatch')
    expected_teacher = {
        'da3': 'depth_anything_3_metric_depth',
        'vggt': 'vggt',
    }[os.environ['FIELD2PLAN_GEOMETRY_TEACHER_TYPE']]
    if geometry.manifest['teacher']['name'] != expected_teacher:
        raise RuntimeError(
            f'geometry teacher mismatch: expected={expected_teacher} '
            f"actual={geometry.manifest['teacher']['name']}"
        )
    readers.append(geometry)
for index in sorted({0, len(tokens) // 2, len(tokens) - 1}):
    for reader in readers:
        reader.load(tokens[index])
print('[field2plan-geometry] cache preflight OK sampled_tokens=3')
PY

if [ "${FIELD2PLAN_PREFLIGHT_ONLY:-0}" = "1" ]; then
  exit 0
fi

if [ -f "$run_dir/.field2plan_complete" ]; then
  echo "[field2plan-geometry] run already complete; refusing duplicate training: $run_dir"
  exit 0
fi
if [ -f "$run_dir/.field2plan_started" ] && [ "${FIELD2PLAN_ALLOW_RESTART_FROM_SCRATCH:-0}" != "1" ]; then
  echo "[field2plan-geometry] prior incomplete launch detected. Refusing an implicit restart from scratch; inspect checkpoints or set an explicit resume policy." >&2
  exit 3
fi
mkdir -p "$run_dir"
touch "$run_dir/.field2plan_started"

random_teacher=false
shuffle_teacher=false
equal_capacity=false
gt_mlp=false
case "$teacher_mode" in
  random) random_teacher=true ;;
  shuffled) shuffle_teacher=true ;;
  equal_capacity) equal_capacity=true ;;
  gt_mlp) gt_mlp=true ;;
esac
build_head=true
if [ "$teacher_mode" = "gt_mlp" ]; then build_head=false; fi

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 16 \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_port "${MAIN_PROCESS_PORT:-29693}" \
  --mixed_precision bf16 \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/cfg_field2plan_mvp.yaml \
  --run_id "$run_id" \
  --seed "$run_seed" \
  --field2plan.geometry.teacher_type "$FIELD2PLAN_GEOMETRY_TEACHER_TYPE" \
  --field2plan.geometry.cache_dir "$FIELD2PLAN_GEOMETRY_CACHE" \
  "${geometry_manifest_args[@]}" \
  --field2plan.geometry.supervision.enabled "$supervision" \
  --field2plan.geometry.supervision.build_head "$build_head" \
  --field2plan.controls.disable_access "$disable_access" \
  --field2plan.controls.random_teacher "$random_teacher" \
  --field2plan.controls.shuffle_teacher_across_batch "$shuffle_teacher" \
  --field2plan.controls.equal_capacity_no_teacher "$equal_capacity" \
  --field2plan.controls.gt_mlp_teacher "$gt_mlp" \
  --field2plan.controls.teacher_seed "$run_seed" \
  --trainer.loss_weights.geometry_depth 0.0002 \
  --trainer.loss_weights.geometry_occupancy 0.002 \
  --trainer.loss_weights.geometry_free_space 0.002 \
  --trainer.loss_weights.geometry_relative 0.002 \
  --datasets.vla_data.per_device_batch_size "$per_device_batch" \
  --trainer.gradient_accumulation_steps "$gradient_accumulation" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS:-100000}"

touch "$run_dir/.field2plan_complete"
