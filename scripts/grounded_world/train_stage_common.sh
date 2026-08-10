#!/usr/bin/env bash
# Shared fail-fast one-node/16-accelerator GroundedWorld training launcher.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
source "$project_root/env.sh"

stage="${GROUNDEDWORLD_STAGE:?GROUNDEDWORLD_STAGE is required}"
case "$stage" in
  stage1) config_name=cfg_groundedworld_stage1.yaml ;;
  stage2) config_name=cfg_groundedworld_stage2.yaml ;;
  stage3) config_name=cfg_groundedworld_stage3.yaml ;;
  *) echo "[groundedworld] invalid stage=$stage" >&2; exit 2 ;;
esac

export GROUNDEDWORLD_DATALIST_PATH="${GROUNDEDWORLD_DATALIST_PATH:-$project_root/train_meta.json}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${DRIVEDREAMER_SHARED_ROOT:?}/navsim_exp}"
export VLM_ATTN_IMPLEMENTATION="${VLM_ATTN_IMPLEMENTATION:-flash_attention_2}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TRAINING_SKIP_FINAL_SAVE=0
unset NAVSIM_FEATURE_CACHE_ROOT NAVSIM_CACHE_COMPONENTS

external_prior="${GROUNDEDWORLD_EXTERNAL_PRIOR:-vggt_driving_jepa}"
teacher_mode="${GROUNDEDWORLD_TEACHER_MODE:-real}"
future_enabled="${GROUNDEDWORLD_FUTURE_ENABLED:-0}"
world_access="${GROUNDEDWORLD_WORLD_ACCESS:-0}"
refiner_enabled="${GROUNDEDWORLD_REFINER_ENABLED:-1}"
consequence_enabled="${GROUNDEDWORLD_CONSEQUENCE_ENABLED:-0}"
stage3_phase="${GROUNDEDWORLD_STAGE3_PHASE:-A}"
stage3_direct_init="${GROUNDEDWORLD_STAGE3_DIRECT_INIT:-0}"
run_seed="${GROUNDEDWORLD_RUN_SEED:-42}"
for value_name in future_enabled world_access refiner_enabled consequence_enabled; do
  value="${!value_name}"
  if [ "$value" != "0" ] && [ "$value" != "1" ]; then
    echo "[groundedworld] $value_name must be 0 or 1" >&2
    exit 2
  fi
done
if ! [[ "$run_seed" =~ ^[0-9]+$ ]]; then
  echo "[groundedworld] run seed must be non-negative" >&2
  exit 2
fi
if [ "$stage3_direct_init" != "0" ] && [ "$stage3_direct_init" != "1" ]; then
  echo "[groundedworld] GROUNDEDWORLD_STAGE3_DIRECT_INIT must be 0 or 1" >&2
  exit 2
fi

actual_devices="$(python -c 'import torch; print(torch.cuda.device_count())')"
local_processes="${LOCAL_NUM_PROCESSES:-16}"
num_machines="${NUM_MACHINES:-1}"
machine_rank="${MACHINE_RANK:-0}"
per_device_batch="${PER_DEVICE_BATCH_SIZE:-2}"
gradient_accumulation="${GRADIENT_ACCUMULATION_STEPS:-1}"
effective_batch=$((local_processes * num_machines * per_device_batch * gradient_accumulation))
if (( local_processes != 16 || num_machines != 1 || machine_rank != 0 )); then
  echo "[groundedworld] formal topology is one node x 16 processes" >&2
  exit 2
fi
if (( effective_batch != 32 )); then
  echo "[groundedworld] effective batch must be 32, got $effective_batch" >&2
  exit 2
fi
if [ "${GROUNDEDWORLD_TOPOLOGY_ONLY:-0}" != "1" ] && (( actual_devices < 16 )); then
  echo "[groundedworld] need 16 visible accelerators, found $actual_devices" >&2
  exit 2
fi

export NAVSIM_NUM_WORKERS="${NAVSIM_NUM_WORKERS:-6}"
export NAVSIM_PREFETCH_FACTOR="${NAVSIM_PREFETCH_FACTOR:-4}"
export NAVSIM_PIN_MEMORY="${NAVSIM_PIN_MEMORY:-1}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

run_id="${RUN_ID:-groundedworld-${stage}-seed${run_seed}-$(date +'%Y%m%d_%H%M%S')}"
run_dir="$NAVSIM_EXP_ROOT/$run_id"
export TRITON_CACHE_DIR="${TRITON_LOCAL_CACHE_ROOT:-/tmp/groundedworld-triton}/$run_id/node0"
mkdir -p "$TRITON_CACHE_DIR" "$NAVSIM_EXP_ROOT/launcher_logs"

echo "[groundedworld] stage=$stage run_id=$run_id"
echo "[groundedworld] external_prior=$external_prior teacher_mode=$teacher_mode"
echo "[groundedworld] future=$future_enabled access=$world_access consequence=$consequence_enabled"
echo "[groundedworld] topology=1x16 effective_batch=$effective_batch"
if [ "${GROUNDEDWORLD_TOPOLOGY_ONLY:-0}" = "1" ]; then
  exit 0
fi

required=(
  "$BASE_VLM/config.json"
  "$GROUNDEDWORLD_DATALIST_PATH"
  "$DATA_ROOT/meta/train"
)
case "$external_prior" in
  *vggt*) required+=("${GROUNDEDWORLD_GEOMETRY_CACHE:?}/manifest.json") ;;
esac
case "$external_prior" in
  *jepa*|*random_frozen*)
    if [ "$teacher_mode" != "none" ]; then
      required+=("${GROUNDEDWORLD_DYNAMICS_PRIOR_CACHE:?}/manifest.json")
    fi
    ;;
esac
if [ "$future_enabled" = "1" ]; then
  required+=("${GROUNDEDWORLD_FUTURE_TARGET_CACHE:?}/manifest.json")
fi
if [ "$stage" = "stage2" ]; then
  required+=("${GROUNDEDWORLD_STAGE1_CHECKPOINT:?}")
  export GROUNDEDWORLD_WORLD_CHECKPOINT="$GROUNDEDWORLD_STAGE1_CHECKPOINT"
fi
if [ "$stage" = "stage3" ] && [ "$external_prior" = "none" ] \
  && [ "$future_enabled" = "0" ] && [ "$world_access" = "0" ] \
  && [ "$refiner_enabled" = "0" ] && [ "$consequence_enabled" = "0" ]; then
  echo "[groundedworld] fixed B0 baseline must be evaluated from its supplied checkpoint, not retrained" >&2
  exit 2
fi
if [ "$stage" = "stage3" ]; then
  if [ "$stage3_phase" = "B" ] && [ "$stage3_direct_init" != "1" ]; then
    export GROUNDEDWORLD_WORLD_CHECKPOINT="${GROUNDEDWORLD_STAGE3A_CHECKPOINT:?}"
  elif [ "$future_enabled" = "1" ]; then
    export GROUNDEDWORLD_WORLD_CHECKPOINT="${GROUNDEDWORLD_STAGE2_CHECKPOINT:?}"
  else
    export GROUNDEDWORLD_WORLD_CHECKPOINT="${GROUNDEDWORLD_STAGE1_CHECKPOINT:?}"
  fi
  required+=("$GROUNDEDWORLD_WORLD_CHECKPOINT")
  if [ "$stage3_phase" = "A" ] || [ "$stage3_direct_init" = "1" ]; then
    required+=("${GROUNDEDWORLD_BASELINE_CHECKPOINT:?}")
  fi
fi
if [ "$consequence_enabled" = "1" ]; then
  required+=("${GROUNDEDWORLD_CONSEQUENCE_CACHE:?}/manifest.json")
fi
for path in "${required[@]}"; do
  if [ ! -e "$path" ]; then
    echo "[groundedworld] missing required path: $path" >&2
    exit 2
  fi
done

if [[ "$external_prior" == *vggt* ]]; then
  export GROUNDEDWORLD_GEOMETRY_MANIFEST_SHA256="${GROUNDEDWORLD_GEOMETRY_MANIFEST_SHA256:-$(sha256sum "$GROUNDEDWORLD_GEOMETRY_CACHE/manifest.json" | awk '{print $1}')}"
fi
if [ -n "${GROUNDEDWORLD_DYNAMICS_PRIOR_CACHE:-}" ]; then
  export GROUNDEDWORLD_DYNAMICS_PRIOR_MANIFEST_SHA256="${GROUNDEDWORLD_DYNAMICS_PRIOR_MANIFEST_SHA256:-$(sha256sum "$GROUNDEDWORLD_DYNAMICS_PRIOR_CACHE/manifest.json" | awk '{print $1}')}"
fi
if [ "$future_enabled" = "1" ]; then
  export GROUNDEDWORLD_FUTURE_TARGET_MANIFEST_SHA256="${GROUNDEDWORLD_FUTURE_TARGET_MANIFEST_SHA256:-$(sha256sum "$GROUNDEDWORLD_FUTURE_TARGET_CACHE/manifest.json" | awk '{print $1}')}"
fi
if [ "$consequence_enabled" = "1" ]; then
  export GROUNDEDWORLD_CONSEQUENCE_MANIFEST_SHA256="${GROUNDEDWORLD_CONSEQUENCE_MANIFEST_SHA256:-$(sha256sum "$GROUNDEDWORLD_CONSEQUENCE_CACHE/manifest.json" | awk '{print $1}')}"
fi

if [ "${GROUNDEDWORLD_PREFLIGHT_ONLY:-0}" = "1" ]; then
  echo "[groundedworld] preflight OK"
  exit 0
fi
if [ -e "$run_dir/.groundedworld_complete" ]; then
  echo "[groundedworld] run already complete: $run_dir"
  exit 0
fi
if [ -e "$run_dir/.groundedworld_started" ] && [ "${GROUNDEDWORLD_ALLOW_RESTART_FROM_SCRATCH:-0}" != "1" ]; then
  echo "[groundedworld] refusing implicit restart of incomplete run: $run_dir" >&2
  exit 3
fi
mkdir -p "$run_dir"
touch "$run_dir/.groundedworld_started"

extra_args=()
if [ "$stage" = "stage3" ] && [ "$external_prior" = "none" ] && [ "$future_enabled" = "0" ] && [ "$world_access" = "0" ]; then
  extra_args+=(--grounded_world.enabled false)
fi

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 16 \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_port "${MAIN_PROCESS_PORT:-29711}" \
  --mixed_precision bf16 \
  starVLA/training/train_starvla.py \
  --config_yaml "starVLA/config/training/$config_name" \
  --run_id "$run_id" \
  --seed "$run_seed" \
  --grounded_world.prior.source "$external_prior" \
  --grounded_world.prior.teacher_mode "$teacher_mode" \
  --grounded_world.future.enabled "$future_enabled" \
  --grounded_world.planner.world_access "$world_access" \
  --grounded_world.planner.refiner_enabled "$refiner_enabled" \
  --grounded_world.consequence.enabled "$consequence_enabled" \
  --datasets.vla_data.per_device_batch_size "$per_device_batch" \
  --trainer.gradient_accumulation_steps "$gradient_accumulation" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS:-100000}" \
  "${extra_args[@]}"

touch "$run_dir/.groundedworld_complete"
