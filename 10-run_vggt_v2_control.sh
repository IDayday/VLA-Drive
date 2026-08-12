#!/usr/bin/env bash
# Launch one full-budget V2 causal control on a single 16-PPU PAI-DLC node.
#
# Modes:
#   no_teacher_access       equal-capacity student/planner without VGGT teacher
#   supervision_no_access  VGGT supervision with planner memory access disabled

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

mode="${VGGT_CONTROL_MODE:-${1:-}}"
case "$mode" in
  no_teacher_access)
    overlay="$DRIVEDREAMER_ROOT/starVLA/config/training/vggt_query_control_no_teacher_access.yaml"
    require_teacher_cache=0
    default_port=29711
    ;;
  supervision_no_access)
    overlay="$DRIVEDREAMER_ROOT/starVLA/config/training/vggt_query_control_supervision_no_access.yaml"
    require_teacher_cache=1
    default_port=29712
    ;;
  *)
    echo "Usage: VGGT_CONTROL_MODE={no_teacher_access|supervision_no_access} bash $0" >&2
    exit 2
    ;;
esac

if [[ ! -f "$overlay" ]]; then
  echo "Missing control overlay: $overlay" >&2
  exit 2
fi
if [[ ! -d "$VGGT_BASE_VLM" ]]; then
  echo "Missing V2 15-token VLM: $VGGT_BASE_VLM" >&2
  exit 2
fi

timestamp="$(date +'%Y%m%d_%H%M%S')"
run_id="${RUN_ID:-vggt-v2-control-${mode}-${PAI_JOB_ID:-$timestamp}}"
run_dir="$NAVSIM_EXP_ROOT/$run_id"
if [[ -e "$run_dir" ]]; then
  echo "Refusing to reuse existing run directory: $run_dir" >&2
  echo "Set a new RUN_ID; controls never overwrite or silently resume outputs." >&2
  exit 2
fi

export RUN_ID="$run_id"
export VGGT_EXPERIMENT_OVERLAY="$overlay"
export VGGT_REQUIRE_TEACHER_CACHE="$require_teacher_cache"
export VLM_ATTN_IMPLEMENTATION="${VGGT_VLM_ATTN_IMPLEMENTATION:-sdpa}"
export LOCAL_NUM_PROCESSES="${LOCAL_NUM_PROCESSES:-16}"
export NUM_PROCESSES="${NUM_PROCESSES:-16}"
export NUM_MACHINES="${NUM_MACHINES:-1}"
export MACHINE_RANK="${MACHINE_RANK:-0}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-32}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100000}"
export NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-5000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
export TRAINING_LOGGING_FREQUENCY="${TRAINING_LOGGING_FREQUENCY:-50}"
export BASE_LEARNING_RATE="${BASE_LEARNING_RATE:-1e-5}"
export ACTION_LEARNING_RATE="${ACTION_LEARNING_RATE:-1e-5}"
export VGGT_LEARNING_RATE="${VGGT_LEARNING_RATE:-3e-5}"
export OPTIMIZER_WEIGHT_DECAY="${OPTIMIZER_WEIGHT_DECAY:-1e-3}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-${MASTER_PORT:-$default_port}}"

launcher_log_dir="$NAVSIM_EXP_ROOT/launcher_logs"
mkdir -p "$launcher_log_dir"
launcher_log="$launcher_log_dir/${run_id}.control.log"
exec > >(tee -a "$launcher_log") 2>&1

echo "[vggt-control] mode=$mode"
echo "[vggt-control] run_id=$run_id"
echo "[vggt-control] overlay=$overlay"
echo "[vggt-control] teacher_cache_required=$require_teacher_cache"
echo "[vggt-control] topology=${NUM_PROCESSES}x batch${PER_DEVICE_BATCH_SIZE} x accum${GRADIENT_ACCUMULATION_STEPS}"
echo "[vggt-control] max_steps=$MAX_TRAIN_STEPS output=$run_dir"
echo "[vggt-control] log=$launcher_log"

cache_manifest="$NAVSIM_VGGT_CACHE_ROOT/vggt_query/manifest.json"
if [[ "$require_teacher_cache" == "1" ]]; then
  # Read-only synchronization with the full V2 job. This control must never
  # call the cache writer or attempt to repair a partial cache itself.
  wait_minutes="${VGGT_CONTROL_CACHE_WAIT_MINUTES:-360}"
  if ! [[ "$wait_minutes" =~ ^[0-9]+$ ]]; then
    echo "VGGT_CONTROL_CACHE_WAIT_MINUTES must be a non-negative integer" >&2
    exit 2
  fi
  manifest_is_complete() {
    [[ -f "$cache_manifest" ]] || return 1
    python - "$cache_manifest" <<'PY'
import json
import sys
from pathlib import Path

try:
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
expected = {
    "complete": True,
    "query_count": 195,
    "feature_dim": 1024,
    "teacher_layer_index": 11,
    "teacher_attention_branch": "global",
}
raise SystemExit(0 if all(manifest.get(key) == value for key, value in expected.items()) else 1)
PY
  }
  wait_seconds=$((wait_minutes * 60))
  waited=0
  echo "[vggt-control] cache policy=read-only; waiting for complete atomic manifest"
  while ! manifest_is_complete; do
    if (( waited >= wait_seconds )); then
      echo "Timed out waiting for complete atomic VGGT cache manifest: $cache_manifest" >&2
      echo "This control did not generate or modify the cache." >&2
      exit 2
    fi
    if (( waited % 300 == 0 )); then
      echo "[vggt-control] waiting for shared V2 cache: ${waited}s/${wait_seconds}s"
    fi
    sleep 60
    waited=$((waited + 60))
  done
  echo "[vggt-control] complete cache manifest observed after ${waited}s; validating read-only"
  python "$DRIVEDREAMER_ROOT/tools/precompute_vggt_query_cache.py" \
    --validate-only \
    --datalist-path "$NAVSIM_DATALIST_PATH" \
    --data-root "$DATA_ROOT" \
    --cache-root "$NAVSIM_VGGT_CACHE_ROOT"
  echo "[vggt-control] read-only cache validation PASS"
fi

if [[ "${VGGT_CONTROL_RUN_SMOKE:-1}" == "1" ]]; then
  smoke_run_id="${run_id}-smoke"
  smoke_dir="$NAVSIM_EXP_ROOT/$smoke_run_id"
  if [[ -e "$smoke_dir" ]]; then
    echo "Refusing to reuse existing control smoke directory: $smoke_dir" >&2
    exit 2
  fi
  echo "[vggt-control] starting 2-step forward/backward smoke: $smoke_run_id"
  env \
    RUN_ID="$smoke_run_id" \
    VGGT_DEBUG=1 \
    MAX_TRAIN_STEPS=2 \
    NUM_WARMUP_STEPS=2 \
    SAVE_INTERVAL=999999 \
    TRAINING_SKIP_FINAL_SAVE=1 \
    bash "$DRIVEDREAMER_ROOT/8-train_vggt_action.sh"
  echo "[vggt-control] smoke PASS; starting formal run"
elif [[ "${VGGT_CONTROL_RUN_SMOKE:-1}" != "0" ]]; then
  echo "VGGT_CONTROL_RUN_SMOKE must be 0 or 1" >&2
  exit 2
fi

bash "$DRIVEDREAMER_ROOT/8-train_vggt_action.sh"
