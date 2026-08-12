#!/usr/bin/env bash
# One-command, non-interactive VGGT-query preparation and training on PAI-DLC PPU.
#
# Default formal topology and batch:
#   1 DLC node x 16 PPU x batch 2 x accumulation 1 = effective batch 32.
# Completed token/cache stages are validated and skipped on a rerun.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

phase="bootstrap"
on_error() {
  local status=$?
  echo "[vggt-pipeline] FAILED phase=${phase} exit_code=${status}" >&2
  exit "$status"
}
trap on_error ERR

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[vggt-pipeline] Missing required file: $1" >&2
    exit 2
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "[vggt-pipeline] Missing required directory: $1" >&2
    exit 2
  fi
}

print_command() {
  printf '[vggt-pipeline] DRY-RUN:'
  printf ' %q' "$@"
  printf '\n'
}

run_command() {
  if [[ "${VGGT_PIPELINE_DRY_RUN:-0}" == "1" ]]; then
    print_command "$@"
  else
    "$@"
  fi
}

phase="topology"
dry_run="${VGGT_PIPELINE_DRY_RUN:-0}"
num_machines="${NUM_MACHINES:-${WORLD_SIZE:-1}}"
machine_rank="${MACHINE_RANK:-${RANK:-0}}"
if [[ "$dry_run" == "1" ]]; then
  detected_local_ppus="${NPROC_PER_NODE:-${LOCAL_NUM_PROCESSES:-16}}"
else
  detected_local_ppus="$(python -c 'import torch; print(torch.cuda.device_count())')"
fi
local_ppus="${LOCAL_NUM_PROCESSES:-${NPROC_PER_NODE:-$detected_local_ppus}}"
expected_ppus="${VGGT_EXPECTED_PPU_COUNT:-16}"

for pair in \
  "NUM_MACHINES:$num_machines" \
  "MACHINE_RANK:$machine_rank" \
  "LOCAL_NUM_PROCESSES:$local_ppus" \
  "VGGT_EXPECTED_PPU_COUNT:$expected_ppus"; do
  variable="${pair%%:*}"
  value="${pair#*:}"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "[vggt-pipeline] $variable must be an integer, got: $value" >&2
    exit 2
  fi
done
global_processes="${NUM_PROCESSES:-$((num_machines * local_ppus))}"
if ! [[ "$global_processes" =~ ^[0-9]+$ ]]; then
  echo "[vggt-pipeline] NUM_PROCESSES must be an integer, got: $global_processes" >&2
  exit 2
fi
if (( num_machines != 1 || machine_rank != 0 )); then
  echo "[vggt-pipeline] This end-to-end cache+train entrypoint requires one DLC node." >&2
  echo "[vggt-pipeline] Requested topology: nodes=$num_machines node_rank=$machine_rank." >&2
  echo "[vggt-pipeline] Use one ml.gp7vf.* 16-PPU node; the cache writer is single-node sharded." >&2
  exit 2
fi
if (( local_ppus < 1 || global_processes != local_ppus )); then
  echo "[vggt-pipeline] Invalid single-node process topology: local=$local_ppus global=$global_processes" >&2
  exit 2
fi
if (( local_ppus != expected_ppus )); then
  echo "[vggt-pipeline] Expected $expected_ppus visible PPUs, DLC reports $local_ppus." >&2
  exit 2
fi

target_effective_batch="${TARGET_EFFECTIVE_BATCH_SIZE:-32}"
per_device_batch="${PER_DEVICE_BATCH_SIZE:-2}"
gradient_accumulation="${GRADIENT_ACCUMULATION_STEPS:-1}"
for pair in \
  "TARGET_EFFECTIVE_BATCH_SIZE:$target_effective_batch" \
  "PER_DEVICE_BATCH_SIZE:$per_device_batch" \
  "GRADIENT_ACCUMULATION_STEPS:$gradient_accumulation"; do
  variable="${pair%%:*}"
  value="${pair#*:}"
  if ! is_positive_integer "$value"; then
    echo "[vggt-pipeline] $variable must be a positive integer, got: $value" >&2
    exit 2
  fi
done
effective_batch=$((global_processes * per_device_batch * gradient_accumulation))
if (( effective_batch != target_effective_batch )); then
  echo "[vggt-pipeline] Refusing effective batch $effective_batch; expected $target_effective_batch." >&2
  echo "[vggt-pipeline] Formula: $global_processes processes x $per_device_batch batch x $gradient_accumulation accumulation." >&2
  exit 2
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  visible_devices=""
  for ((device_index = 0; device_index < local_ppus; device_index++)); do
    [[ -n "$visible_devices" ]] && visible_devices+=","
    visible_devices+="$device_index"
  done
  export CUDA_VISIBLE_DEVICES="$visible_devices"
fi

# Native SDPA is the conservative default for PPU. A PPU-specific flash-attn
# wheel may be selected explicitly without changing shared configuration.
export VLM_ATTN_IMPLEMENTATION="${VGGT_VLM_ATTN_IMPLEMENTATION:-sdpa}"
export NUM_MACHINES="$num_machines"
export MACHINE_RANK="$machine_rank"
export LOCAL_NUM_PROCESSES="$local_ppus"
export NUM_PROCESSES="$global_processes"
export PER_DEVICE_BATCH_SIZE="$per_device_batch"
export GRADIENT_ACCUMULATION_STEPS="$gradient_accumulation"
export TARGET_EFFECTIVE_BATCH_SIZE="$target_effective_batch"
export VGGT_CACHE_NUM_PROCESSES="${VGGT_CACHE_NUM_PROCESSES:-$local_ppus}"
export VGGT_CACHE_BATCH_SIZE="${VGGT_CACHE_BATCH_SIZE:-1}"
export VGGT_CACHE_MAP_SIZE_GB="${VGGT_CACHE_MAP_SIZE_GB:-16}"
export MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-${MASTER_ADDR:-127.0.0.1}}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-${MASTER_PORT:-29689}}"
if [[ "${VGGT_DEBUG:-0}" == "1" ]]; then
  export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20}"
  export NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-2}"
  export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
  export TRAINING_LOGGING_FREQUENCY="${TRAINING_LOGGING_FREQUENCY:-1}"
else
  export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100000}"
  export NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-5000}"
  export SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
  export TRAINING_LOGGING_FREQUENCY="${TRAINING_LOGGING_FREQUENCY:-50}"
fi
export BASE_LEARNING_RATE="${BASE_LEARNING_RATE:-1e-5}"
export ACTION_LEARNING_RATE="${ACTION_LEARNING_RATE:-1e-5}"
export VGGT_LEARNING_RATE="${VGGT_LEARNING_RATE:-3e-5}"
export OPTIMIZER_WEIGHT_DECAY="${OPTIMIZER_WEIGHT_DECAY:-1e-3}"
export RUN_ID="${RUN_ID:-vggt-query-v2-layer11-global-${PAI_JOB_ID:-$(date +'%Y%m%d_%H%M%S')}}"
export TRITON_CACHE_DIR="${TRITON_LOCAL_CACHE_ROOT:-/tmp/drivedreamer-triton}/${RUN_ID}/node${machine_rank}"

launcher_log_dir="$NAVSIM_EXP_ROOT/launcher_logs"
mkdir -p "$launcher_log_dir"
launcher_log="$launcher_log_dir/${RUN_ID}.pipeline.log"
if [[ "$dry_run" != "1" ]]; then
  mkdir -p "$TRITON_CACHE_DIR"
  exec > >(tee -a "$launcher_log") 2>&1
fi

echo "[vggt-pipeline] project_root=$DRIVEDREAMER_ROOT"
echo "[vggt-pipeline] run_id=$RUN_ID"
echo "[vggt-pipeline] topology=nodes:$num_machines node_rank:$machine_rank local_ppus:$local_ppus global_processes:$global_processes"
echo "[vggt-pipeline] effective_batch=$effective_batch (per_device=$per_device_batch accumulation=$gradient_accumulation)"
echo "[vggt-pipeline] max_train_steps=$MAX_TRAIN_STEPS attention=$VLM_ATTN_IMPLEMENTATION"
echo "[vggt-pipeline] optimization=base_lr:$BASE_LEARNING_RATE action_lr:$ACTION_LEARNING_RATE vggt_lr:$VGGT_LEARNING_RATE weight_decay:$OPTIMIZER_WEIGHT_DECAY warmup:$NUM_WARMUP_STEPS save_every:$SAVE_INTERVAL"
echo "[vggt-pipeline] vggt_repo=$VGGT_REPO"
echo "[vggt-pipeline] vggt_checkpoint=$VGGT_CHECKPOINT"
echo "[vggt-pipeline] source_vlm=$VGGT_SOURCE_VLM"
echo "[vggt-pipeline] token_vlm=$VGGT_BASE_VLM"
echo "[vggt-pipeline] data_root=$DATA_ROOT"
echo "[vggt-pipeline] datalist=$NAVSIM_DATALIST_PATH"
echo "[vggt-pipeline] sensor_root=$NAVSIM_TRAINVAL_SENSOR_ROOT"
echo "[vggt-pipeline] cache_root=$NAVSIM_VGGT_CACHE_ROOT"
echo "[vggt-pipeline] experiment_root=$NAVSIM_EXP_ROOT"
echo "[vggt-pipeline] launcher_log=$launcher_log"

tokens_file="$DRIVEDREAMER_ROOT/starVLA/model/modules/vlm/tools/add_qwen_special_tokens/vggt_global_query_tokens_15.txt"
cache_manifest="$NAVSIM_VGGT_CACHE_ROOT/vggt_query/manifest.json"
cache_validate=(
  python "$DRIVEDREAMER_ROOT/tools/precompute_vggt_query_cache.py"
  --validate-only
  --datalist-path "$NAVSIM_DATALIST_PATH"
  --data-root "$DATA_ROOT"
  --cache-root "$NAVSIM_VGGT_CACHE_ROOT"
)

if [[ "$dry_run" == "1" ]]; then
  print_command "${PPU_SMI_BIN:-ppu-smi}" -L
  print_command torchrun --standalone --nnodes=1 --nproc-per-node="$local_ppus" "$DRIVEDREAMER_ROOT/tools/check_ppu_runtime.py"
  print_command env CUDA_VISIBLE_DEVICES="${VGGT_TOKEN_VISIBLE_DEVICE:-0}" VGGT_TOKEN_DEVICE="${VGGT_TOKEN_DEVICE:-cuda}" VLM_ATTN_IMPLEMENTATION="$VLM_ATTN_IMPLEMENTATION" bash "$DRIVEDREAMER_ROOT/7-add_vggt_tokens.sh"
  print_command env VGGT_CACHE_NUM_PROCESSES="$local_ppus" bash "$DRIVEDREAMER_ROOT/tools/cache_vggt_queries.sh"
  print_command "${cache_validate[@]}"
  print_command env VGGT_DEBUG=1 VGGT_INTERVENTION_INTERVAL=1 MAX_TRAIN_STEPS=2 SAVE_INTERVAL=999999 TRAINING_SKIP_FINAL_SAVE=1 RUN_ID="${RUN_ID}-smoke" bash "$DRIVEDREAMER_ROOT/8-train_vggt_action.sh"
  print_command bash "$DRIVEDREAMER_ROOT/8-train_vggt_action.sh"
  echo "[vggt-pipeline] DRY-RUN complete; no model, cache, or training output was generated."
  exit 0
fi

phase="ppu-runtime-preflight"
if [[ -n "${PPU_SMI_BIN:-}" ]]; then
  ppu_smi="$PPU_SMI_BIN"
elif command -v ppu-smi >/dev/null 2>&1; then
  ppu_smi="$(command -v ppu-smi)"
else
  ppu_smi="/usr/local/PPU_SDK/ppu-smi/bin/ppu-smi"
fi
if [[ ! -x "$ppu_smi" ]]; then
  echo "[vggt-pipeline] ppu-smi was not found; use an official PAI-PPU training image." >&2
  exit 2
fi
ppu_release="${PPU_SDK_RELEASE_FILE:-/usr/local/PPU_SDK/release.yaml}"
require_file "$ppu_release"
echo "[vggt-pipeline] PPU SDK release:"
sed -n '1,80p' "$ppu_release"
echo "[vggt-pipeline] Visible PPU devices:"
"$ppu_smi" -L
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | tail -n 4
fi

if (( detected_local_ppus != local_ppus )); then
  echo "[vggt-pipeline] PyTorch sees $detected_local_ppus PPUs but DLC topology requests $local_ppus." >&2
  exit 2
fi

python - <<'PY'
import importlib
import json
import torch

required = ("transformers", "accelerate", "deepspeed", "safetensors", "lmdb")
versions = {"torch": torch.__version__}
for name in required:
    try:
        module = importlib.import_module(name)
    except Exception as error:
        raise RuntimeError(
            f"Required package {name!r} is unavailable in the PPU image: {error}. "
            "Install the PPU-compatible build; do not replace the preinstalled PyTorch."
        ) from error
    versions[name] = getattr(module, "__version__", "UNKNOWN")
if not torch.cuda.is_available():
    raise RuntimeError("PPU CUDA-compatible torch.cuda runtime is unavailable")
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("The visible PPU runtime does not report BF16 support")
versions["device_count"] = torch.cuda.device_count()
versions["device_0"] = torch.cuda.get_device_name(0)
print("[vggt-pipeline] runtime=" + json.dumps(versions, sort_keys=True))
PY

if [[ "$VLM_ATTN_IMPLEMENTATION" == "flash_attention_2" ]]; then
  python - <<'PY'
try:
    import flash_attn
except Exception as error:
    raise RuntimeError(
        "VGGT_VLM_ATTN_IMPLEMENTATION=flash_attention_2 requires the PPU-specific "
        f"flash-attn build: {error}"
    ) from error
print(f"[vggt-pipeline] PPU flash-attn={getattr(flash_attn, '__version__', 'UNKNOWN')}")
PY
fi

# Do not override NCCL_* here. Official PAI-PPU images configure ACCL-P and
# their network interface settings. This collective smoke test verifies the
# effective configuration before hours of cache generation begin.
run_command torchrun --standalone --nnodes=1 --nproc-per-node="$local_ppus" \
  "$DRIVEDREAMER_ROOT/tools/check_ppu_runtime.py"

if [[ "${VGGT_PIPELINE_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[vggt-pipeline] PPU preflight passed; VGGT_PIPELINE_PREFLIGHT_ONLY=1, stopping."
  exit 0
fi

phase="token-model"
if [[ -f "$VGGT_BASE_VLM/config.json" && -f "$VGGT_BASE_VLM/added_custom_token_id_map.json" ]]; then
  python "$DRIVEDREAMER_ROOT/tools/validate_vggt_token_model.py" \
    --model-dir "$VGGT_BASE_VLM" \
    --tokens-file "$tokens_file" \
    --expected-count 15
  echo "[vggt-pipeline] Token model already complete; generation skipped."
else
  if [[ -e "$VGGT_BASE_VLM" ]]; then
    echo "[vggt-pipeline] Incomplete token-model output exists: $VGGT_BASE_VLM" >&2
    echo "[vggt-pipeline] Move it aside and rerun; the pipeline will not delete or overwrite it." >&2
    exit 2
  fi
  require_dir "$VGGT_SOURCE_VLM"
  require_file "$VGGT_SOURCE_VLM/config.json"
  mkdir -p "$(dirname -- "$VGGT_BASE_VLM")"
  run_command env \
    CUDA_VISIBLE_DEVICES="${VGGT_TOKEN_VISIBLE_DEVICE:-0}" \
    VGGT_TOKEN_DEVICE="${VGGT_TOKEN_DEVICE:-cuda}" \
    VLM_ATTN_IMPLEMENTATION="$VLM_ATTN_IMPLEMENTATION" \
    bash "$DRIVEDREAMER_ROOT/7-add_vggt_tokens.sh"
  python "$DRIVEDREAMER_ROOT/tools/validate_vggt_token_model.py" \
    --model-dir "$VGGT_BASE_VLM" \
    --tokens-file "$tokens_file" \
    --expected-count 15
fi

phase="vggt-cache"
require_file "$NAVSIM_DATALIST_PATH"
require_dir "$DATA_ROOT/meta/${SPLIT:-train}"
if [[ -n "${VGGT_CACHE_MAX_SAMPLES:-}" && "${VGGT_DEBUG:-0}" != "1" ]]; then
  echo "[vggt-pipeline] VGGT_CACHE_MAX_SAMPLES is set for a formal run." >&2
  echo "[vggt-pipeline] Unset it; a partial cache cannot satisfy the full datalist manifest." >&2
  exit 2
fi
if [[ ! -f "$cache_manifest" ]]; then
  require_dir "$VGGT_REPO"
  require_file "$VGGT_CHECKPOINT"
  require_dir "$NAVSIM_TRAINVAL_SENSOR_ROOT"
  mkdir -p "$NAVSIM_VGGT_CACHE_ROOT"
  run_command env \
    VGGT_CACHE_NUM_PROCESSES="$local_ppus" \
    VGGT_CACHE_BATCH_SIZE="$VGGT_CACHE_BATCH_SIZE" \
    VGGT_CACHE_MAP_SIZE_GB="$VGGT_CACHE_MAP_SIZE_GB" \
    bash "$DRIVEDREAMER_ROOT/tools/cache_vggt_queries.sh"
else
  echo "[vggt-pipeline] Cache manifest exists; generation skipped and validation started."
fi
require_file "$cache_manifest"
if [[ "${VGGT_CACHE_FULL_VALIDATE:-1}" == "1" ]]; then
  run_command "${cache_validate[@]}"
else
  python - "$cache_manifest" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
manifest = json.loads(path.read_text(encoding="utf-8"))
expected = {
    "complete": True,
    "query_count": 195,
    "feature_dim": 1024,
    "teacher_layer_index": 11,
    "teacher_attention_branch": "global",
    "include_special_tokens": True,
}
for key, value in expected.items():
    if manifest.get(key) != value:
        raise RuntimeError(f"Invalid cache manifest {key}: {manifest.get(key)!r} != {value!r}")
print(f"[vggt-pipeline] Cache manifest contract PASS: {path}")
PY
fi

if [[ "${VGGT_PIPELINE_SKIP_TRAIN:-0}" == "1" ]]; then
  echo "[vggt-pipeline] Preparation complete; VGGT_PIPELINE_SKIP_TRAIN=1, stopping."
  exit 0
fi

phase="training-smoke"
smoke_run_id="${RUN_ID}-smoke"
smoke_marker="$NAVSIM_EXP_ROOT/$smoke_run_id/.vggt_smoke_complete"
if [[ "${VGGT_RUN_SMOKE_BEFORE_FORMAL:-1}" == "1" ]]; then
  if [[ -f "$smoke_marker" ]]; then
    echo "[vggt-pipeline] PPU V2 smoke already passed: $smoke_marker"
  else
    run_command env \
      VGGT_DEBUG=1 \
      VGGT_INTERVENTION_INTERVAL=1 \
      MAX_TRAIN_STEPS=2 \
      SAVE_INTERVAL=999999 \
      TRAINING_SKIP_FINAL_SAVE=1 \
      RUN_ID="$smoke_run_id" \
      bash "$DRIVEDREAMER_ROOT/8-train_vggt_action.sh"
    mkdir -p "$(dirname -- "$smoke_marker")"
    touch "$smoke_marker"
    echo "[vggt-pipeline] PPU V2 forward/backward smoke passed."
  fi
fi

phase="training"
run_command bash "$DRIVEDREAMER_ROOT/8-train_vggt_action.sh"

phase="post-training-diagnostics"
run_dir="$NAVSIM_EXP_ROOT/$RUN_ID"
if [[ -f "$run_dir/vggt_diagnostics.jsonl" ]]; then
  run_command python "$DRIVEDREAMER_ROOT/tools/diagnose_vggt_training.py" "$run_dir"
fi
echo "[vggt-pipeline] COMPLETE run_dir=$run_dir"
