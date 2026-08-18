#!/usr/bin/env bash
# One-command dense VGGT cache generation and bottleneck training on PAI-DLC.
#
# Default formal geometry:
#   1 DLC node x 16 PPU x batch 2 x accumulation 1 = effective batch 32.
# The VGGT teacher is used only for offline cache generation. Training uses the
# standard WorldAction VLM and never runs the old 15-token VGGT-V2 preparation.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

phase="bootstrap"
on_error() {
  local status=$?
  echo "[vggt-dense-dlc] FAILED phase=$phase exit_code=$status" >&2
  exit "$status"
}
trap on_error ERR

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[vggt-dense-dlc] Missing required file: $1" >&2
    exit 2
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "[vggt-dense-dlc] Missing required directory: $1" >&2
    exit 2
  fi
}

require_positive_integer() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "[vggt-dense-dlc] $name must be a positive integer, got: $value" >&2
    exit 2
  fi
}

print_command() {
  printf '[vggt-dense-dlc] DRY-RUN:'
  printf ' %q' "$@"
  printf '\n'
}

phase="topology"
dry_run="${VGGT_DENSE_DLC_DRY_RUN:-0}"
num_machines="${NUM_MACHINES:-${WORLD_SIZE:-1}}"
machine_rank="${MACHINE_RANK:-${RANK:-0}}"
if [[ "$dry_run" == "1" ]]; then
  detected_local_ppus="${NPROC_PER_NODE:-${LOCAL_NUM_PROCESSES:-16}}"
else
  detected_local_ppus="$(python -c 'import torch; print(torch.cuda.device_count())')"
fi
local_ppus="${LOCAL_NUM_PROCESSES:-${NPROC_PER_NODE:-$detected_local_ppus}}"
global_processes="${NUM_PROCESSES:-$((num_machines * local_ppus))}"
expected_ppus="${VGGT_DENSE_EXPECTED_PPU_COUNT:-16}"

require_positive_integer NUM_MACHINES "$num_machines"
if ! [[ "$machine_rank" =~ ^[0-9]+$ ]]; then
  echo "[vggt-dense-dlc] MACHINE_RANK must be a non-negative integer, got: $machine_rank" >&2
  exit 2
fi
require_positive_integer LOCAL_NUM_PROCESSES "$local_ppus"
require_positive_integer NUM_PROCESSES "$global_processes"
require_positive_integer VGGT_DENSE_EXPECTED_PPU_COUNT "$expected_ppus"
if (( num_machines != 1 || machine_rank != 0 )); then
  echo "[vggt-dense-dlc] The combined cache+train launcher requires one DLC node." >&2
  echo "[vggt-dense-dlc] Use the two 11-* scripts directly for an explicit multi-node launch." >&2
  exit 2
fi
if (( global_processes != local_ppus )); then
  echo "[vggt-dense-dlc] Invalid single-node topology: local=$local_ppus global=$global_processes" >&2
  exit 2
fi
if (( local_ppus != expected_ppus )); then
  echo "[vggt-dense-dlc] Expected $expected_ppus visible PPUs, found/requested $local_ppus." >&2
  exit 2
fi

per_device_batch="${PER_DEVICE_BATCH_SIZE:-2}"
gradient_accumulation="${GRADIENT_ACCUMULATION_STEPS:-1}"
target_effective_batch="${TARGET_EFFECTIVE_BATCH_SIZE:-32}"
require_positive_integer PER_DEVICE_BATCH_SIZE "$per_device_batch"
require_positive_integer GRADIENT_ACCUMULATION_STEPS "$gradient_accumulation"
require_positive_integer TARGET_EFFECTIVE_BATCH_SIZE "$target_effective_batch"
effective_batch=$((global_processes * per_device_batch * gradient_accumulation))
if (( effective_batch != target_effective_batch )); then
  echo "[vggt-dense-dlc] Refusing effective batch $effective_batch; expected $target_effective_batch." >&2
  echo "[vggt-dense-dlc] Formula: $global_processes x $per_device_batch x $gradient_accumulation." >&2
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

export NUM_MACHINES="$num_machines"
export MACHINE_RANK="$machine_rank"
export LOCAL_NUM_PROCESSES="$local_ppus"
export NUM_PROCESSES="$global_processes"
export VGGT_DENSE_CACHE_NUM_PROCESSES="${VGGT_DENSE_CACHE_NUM_PROCESSES:-$local_ppus}"
export VGGT_DENSE_CACHE_BATCH_SIZE="${VGGT_DENSE_CACHE_BATCH_SIZE:-1}"
export VGGT_DENSE_CACHE_FULL=1
export PER_DEVICE_BATCH_SIZE="$per_device_batch"
export GRADIENT_ACCUMULATION_STEPS="$gradient_accumulation"
export TARGET_EFFECTIVE_BATCH_SIZE="$target_effective_batch"
export EXPECTED_TRAIN_SAMPLES="${EXPECTED_TRAIN_SAMPLES:-103288}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100000}"
export NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-5000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
export TRAINING_LOGGING_FREQUENCY="${TRAINING_LOGGING_FREQUENCY:-50}"
export BASE_LEARNING_RATE="${BASE_LEARNING_RATE:-1e-5}"
export ACTION_LEARNING_RATE="${ACTION_LEARNING_RATE:-1e-5}"
export VGGT_DENSE_LEARNING_RATE="${VGGT_DENSE_LEARNING_RATE:-5e-5}"
export OPTIMIZER_WEIGHT_DECAY="${OPTIMIZER_WEIGHT_DECAY:-1e-3}"
export VGGT_DENSE_VLM_ATTN_IMPLEMENTATION="${VGGT_DENSE_VLM_ATTN_IMPLEMENTATION:-sdpa}"
export VGGT_DENSE_INTERVENTION_INTERVAL="${VGGT_DENSE_INTERVENTION_INTERVAL:-500}"
export MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-${MASTER_ADDR:-127.0.0.1}}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-${MASTER_PORT:-29691}}"
timestamp="$(date +'%Y%m%d_%H%M%S')"
export RUN_ID="${RUN_ID:-vggt-dense-bottleneck-${PAI_JOB_ID:-$timestamp}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TRITON_LOCAL_CACHE_ROOT:-/tmp/drivedreamer-triton}/${RUN_ID}/node${machine_rank}}"

split="${SPLIT:-train}"
cache_manifest="$NAVSIM_VGGT_DENSE_CACHE_ROOT/vggt_dense/manifest.json"
cache_validate=(
  bash "$DRIVEDREAMER_ROOT/11-precompute_vggt_dense_cache.sh"
  --validate-only
  --datalist-path "$NAVSIM_DATALIST_PATH"
  --data-root "$DATA_ROOT"
  --split "$split"
  --cache-root "$NAVSIM_VGGT_DENSE_CACHE_ROOT"
)

echo "[vggt-dense-dlc] project_root=$DRIVEDREAMER_ROOT"
echo "[vggt-dense-dlc] run_id=$RUN_ID"
echo "[vggt-dense-dlc] topology=nodes:$num_machines node_rank:$machine_rank local_ppus:$local_ppus global_processes:$global_processes"
echo "[vggt-dense-dlc] effective_batch=$effective_batch (per_device=$per_device_batch accumulation=$gradient_accumulation)"
echo "[vggt-dense-dlc] training=steps:$MAX_TRAIN_STEPS warmup:$NUM_WARMUP_STEPS save_every:$SAVE_INTERVAL log_every:$TRAINING_LOGGING_FREQUENCY"
echo "[vggt-dense-dlc] attention=$VGGT_DENSE_VLM_ATTN_IMPLEMENTATION expected_samples=$EXPECTED_TRAIN_SAMPLES"
echo "[vggt-dense-dlc] datalist=$NAVSIM_DATALIST_PATH"
echo "[vggt-dense-dlc] sensor_root=$NAVSIM_TRAINVAL_SENSOR_ROOT"
echo "[vggt-dense-dlc] dense_cache_root=$NAVSIM_VGGT_DENSE_CACHE_ROOT"
echo "[vggt-dense-dlc] experiment_root=$NAVSIM_EXP_ROOT"

if [[ "$dry_run" == "1" ]]; then
  print_command "${PPU_SMI_BIN:-ppu-smi}" -L
  print_command torchrun --standalone --nnodes=1 --nproc-per-node="$local_ppus" "$DRIVEDREAMER_ROOT/tools/check_ppu_runtime.py"
  print_command env VGGT_DENSE_CACHE_FULL=1 VGGT_DENSE_CACHE_NUM_PROCESSES="$local_ppus" bash "$DRIVEDREAMER_ROOT/11-precompute_vggt_dense_cache.sh"
  print_command "${cache_validate[@]}"
  print_command env VGGT_DENSE_INTERVENTION_INTERVAL=1 MAX_TRAIN_STEPS=2 NUM_WARMUP_STEPS=2 SAVE_INTERVAL=999999 TRAINING_LOGGING_FREQUENCY=1 TRAINING_SKIP_FINAL_SAVE=1 RUN_ID="${RUN_ID}-smoke" bash "$DRIVEDREAMER_ROOT/11-train_vggt_dense_bottleneck.sh"
  print_command bash "$DRIVEDREAMER_ROOT/11-train_vggt_dense_bottleneck.sh"
  echo "[vggt-dense-dlc] DRY-RUN complete; no cache or training output was generated."
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
  echo "[vggt-dense-dlc] ppu-smi was not found; use the standard PAI-DLC PPU image." >&2
  exit 2
fi
require_file "${PPU_SDK_RELEASE_FILE:-/usr/local/PPU_SDK/release.yaml}"
"$ppu_smi" -L
if (( detected_local_ppus != local_ppus )); then
  echo "[vggt-dense-dlc] PyTorch sees $detected_local_ppus PPUs, topology requests $local_ppus." >&2
  exit 2
fi

python - <<'PY'
import importlib
import json
import torch

required = ("accelerate", "cv2", "deepspeed", "lmdb", "PIL", "safetensors", "transformers")
versions = {"torch": torch.__version__}
for name in required:
    module = importlib.import_module(name)
    versions[name] = getattr(module, "__version__", "UNKNOWN")
if not torch.cuda.is_available():
    raise RuntimeError("PPU CUDA-compatible torch.cuda runtime is unavailable")
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("The visible PPU runtime does not report BF16 support")
versions["device_count"] = torch.cuda.device_count()
versions["device_0"] = torch.cuda.get_device_name(0)
print("[vggt-dense-dlc] runtime=" + json.dumps(versions, sort_keys=True))
PY

# The official PPU image configures ACCL-P. Do not inject NVIDIA NCCL tuning.
torchrun --standalone --nnodes=1 --nproc-per-node="$local_ppus" \
  "$DRIVEDREAMER_ROOT/tools/check_ppu_runtime.py"
if [[ "${VGGT_DENSE_DLC_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[vggt-dense-dlc] PPU preflight PASS; stopping as requested."
  exit 0
fi

launcher_log_dir="$NAVSIM_EXP_ROOT/launcher_logs"
mkdir -p "$launcher_log_dir" "$TRITON_CACHE_DIR"
launcher_log="$launcher_log_dir/${RUN_ID}.dense-pipeline.log"
exec > >(tee -a "$launcher_log") 2>&1
echo "[vggt-dense-dlc] launcher_log=$launcher_log"

phase="dense-cache"
require_file "$NAVSIM_DATALIST_PATH"
require_dir "$DATA_ROOT/meta/$split"
if [[ "${VGGT_DENSE_DLC_SKIP_CACHE:-0}" != "1" && ! -f "$cache_manifest" ]]; then
  require_dir "$VGGT_REPO"
  require_file "$VGGT_CHECKPOINT"
  require_dir "$NAVSIM_TRAINVAL_SENSOR_ROOT"
  mkdir -p "$NAVSIM_VGGT_DENSE_CACHE_ROOT"
  bash "$DRIVEDREAMER_ROOT/11-precompute_vggt_dense_cache.sh"
elif [[ -f "$cache_manifest" ]]; then
  echo "[vggt-dense-dlc] Complete-manifest candidate found; generation skipped."
else
  echo "[vggt-dense-dlc] Cache generation skipped but manifest is missing: $cache_manifest" >&2
  exit 2
fi
require_file "$cache_manifest"
if [[ "${VGGT_DENSE_CACHE_FULL_VALIDATE:-1}" == "1" ]]; then
  "${cache_validate[@]}"
else
  python - "$cache_manifest" "$EXPECTED_TRAIN_SAMPLES" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
manifest = json.loads(path.read_text(encoding="utf-8"))
expected = {
    "component": "vggt_dense",
    "complete": True,
    "sample_count": int(sys.argv[2]),
    "teacher_layer_index": 23,
    "teacher_layer": "aggregator[-1]",
    "teacher_attention_branch": "full_aggregated_feature",
    "include_special_tokens": False,
    "spatial_pooling": None,
    "feature_dim": 2048,
}
for key, value in expected.items():
    if manifest.get(key) != value:
        raise RuntimeError(f"Invalid dense cache manifest {key}: {manifest.get(key)!r} != {value!r}")
print(f"[vggt-dense-dlc] Dense cache manifest contract PASS: {path}")
PY
fi

if [[ "${VGGT_DENSE_DLC_SKIP_TRAIN:-0}" == "1" ]]; then
  echo "[vggt-dense-dlc] Dense cache preparation complete; training skipped."
  exit 0
fi

phase="training-assets"
require_file "$BASE_VLM/config.json"
require_file "${TRAIN_CONFIG_YAML:-$DRIVEDREAMER_ROOT/starVLA/config/training/cfg_yaw_1225.yaml}"
require_file "$DRIVEDREAMER_ROOT/starVLA/config/training/vggt_dense_bottleneck.yaml"
if [[ -n "${ACTION_ONLY_CHECKPOINT:-}" ]]; then
  require_file "$ACTION_ONLY_CHECKPOINT"
fi
run_dir="$NAVSIM_EXP_ROOT/$RUN_ID"
if [[ -e "$run_dir" ]]; then
  echo "[vggt-dense-dlc] Refusing to overwrite existing experiment: $run_dir" >&2
  exit 2
fi

phase="training-smoke"
smoke_run_id="${RUN_ID}-smoke"
smoke_dir="$NAVSIM_EXP_ROOT/$smoke_run_id"
smoke_marker="$smoke_dir/.vggt_dense_smoke_complete"
if [[ "${VGGT_DENSE_RUN_SMOKE_BEFORE_FORMAL:-1}" == "1" ]]; then
  if [[ -f "$smoke_marker" ]]; then
    echo "[vggt-dense-dlc] Prior two-step smoke PASS: $smoke_marker"
  else
    if [[ -e "$smoke_dir" ]]; then
      echo "[vggt-dense-dlc] Incomplete smoke directory exists: $smoke_dir" >&2
      echo "[vggt-dense-dlc] Move it aside or choose a new RUN_ID; nothing is overwritten." >&2
      exit 2
    fi
    env \
      VGGT_DENSE_INTERVENTION_INTERVAL=1 \
      MAX_TRAIN_STEPS=2 \
      NUM_WARMUP_STEPS=2 \
      SAVE_INTERVAL=999999 \
      TRAINING_LOGGING_FREQUENCY=1 \
      TRAINING_SKIP_FINAL_SAVE=1 \
      RUN_ID="$smoke_run_id" \
      bash "$DRIVEDREAMER_ROOT/11-train_vggt_dense_bottleneck.sh"
    touch "$smoke_marker"
    echo "[vggt-dense-dlc] Two-step forward/backward smoke PASS."
  fi
elif [[ "${VGGT_DENSE_RUN_SMOKE_BEFORE_FORMAL:-1}" != "0" ]]; then
  echo "[vggt-dense-dlc] VGGT_DENSE_RUN_SMOKE_BEFORE_FORMAL must be 0 or 1" >&2
  exit 2
fi

phase="formal-training"
bash "$DRIVEDREAMER_ROOT/11-train_vggt_dense_bottleneck.sh"
echo "[vggt-dense-dlc] COMPLETE run_dir=$run_dir"
