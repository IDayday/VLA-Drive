#!/usr/bin/env bash
# One-command PAI-DLC V3 pipeline:
# runtime -> token model -> teacher codec/gates -> V3 cache -> smoke -> train -> student gate.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

phase="bootstrap"
on_error() {
  status=$?
  echo "[vggt-v3] FAILED phase=$phase exit_code=$status" >&2
  exit "$status"
}
trap on_error ERR

print_command() {
  printf '[vggt-v3] DRY-RUN:'
  printf ' %q' "$@"
  printf '\n'
}

require_file() { [[ -f "$1" ]] || { echo "[vggt-v3] Missing file: $1" >&2; exit 2; }; }
require_dir() { [[ -d "$1" ]] || { echo "[vggt-v3] Missing directory: $1" >&2; exit 2; }; }

dry_run="${VGGT_PIPELINE_DRY_RUN:-0}"
expected_ppus="${VGGT_EXPECTED_PPU_COUNT:-16}"
if [[ "$dry_run" == "1" ]]; then
  detected_ppus="${LOCAL_NUM_PROCESSES:-${NPROC_PER_NODE:-$expected_ppus}}"
else
  detected_ppus="$(python -c 'import torch; print(torch.cuda.device_count())')"
fi
local_ppus="${LOCAL_NUM_PROCESSES:-${NPROC_PER_NODE:-$detected_ppus}}"
num_machines="${NUM_MACHINES:-${WORLD_SIZE:-1}}"
machine_rank="${MACHINE_RANK:-${RANK:-0}}"
if ! [[ "$expected_ppus" =~ ^[1-9][0-9]*$ && "$local_ppus" =~ ^[1-9][0-9]*$ ]]; then
  echo "[vggt-v3] PPU counts must be positive integers" >&2
  exit 2
fi
if (( num_machines != 1 || machine_rank != 0 || local_ppus != expected_ppus )); then
  echo "[vggt-v3] This cache+train pipeline requires one DLC node with $expected_ppus PPUs." >&2
  echo "[vggt-v3] Found nodes=$num_machines rank=$machine_rank local_ppus=$local_ppus." >&2
  exit 2
fi
if [[ "$dry_run" != "1" && "$detected_ppus" != "$local_ppus" ]]; then
  echo "[vggt-v3] PyTorch sees $detected_ppus devices, requested $local_ppus." >&2
  exit 2
fi

export NUM_MACHINES=1 MACHINE_RANK=0
export LOCAL_NUM_PROCESSES="$local_ppus" NUM_PROCESSES="$local_ppus"
export VGGT_CODEC_NUM_PROCESSES="${VGGT_CODEC_NUM_PROCESSES:-$local_ppus}"
export VGGT_CACHE_NUM_PROCESSES="${VGGT_CACHE_NUM_PROCESSES:-$local_ppus}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-32}"
effective_batch=$((local_ppus * PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
if (( effective_batch != TARGET_EFFECTIVE_BATCH_SIZE )); then
  echo "[vggt-v3] Effective batch $effective_batch != target $TARGET_EFFECTIVE_BATCH_SIZE" >&2
  exit 2
fi
export VLM_ATTN_IMPLEMENTATION="${VGGT_VLM_ATTN_IMPLEMENTATION:-sdpa}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100000}"
export NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-5000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
export TRAINING_LOGGING_FREQUENCY="${TRAINING_LOGGING_FREQUENCY:-50}"
export BASE_LEARNING_RATE="${BASE_LEARNING_RATE:-1e-5}"
export ACTION_LEARNING_RATE="${ACTION_LEARNING_RATE:-1e-5}"
export VGGT_LEARNING_RATE="${VGGT_LEARNING_RATE:-3e-5}"
export OPTIMIZER_WEIGHT_DECAY="${OPTIMIZER_WEIGHT_DECAY:-1e-3}"
export RUN_ID="${RUN_ID:-vggt-query-v3-layer11-global-codec-m195-${PAI_JOB_ID:-$(date +'%Y%m%d_%H%M%S')}}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-${MASTER_PORT:-29689}}"
if ! [[ "$MAX_TRAIN_STEPS" =~ ^[1-9][0-9]*$ && "$SAVE_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
  echo "[vggt-v3] MAX_TRAIN_STEPS and SAVE_INTERVAL must be positive integers" >&2
  exit 2
fi
if [[ "${VGGT_V3_RUN_DOWNSTREAM_AFTER_TRAIN:-1}" == "1" ]] \
  && (( MAX_TRAIN_STEPS % SAVE_INTERVAL != 0 )); then
  echo "[vggt-v3] Downstream gate needs a steps_${MAX_TRAIN_STEPS} checkpoint; SAVE_INTERVAL must divide MAX_TRAIN_STEPS." >&2
  exit 2
fi

echo "[vggt-v3] root=$DRIVEDREAMER_ROOT run=$RUN_ID ppus=$local_ppus effective_batch=$effective_batch"
echo "[vggt-v3] codec=$VGGT_V3_CODEC cache=$NAVSIM_VGGT_V3_CACHE_ROOT"
echo "[vggt-v3] train_steps=$MAX_TRAIN_STEPS save_interval=$SAVE_INTERVAL attention=$VLM_ATTN_IMPLEMENTATION"

if [[ "$dry_run" == "1" ]]; then
  print_command torchrun --standalone --nnodes=1 --nproc-per-node="$local_ppus" tools/check_ppu_runtime.py
  print_command bash 7-add_vggt_tokens.sh
  print_command python tools/validate_vggt_native_tail.py --datalist-path "$NAVSIM_DATALIST_PATH" --data-root "$DATA_ROOT" --sensor-root "$NAVSIM_TRAINVAL_SENSOR_ROOT" --vggt-repo "$VGGT_REPO" --vggt-checkpoint "$VGGT_CHECKPOINT" --output "$VGGT_V3_CODEC_ROOT/native_tail_contract.json"
  print_command bash tools/train_vggt_native_codec.sh
  print_command bash tools/cache_vggt_v3_queries.sh
  print_command python tools/precompute_vggt_query_cache.py --validate-only --datalist-path "$NAVSIM_DATALIST_PATH" --data-root "$DATA_ROOT" --cache-root "$NAVSIM_VGGT_V3_CACHE_ROOT"
  print_command env VGGT_DEBUG=1 MAX_TRAIN_STEPS=2 RUN_ID="${RUN_ID}-smoke" bash 8-train_vggt_v3_action.sh
  print_command env VGGT_DEBUG=0 TRAINING_SKIP_FINAL_SAVE=0 bash 8-train_vggt_v3_action.sh
  print_command python tools/diagnose_vggt_training.py "$NAVSIM_EXP_ROOT/$RUN_ID"
  print_command python tools/evaluate_vggt_v3_native_downstream.py --run-dir "$NAVSIM_EXP_ROOT/$RUN_ID" --checkpoint-step "$MAX_TRAIN_STEPS" --base-vlm "$VGGT_BASE_VLM" --native-codec "$VGGT_V3_CODEC" --vggt-repo "$VGGT_REPO" --vggt-checkpoint "$VGGT_CHECKPOINT" --cache-root "$NAVSIM_VGGT_V3_CACHE_ROOT" --datalist-path "$NAVSIM_DATALIST_PATH" --data-root "$DATA_ROOT" --output "$NAVSIM_EXP_ROOT/$RUN_ID/v3_native_downstream.json"
  exit 0
fi

log_dir="$NAVSIM_EXP_ROOT/launcher_logs"
mkdir -p "$log_dir"
exec > >(tee -a "$log_dir/${RUN_ID}.pipeline.log") 2>&1

phase="runtime-preflight"
if command -v ppu-smi >/dev/null 2>&1; then
  ppu-smi -L
elif [[ -x /usr/local/PPU_SDK/ppu-smi/bin/ppu-smi ]]; then
  /usr/local/PPU_SDK/ppu-smi/bin/ppu-smi -L
else
  echo "[vggt-v3] ppu-smi not found; use the official PAI-PPU DLC image." >&2
  exit 2
fi
python - <<'PY'
import importlib
import json
import torch
versions = {"torch": torch.__version__, "device_count": torch.cuda.device_count()}
for name in ("accelerate", "deepspeed", "lmdb", "safetensors", "transformers"):
    module = importlib.import_module(name)
    versions[name] = getattr(module, "__version__", "UNKNOWN")
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise RuntimeError("The DLC CUDA-compatible PPU runtime must support BF16")
print("[vggt-v3] runtime=" + json.dumps(versions, sort_keys=True))
PY
torchrun --standalone --nnodes=1 --nproc-per-node="$local_ppus" tools/check_ppu_runtime.py
if [[ "${VGGT_PIPELINE_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[vggt-v3] preflight complete"
  exit 0
fi

phase="paths"
require_file "$NAVSIM_DATALIST_PATH"
require_dir "$DATA_ROOT/meta/${SPLIT:-train}"
require_dir "$NAVSIM_TRAINVAL_SENSOR_ROOT"
require_dir "$VGGT_REPO"
require_file "$VGGT_CHECKPOINT"

phase="token-model"
tokens_file="$DRIVEDREAMER_ROOT/starVLA/model/modules/vlm/tools/add_qwen_special_tokens/vggt_global_query_tokens_15.txt"
if [[ -f "$VGGT_BASE_VLM/config.json" && -f "$VGGT_BASE_VLM/added_custom_token_id_map.json" ]]; then
  python tools/validate_vggt_token_model.py --model-dir "$VGGT_BASE_VLM" --tokens-file "$tokens_file" --expected-count 15
else
  [[ ! -e "$VGGT_BASE_VLM" ]] || { echo "[vggt-v3] Incomplete token model exists: $VGGT_BASE_VLM" >&2; exit 2; }
  bash 7-add_vggt_tokens.sh
  python tools/validate_vggt_token_model.py --model-dir "$VGGT_BASE_VLM" --tokens-file "$tokens_file" --expected-count 15
fi

phase="native-codec"
python tools/validate_vggt_native_tail.py \
  --datalist-path "$NAVSIM_DATALIST_PATH" \
  --data-root "$DATA_ROOT" \
  --sensor-root "$NAVSIM_TRAINVAL_SENSOR_ROOT" \
  --vggt-repo "$VGGT_REPO" \
  --vggt-checkpoint "$VGGT_CHECKPOINT" \
  --output "$VGGT_V3_CODEC_ROOT/native_tail_contract.json"
codec_report="$VGGT_V3_CODEC_ROOT/report.json"
if [[ ! -f "$VGGT_V3_CODEC" || ! -f "$codec_report" ]]; then
  bash tools/train_vggt_native_codec.sh
else
  echo "[vggt-v3] codec artifacts exist; training skipped and gates rechecked"
fi
require_file "$VGGT_V3_CODEC"
require_file "$codec_report"
python - "$codec_report" "$VGGT_V3_CODEC" <<'PY'
import json, sys, torch
report = json.load(open(sys.argv[1], encoding="utf-8"))
checkpoint = torch.load(sys.argv[2], map_location="cpu", weights_only=True)
if not report.get("gates", {}).get("teacher_codec_downstream", False):
    raise RuntimeError("V3 codec did not pass reconstruction/tail/camera gates")
if report.get("design", {}).get("compact_tokens") != 195:
    raise RuntimeError("V3 codec does not preserve the 195-token contract")
source = report.get("design", {}).get("encoder_source", {})
if source.get("layer_index") != 11 or source.get("attention_branch") != "global":
    raise RuntimeError("V3 codec encoder source is not layer-11 global only")
if checkpoint.get("gates") != report.get("gates"):
    raise RuntimeError("V3 codec checkpoint/report gate identity mismatch")
if checkpoint.get("schema_version") != 3:
    raise RuntimeError("V3 codec does not use the layer11-global-only decoder schema")
if checkpoint.get("config", {}).get("latent_dim") != 1024:
    raise RuntimeError("V3 codec latent dimension is not 1024")
print("[vggt-v3] codec gates PASS")
PY

phase="v3-cache"
cache_manifest="$NAVSIM_VGGT_V3_CACHE_ROOT/vggt_query/manifest.json"
if [[ ! -f "$cache_manifest" ]]; then
  bash tools/cache_vggt_v3_queries.sh
else
  echo "[vggt-v3] V3 cache manifest exists; generation skipped"
fi
python tools/precompute_vggt_query_cache.py \
  --validate-only \
  --datalist-path "$NAVSIM_DATALIST_PATH" \
  --data-root "$DATA_ROOT" \
  --cache-root "$NAVSIM_VGGT_V3_CACHE_ROOT"

if [[ "${VGGT_PIPELINE_SKIP_TRAIN:-0}" == "1" ]]; then
  echo "[vggt-v3] codec/cache complete; training skipped"
  exit 0
fi

phase="training-smoke"
smoke_id="${RUN_ID}-smoke"
smoke_marker="$NAVSIM_EXP_ROOT/$smoke_id/.vggt_v3_smoke_complete"
if [[ "${VGGT_RUN_SMOKE_BEFORE_FORMAL:-1}" == "1" && ! -f "$smoke_marker" ]]; then
  env VGGT_DEBUG=1 VGGT_INTERVENTION_INTERVAL=1 MAX_TRAIN_STEPS=2 \
    SAVE_INTERVAL=999999 TRAINING_SKIP_FINAL_SAVE=1 RUN_ID="$smoke_id" \
    bash 8-train_vggt_v3_action.sh
  mkdir -p "$(dirname -- "$smoke_marker")"
  touch "$smoke_marker"
fi

phase="formal-training"
formal_marker="$NAVSIM_EXP_ROOT/$RUN_ID/.vggt_v3_training_complete"
if [[ ! -f "$formal_marker" ]]; then
  env VGGT_DEBUG=0 TRAINING_SKIP_FINAL_SAVE=0 bash 8-train_vggt_v3_action.sh
  touch "$formal_marker"
else
  echo "[vggt-v3] formal training marker exists; training skipped"
fi

phase="planner-utilization-diagnostics"
run_dir="$NAVSIM_EXP_ROOT/$RUN_ID"
if [[ -f "$run_dir/vggt_diagnostics.jsonl" ]]; then
  python tools/diagnose_vggt_training.py "$run_dir" \
    | tee "$run_dir/vggt_diagnostics_summary.json"
fi
phase="student-native-downstream"
if [[ "${VGGT_V3_RUN_DOWNSTREAM_AFTER_TRAIN:-1}" == "1" ]]; then
  python tools/evaluate_vggt_v3_native_downstream.py \
    --run-dir "$run_dir" \
    --checkpoint-step "$MAX_TRAIN_STEPS" \
    --base-vlm "$VGGT_BASE_VLM" \
    --native-codec "$VGGT_V3_CODEC" \
    --vggt-repo "$VGGT_REPO" \
    --vggt-checkpoint "$VGGT_CHECKPOINT" \
    --cache-root "$NAVSIM_VGGT_V3_CACHE_ROOT" \
    --datalist-path "$NAVSIM_DATALIST_PATH" \
    --data-root "$DATA_ROOT" \
    --samples "${VGGT_V3_DOWNSTREAM_SAMPLES:-256}" \
    --output "$run_dir/v3_native_downstream.json"
fi

echo "[vggt-v3] COMPLETE run_dir=$NAVSIM_EXP_ROOT/$RUN_ID"
