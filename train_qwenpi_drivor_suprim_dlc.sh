#!/usr/bin/env bash
# One-task, non-interactive DLC training for QwenPI-DrivoRSuprim.
#
# Frozen pretrained Qwen -> trainable Q-Former + Flow-DiT -> online DrivoR
# labels -> unified DriveSuprim static/dynamic scoring.  All trainable modules
# use one optimizer, scheduler, backward pass, and joint Accelerator checkpoint.
# Production topology: 8 accelerators x micro-batch 8 x accumulation 1 = 64.

set -Eeuo pipefail

script_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="${VLA_PROJECT_ROOT:-$script_root}"
if [[ ! -f "$project_root/starVLA/model/framework/QwenPI_DrivoRSuprim.py" ]]; then
  echo "[qds] QwenPI-DrivoRSuprim is absent under VLA_PROJECT_ROOT=$project_root" >&2
  echo "[qds] Point VLA_PROJECT_ROOT at the checkout containing this implementation." >&2
  exit 2
fi
source "$project_root/load_env.sh"
cd "$project_root"

required_value() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "[qds] required environment variable is empty: $name" >&2
    exit 2
  fi
}

required_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "[qds] missing required path: $path" >&2
    exit 2
  fi
}

export QDS_CONFIG_YAML="${QDS_CONFIG_YAML:-$project_root/starVLA/config/training/qwenpi_drivor_suprim.yaml}"
export QWEN_VLM_PATH="${QWEN_VLM_PATH:-${BASE_VLM:-}}"
export QDS_ASSET_ROOT="${QDS_ASSET_ROOT:-${DRIVEDREAMER_SHARED_ROOT:-$project_root}/ddp_drs_assets}"
export SUPRIM_VOCAB_PATH="${SUPRIM_VOCAB_PATH:-$QDS_ASSET_ROOT/drivesuprim/test_8192_kmeans.npy}"
export QDS_STATIC_SCORE_AGGREGATE="${QDS_STATIC_SCORE_AGGREGATE:-$QDS_ASSET_ROOT/drivesuprim/official_cache/traj_pdm_v2/ori/vocab_score_8192_navtrain_final/navtrain.pkl}"
export QDS_STATIC_SCORE_SHARDS="${QDS_STATIC_SCORE_SHARDS:-$QDS_ASSET_ROOT/drivesuprim/static_scores_navtrain_sharded}"
export QDS_SPLIT_STATIC_SCORE_CACHE="${QDS_SPLIT_STATIC_SCORE_CACHE:-1}"
export QDS_DOWNLOAD_STATIC_SCORE="${QDS_DOWNLOAD_STATIC_SCORE:-1}"

export DATA_ROOT="${DATA_ROOT:-$project_root/navsim_dataset}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$project_root/navsim_dataset_raw}"
export NAVSIM_DATALIST_PATH="${NAVSIM_DATALIST_PATH:-$project_root/train_meta.json}"
export QDS_SPLIT="${QDS_SPLIT:-train}"
export QDS_NAVSIM_LOG_PATH="${QDS_NAVSIM_LOG_PATH:-$OPENSCENE_DATA_ROOT/navsim_logs/trainval}"
export QDS_NAVSIM_SENSOR_PATH="${QDS_NAVSIM_SENSOR_PATH:-${NAVSIM_TRAINVAL_SENSOR_ROOT:-$OPENSCENE_DATA_ROOT/sensor_blobs/trainval}}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$OPENSCENE_DATA_ROOT/maps}"
export NAVSIM_METRIC_CACHE_ROOT="${NAVSIM_METRIC_CACHE_ROOT:-${NAVSIM_EXP_ROOT:-$project_root/navsim_exp}/qds_metric_cache_navtrain}"
export QDS_BUILD_METRIC_CACHE="${QDS_BUILD_METRIC_CACHE:-auto}"

export VLA_OUTPUT_ROOT="${VLA_OUTPUT_ROOT:-${NAVSIM_EXP_ROOT:-$project_root/navsim_exp}/qwenpi_drivor_suprim}"
export QDS_RUN_ID="${QDS_RUN_ID:-qwenpi-drivor-suprim-$(date +'%Y%m%d_%H%M%S')}"
export VLA_MAX_TRAIN_STEPS="${VLA_MAX_TRAIN_STEPS:-100000}"
export VLA_WARMUP_STEPS="${VLA_WARMUP_STEPS:-5000}"
export VLA_SAVE_INTERVAL="${VLA_SAVE_INTERVAL:-5000}"
export VLA_EVAL_INTERVAL="${VLA_EVAL_INTERVAL:-200000}"
export VLA_LOG_INTERVAL="${VLA_LOG_INTERVAL:-20}"
export VLA_BATCH_SIZE="${VLA_BATCH_SIZE:-8}"
export QDS_TARGET_EFFECTIVE_BATCH="${QDS_TARGET_EFFECTIVE_BATCH:-64}"
export VLA_RESUME_CKPT="${VLA_RESUME_CKPT:-none}"
export NAVSIM_METRIC_WORKERS="${NAVSIM_METRIC_WORKERS:-1}"
export QDS_METRIC_CACHE_WORKERS="${QDS_METRIC_CACHE_WORKERS:-16}"
export QDS_DEEPSPEED_CONFIG="${QDS_DEEPSPEED_CONFIG:-$project_root/starVLA/config/deepseeds/deepspeed_zero2.yaml}"
export QDS_MAIN_PROCESS_PORT="${QDS_MAIN_PROCESS_PORT:-29691}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-qwenpi-drivor-suprim}"
export WANDB_ENTITY="${WANDB_ENTITY:-local}"
export NAVSIM_USE_FEATURE_CACHE=0
export NAVSIM_FEATURE_CACHE_ROOT=""
export NAVSIM_VIDEO_SOURCE="${NAVSIM_VIDEO_SOURCE:-images}"
export NAVSIM_NUM_WORKERS="${NAVSIM_NUM_WORKERS:-4}"
export NAVSIM_PREFETCH_FACTOR="${NAVSIM_PREFETCH_FACTOR:-2}"
export NAVSIM_PIN_MEMORY="${NAVSIM_PIN_MEMORY:-1}"
export NAVSIM_WORKER_THREADS="${NAVSIM_WORKER_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$project_root:$project_root/navsim:${PYTHONPATH:-}"

for name in QWEN_VLM_PATH DATA_ROOT OPENSCENE_DATA_ROOT NAVSIM_DATALIST_PATH \
  SUPRIM_VOCAB_PATH NAVSIM_METRIC_CACHE_ROOT VLA_OUTPUT_ROOT; do
  required_value "$name"
done
for path in "$QDS_CONFIG_YAML" "$QWEN_VLM_PATH/config.json" \
  "$QWEN_VLM_PATH/model.safetensors" "$DATA_ROOT/meta/$QDS_SPLIT" \
  "$NAVSIM_DATALIST_PATH" "$SUPRIM_VOCAB_PATH" "$QDS_DEEPSPEED_CONFIG"; do
  required_path "$path"
done

actual_devices="$(python -c 'import torch; print(torch.cuda.device_count())')"
if ! [[ "$actual_devices" =~ ^[1-9][0-9]*$ ]]; then
  echo "[qds] joint training requires at least one visible accelerator" >&2
  exit 2
fi
export QDS_LOCAL_PROCESSES="${QDS_LOCAL_PROCESSES:-8}"
for name in QDS_LOCAL_PROCESSES VLA_BATCH_SIZE QDS_TARGET_EFFECTIVE_BATCH \
  VLA_MAX_TRAIN_STEPS VLA_WARMUP_STEPS VLA_SAVE_INTERVAL VLA_EVAL_INTERVAL \
  VLA_LOG_INTERVAL NAVSIM_METRIC_WORKERS QDS_METRIC_CACHE_WORKERS; do
  if ! [[ "${!name}" =~ ^[0-9]+$ ]]; then
    echo "[qds] $name must be a non-negative integer, got ${!name}" >&2
    exit 2
  fi
done
if (( QDS_LOCAL_PROCESSES < 1 || QDS_LOCAL_PROCESSES > actual_devices )); then
  echo "[qds] requested $QDS_LOCAL_PROCESSES devices; $actual_devices are visible" >&2
  exit 2
fi
micro_global=$((QDS_LOCAL_PROCESSES * VLA_BATCH_SIZE))
if (( micro_global < 1 || QDS_TARGET_EFFECTIVE_BATCH % micro_global != 0 )); then
  echo "[qds] effective batch must be divisible by processes x micro batch" >&2
  exit 2
fi
export VLA_GRAD_ACCUM_STEPS=$((QDS_TARGET_EFFECTIVE_BATCH / micro_global))
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((QDS_LOCAL_PROCESSES - 1)))"
fi

mkdir -p "$QDS_ASSET_ROOT/drivesuprim/official_cache" "$VLA_OUTPUT_ROOT/launcher_logs"
if [[ -n "${SUPRIM_STATIC_SCORE_CACHE:-}" ]]; then
  required_path "$SUPRIM_STATIC_SCORE_CACHE"
else
  if [[ ! -f "$QDS_STATIC_SCORE_AGGREGATE" ]]; then
    if [[ "$QDS_DOWNLOAD_STATIC_SCORE" != "1" ]]; then
      echo "[qds] official static score cache is missing: $QDS_STATIC_SCORE_AGGREGATE" >&2
      exit 2
    fi
    echo "[qds] downloading the official DriveSuprim navtrain static-score cache"
    export QDS_HF_LOCAL_DIR="$QDS_ASSET_ROOT/drivesuprim/official_cache"
    python - <<'PY'
import os
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="alkaid-2000/DriveSuprim",
    filename="traj_pdm_v2/ori/vocab_score_8192_navtrain_final/navtrain.pkl",
    local_dir=os.environ["QDS_HF_LOCAL_DIR"],
)
print(f"[qds] downloaded static score cache: {path}")
PY
  fi
  required_path "$QDS_STATIC_SCORE_AGGREGATE"

  if [[ "$QDS_SPLIT_STATIC_SCORE_CACHE" == "1" ]]; then
    if [[ ! -f "$QDS_STATIC_SCORE_SHARDS/_SUCCESS.json" ]]; then
      python "$project_root/tools/split_drivesuprim_static_scores.py" \
        --input "$QDS_STATIC_SCORE_AGGREGATE" \
        --output-root "$QDS_STATIC_SCORE_SHARDS" \
        --split "$QDS_SPLIT" \
        --vocab-size 8192
    fi
    export SUPRIM_STATIC_SCORE_CACHE="$QDS_STATIC_SCORE_SHARDS"
  else
    export SUPRIM_STATIC_SCORE_CACHE="$QDS_STATIC_SCORE_AGGREGATE"
  fi
fi

metric_metadata_count=0
if [[ -d "$NAVSIM_METRIC_CACHE_ROOT/metadata" ]]; then
  metric_metadata_count="$(find "$NAVSIM_METRIC_CACHE_ROOT/metadata" -maxdepth 1 -type f -name '*.csv' | wc -l)"
fi
if (( metric_metadata_count == 0 )); then
  if [[ "$QDS_BUILD_METRIC_CACHE" == "0" ]]; then
    echo "[qds] NAVSIM metric cache is absent: $NAVSIM_METRIC_CACHE_ROOT" >&2
    exit 2
  fi
  for path in "$QDS_NAVSIM_LOG_PATH" "$QDS_NAVSIM_SENSOR_PATH" "$NUPLAN_MAPS_ROOT"; do
    required_path "$path"
  done
  echo "[qds] generating the official NAVSIM $QDS_SPLIT metric cache"
  python "$project_root/navsim/navsim/planning/script/run_metric_caching.py" \
    train_test_split=navtrain \
    navsim_log_path="$QDS_NAVSIM_LOG_PATH" \
    original_sensor_path="$QDS_NAVSIM_SENSOR_PATH" \
    metric_cache_path="$NAVSIM_METRIC_CACHE_ROOT" \
    worker=ray_distributed_no_torch \
    worker.threads_per_node="$QDS_METRIC_CACHE_WORKERS" \
    worker.use_distributed=false \
    gpu=false
fi

export TRITON_CACHE_DIR="${TRITON_LOCAL_CACHE_ROOT:-/tmp/qds-triton}/$QDS_RUN_ID"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_LOCAL_ROOT:-/tmp/qds-extensions}/$QDS_RUN_ID"
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

launcher_log="$VLA_OUTPUT_ROOT/launcher_logs/${QDS_RUN_ID}.log"
exec > >(tee -a "$launcher_log") 2>&1
echo "[qds] project_root=$project_root"
echo "[qds] run_id=$QDS_RUN_ID devices=$CUDA_VISIBLE_DEVICES processes=$QDS_LOCAL_PROCESSES"
echo "[qds] batch=micro:$VLA_BATCH_SIZE accumulation:$VLA_GRAD_ACCUM_STEPS effective:$QDS_TARGET_EFFECTIVE_BATCH"
echo "[qds] vocab=$SUPRIM_VOCAB_PATH static_scores=$SUPRIM_STATIC_SCORE_CACHE"
echo "[qds] metric_cache=$NAVSIM_METRIC_CACHE_ROOT output=$VLA_OUTPUT_ROOT"
echo "[qds] resume=$VLA_RESUME_CKPT log=$launcher_log"

launch=(
  accelerate launch
  --config_file "$QDS_DEEPSPEED_CONFIG"
  --num_machines 1
  --num_processes "$QDS_LOCAL_PROCESSES"
  --main_process_port "$QDS_MAIN_PROCESS_PORT"
  "$project_root/starVLA/training/train_starvla.py"
  --config_yaml "$QDS_CONFIG_YAML"
  --run_id "$QDS_RUN_ID"
  --run_root_dir "$VLA_OUTPUT_ROOT"
  --datasets.vla_data.datalist_path "$NAVSIM_DATALIST_PATH"
  --datasets.vla_data.data_root "$DATA_ROOT"
  --datasets.vla_data.split "$QDS_SPLIT"
  --datasets.vla_data.per_device_batch_size "$VLA_BATCH_SIZE"
  --framework.static_score_store.split "$QDS_SPLIT"
  --trainer.gradient_accumulation_steps "$VLA_GRAD_ACCUM_STEPS"
  --trainer.max_train_steps "$VLA_MAX_TRAIN_STEPS"
  --trainer.num_warmup_steps "$VLA_WARMUP_STEPS"
  --trainer.save_interval "$VLA_SAVE_INTERVAL"
  --trainer.eval_interval "$VLA_EVAL_INTERVAL"
  --trainer.logging_frequency "$VLA_LOG_INTERVAL"
  --trainer.resume_ckpt "$VLA_RESUME_CKPT"
)
"${launch[@]}"

echo "[qds] joint training complete: $VLA_OUTPUT_ROOT/$QDS_RUN_ID"
