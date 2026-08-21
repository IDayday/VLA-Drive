#!/usr/bin/env bash
# Non-interactive, resumable DDP-DRS 2048 training pipeline for one PAI-DLC node.
#
# Pipeline:
#   NAVSIM metric cache (only when absent)
#   -> deterministic K=64 Qwen+DiT proposals
#   -> offline static/dynamic PDM labels
#   -> train_drivor
#   -> train_suprim_static
#   -> train_suprim_joint
#   -> optional joint_finetune

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"
cd "$project_root"

# Prevent an unattended DLC job from accidentally running the launcher copied
# from or invoked through another in-progress worktree (for example
# feature/add-VGGT). The expected branch can only be changed explicitly.
export DDP_DRS_EXPECTED_BRANCH="${DDP_DRS_EXPECTED_BRANCH:-feature/ddp-drs-scene-2048}"
actual_branch="$(git -C "$project_root" branch --show-current)"
if [[ "$actual_branch" != "$DDP_DRS_EXPECTED_BRANCH" ]]; then
  echo "[ddp-drs] wrong source worktree: $project_root" >&2
  echo "[ddp-drs] expected branch: $DDP_DRS_EXPECTED_BRANCH; actual branch: ${actual_branch:-DETACHED}" >&2
  exit 2
fi

required_value() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "[ddp-drs] required environment variable is empty: $name" >&2
    exit 2
  fi
}

export DDP_DRS_CONFIG_YAML="${DDP_DRS_CONFIG_YAML:-$project_root/starVLA/config/training/cfg_yaw_1226.yaml}"
export DDP_DRS_BASE_VLM="${DDP_DRS_BASE_VLM:-${BASE_VLM:-}}"
export DDP_DRS_BASE_DDP_CHECKPOINT="${DDP_DRS_BASE_DDP_CHECKPOINT:-}"
export DDP_DRS_STATIC_VOCAB="${DDP_DRS_STATIC_VOCAB:-}"
export DDP_DRS_SUPRIM_STATIC_SCORE_PATH="${DDP_DRS_SUPRIM_STATIC_SCORE_PATH:-}"
export DDP_DRS_DATALIST_PATH="${DDP_DRS_DATALIST_PATH:-${NAVSIM_DATALIST_PATH:-$project_root/train_meta.json}}"
export DDP_DRS_PROCESSED_DATA_ROOT="${DDP_DRS_PROCESSED_DATA_ROOT:-${DATA_ROOT:-$project_root/navsim_dataset}}"
export DDP_DRS_OPENSCENE_DATA_ROOT="${DDP_DRS_OPENSCENE_DATA_ROOT:-${OPENSCENE_DATA_ROOT:-$project_root/navsim_dataset_raw}}"
export DDP_DRS_NAVSIM_LOG_PATH="${DDP_DRS_NAVSIM_LOG_PATH:-$DDP_DRS_OPENSCENE_DATA_ROOT/navsim_logs/trainval}"
export DDP_DRS_NAVSIM_SENSOR_PATH="${DDP_DRS_NAVSIM_SENSOR_PATH:-${NAVSIM_TRAINVAL_SENSOR_ROOT:-$DDP_DRS_OPENSCENE_DATA_ROOT/sensor_blobs/trainval}}"
export DDP_DRS_MAPS_ROOT="${DDP_DRS_MAPS_ROOT:-${NUPLAN_MAPS_ROOT:-$DDP_DRS_OPENSCENE_DATA_ROOT/maps}}"
export DDP_DRS_METRIC_CACHE_ROOT="${DDP_DRS_METRIC_CACHE_ROOT:-${NAVSIM_EXP_ROOT:-$project_root/navsim_exp}/ddp_drs_metric_cache_navtrain}"
export DDP_DRS_CACHE_ROOT="${DDP_DRS_CACHE_ROOT:-${DRIVEDREAMER_SHARED_ROOT:-$project_root}/ddp_drs_cache/scene2048_navtrain_k64}"
export DDP_DRS_RUN_ROOT="${DDP_DRS_RUN_ROOT:-${NAVSIM_EXP_ROOT:-$project_root/navsim_exp}/ddp_drs_scene2048}"
export DDP_DRS_SPLIT="${DDP_DRS_SPLIT:-train}"
export DDP_DRS_SEED="${DDP_DRS_SEED:-3047}"
export DDP_DRS_NUM_DYNAMIC_CANDIDATES="${DDP_DRS_NUM_DYNAMIC_CANDIDATES:-64}"
export DDP_DRS_DYNAMIC_TOPK="${DDP_DRS_DYNAMIC_TOPK:-16}"
export DDP_DRS_SCORE_WORKERS="${DDP_DRS_SCORE_WORKERS:-8}"
export DDP_DRS_PIPELINE_PHASE="${DDP_DRS_PIPELINE_PHASE:-all}"
export DDP_DRS_BUILD_NAVSIM_METRIC_CACHE="${DDP_DRS_BUILD_NAVSIM_METRIC_CACHE:-auto}"
export DDP_DRS_ATTN_IMPLEMENTATION="${DDP_DRS_ATTN_IMPLEMENTATION:-flash_attention_2}"
export DDP_DRS_DEEPSPEED_CONFIG="${DDP_DRS_DEEPSPEED_CONFIG:-$project_root/starVLA/config/deepseeds/deepspeed_zero2.yaml}"
export DDP_DRS_TARGET_EFFECTIVE_BATCH="${DDP_DRS_TARGET_EFFECTIVE_BATCH:-64}"
export DDP_DRS_PER_DEVICE_BATCH="${DDP_DRS_PER_DEVICE_BATCH:-1}"
export DDP_DRS_DRIVOR_STEPS="${DDP_DRS_DRIVOR_STEPS:-100000}"
export DDP_DRS_SUPRIM_STATIC_STEPS="${DDP_DRS_SUPRIM_STATIC_STEPS:-100000}"
export DDP_DRS_SUPRIM_JOINT_STEPS="${DDP_DRS_SUPRIM_JOINT_STEPS:-100000}"
export DDP_DRS_JOINT_FINETUNE_STEPS="${DDP_DRS_JOINT_FINETUNE_STEPS:-0}"
export DDP_DRS_DRIVOR_LR="${DDP_DRS_DRIVOR_LR:-5e-4}"
export DDP_DRS_SUPRIM_LR="${DDP_DRS_SUPRIM_LR:-1e-4}"
export DDP_DRS_JOINT_FINETUNE_LR="${DDP_DRS_JOINT_FINETUNE_LR:-5e-5}"
export DDP_DRS_SAVE_INTERVAL="${DDP_DRS_SAVE_INTERVAL:-10000}"
export DDP_DRS_LOG_INTERVAL="${DDP_DRS_LOG_INTERVAL:-100}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-ddp-drs-scene2048}"
export OPENSCENE_DATA_ROOT="$DDP_DRS_OPENSCENE_DATA_ROOT"
export NUPLAN_MAPS_ROOT="$DDP_DRS_MAPS_ROOT"
export DATA_ROOT="$DDP_DRS_PROCESSED_DATA_ROOT"
export NAVSIM_EXP_ROOT="$DDP_DRS_RUN_ROOT"
export PYTHONPATH="$project_root:$project_root/navsim:${PYTHONPATH:-}"

# DDP-DRS requires the complete Qwen sequence, so feature caches that replace
# image/Qwen execution are intentionally disabled for this pipeline.
export NAVSIM_FEATURE_CACHE_ROOT=""
export NAVSIM_USE_FEATURE_CACHE=0
export NAVSIM_NUM_WORKERS="${NAVSIM_NUM_WORKERS:-4}"
export NAVSIM_PREFETCH_FACTOR="${NAVSIM_PREFETCH_FACTOR:-2}"
export NAVSIM_PIN_MEMORY="${NAVSIM_PIN_MEMORY:-1}"
export NAVSIM_WORKER_THREADS="${NAVSIM_WORKER_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false

for name in \
  DDP_DRS_BASE_VLM \
  DDP_DRS_BASE_DDP_CHECKPOINT \
  DDP_DRS_STATIC_VOCAB \
  DDP_DRS_DATALIST_PATH \
  DDP_DRS_PROCESSED_DATA_ROOT \
  DDP_DRS_OPENSCENE_DATA_ROOT \
  DDP_DRS_METRIC_CACHE_ROOT \
  DDP_DRS_CACHE_ROOT \
  DDP_DRS_RUN_ROOT; do
  required_value "$name"
done

integer_variables=(
  DDP_DRS_SEED
  DDP_DRS_NUM_DYNAMIC_CANDIDATES
  DDP_DRS_DYNAMIC_TOPK
  DDP_DRS_SCORE_WORKERS
  DDP_DRS_TARGET_EFFECTIVE_BATCH
  DDP_DRS_PER_DEVICE_BATCH
  DDP_DRS_DRIVOR_STEPS
  DDP_DRS_SUPRIM_STATIC_STEPS
  DDP_DRS_SUPRIM_JOINT_STEPS
  DDP_DRS_JOINT_FINETUNE_STEPS
  DDP_DRS_SAVE_INTERVAL
  DDP_DRS_LOG_INTERVAL
  NAVSIM_NUM_WORKERS
)
for name in "${integer_variables[@]}"; do
  if ! [[ "${!name}" =~ ^[0-9]+$ ]]; then
    echo "[ddp-drs] $name must be a non-negative integer, got ${!name}" >&2
    exit 2
  fi
done
if (( DDP_DRS_NUM_DYNAMIC_CANDIDATES != 64 || DDP_DRS_DYNAMIC_TOPK != 16 )); then
  echo "[ddp-drs] fidelity run requires 64 dynamic candidates and dynamic Top-16" >&2
  exit 2
fi
if (( DDP_DRS_SCORE_WORKERS < 1 || DDP_DRS_PER_DEVICE_BATCH < 1 )); then
  echo "[ddp-drs] score workers and per-device batch must be positive" >&2
  exit 2
fi
if (( DDP_DRS_DRIVOR_STEPS < 1 || DDP_DRS_SUPRIM_STATIC_STEPS < 1 || DDP_DRS_SUPRIM_JOINT_STEPS < 1 )); then
  echo "[ddp-drs] the three main training stages require at least one step" >&2
  exit 2
fi
case "$DDP_DRS_PIPELINE_PHASE" in
  all|cache|train) ;;
  *)
    echo "[ddp-drs] DDP_DRS_PIPELINE_PHASE must be all, cache, or train" >&2
    exit 2
    ;;
esac

required_paths=(
  "$DDP_DRS_CONFIG_YAML"
  "$DDP_DRS_BASE_VLM/config.json"
  "$DDP_DRS_BASE_DDP_CHECKPOINT"
  "$DDP_DRS_STATIC_VOCAB"
  "$DDP_DRS_DATALIST_PATH"
  "$DDP_DRS_PROCESSED_DATA_ROOT/meta/$DDP_DRS_SPLIT"
  "$DDP_DRS_DEEPSPEED_CONFIG"
)
if [[ -n "$DDP_DRS_SUPRIM_STATIC_SCORE_PATH" ]]; then
  required_paths+=("$DDP_DRS_SUPRIM_STATIC_SCORE_PATH")
fi
for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "[ddp-drs] missing required path: $path" >&2
    exit 2
  fi
done

actual_local_devices="$(python -c 'import torch; print(torch.cuda.device_count())')"
export DDP_DRS_LOCAL_PROCESSES="${DDP_DRS_LOCAL_PROCESSES:-${NPROC_PER_NODE:-$actual_local_devices}}"
if ! [[ "$DDP_DRS_LOCAL_PROCESSES" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ddp-drs] DDP_DRS_LOCAL_PROCESSES must be positive" >&2
  exit 2
fi
if (( DDP_DRS_LOCAL_PROCESSES > actual_local_devices )); then
  echo "[ddp-drs] requested $DDP_DRS_LOCAL_PROCESSES devices, only $actual_local_devices visible" >&2
  exit 2
fi
if [[ "${NUM_MACHINES:-1}" != "1" ]]; then
  echo "[ddp-drs] the cache/train pipeline is a one-node DLC job; NUM_MACHINES must be 1" >&2
  exit 2
fi
if (( DDP_DRS_TARGET_EFFECTIVE_BATCH % (DDP_DRS_LOCAL_PROCESSES * DDP_DRS_PER_DEVICE_BATCH) != 0 )); then
  echo "[ddp-drs] effective batch is not divisible by devices x micro-batch" >&2
  exit 2
fi
export DDP_DRS_GRADIENT_ACCUMULATION=$((
  DDP_DRS_TARGET_EFFECTIVE_BATCH
  / (DDP_DRS_LOCAL_PROCESSES * DDP_DRS_PER_DEVICE_BATCH)
))
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((DDP_DRS_LOCAL_PROCESSES - 1)))"
  export CUDA_VISIBLE_DEVICES
fi

mkdir -p "$DDP_DRS_CACHE_ROOT" "$DDP_DRS_RUN_ROOT/launcher_logs"
timestamp="$(date +'%Y%m%d_%H%M%S')"
pipeline_id="${DDP_DRS_PIPELINE_ID:-ddp-drs-scene2048-${timestamp}}"
launcher_log="$DDP_DRS_RUN_ROOT/launcher_logs/${pipeline_id}.log"
exec > >(tee -a "$launcher_log") 2>&1

echo "[ddp-drs] root=$project_root"
echo "[ddp-drs] phase=$DDP_DRS_PIPELINE_PHASE devices=$CUDA_VISIBLE_DEVICES processes=$DDP_DRS_LOCAL_PROCESSES"
echo "[ddp-drs] batch=micro:${DDP_DRS_PER_DEVICE_BATCH} accumulation:${DDP_DRS_GRADIENT_ACCUMULATION} effective:${DDP_DRS_TARGET_EFFECTIVE_BATCH}"
echo "[ddp-drs] cache=$DDP_DRS_CACHE_ROOT metric_cache=$DDP_DRS_METRIC_CACHE_ROOT"
echo "[ddp-drs] runs=$DDP_DRS_RUN_ROOT log=$launcher_log"

ddp_checkpoint_sha="$(sha256sum "$DDP_DRS_BASE_DDP_CHECKPOINT" | awk '{print $1}')"
cache_tool="$project_root/tools/generate_ddp_drs_training_cache.py"
static_score_args=()
if [[ -n "$DDP_DRS_SUPRIM_STATIC_SCORE_PATH" ]]; then
  static_score_args+=(--static-score-path "$DDP_DRS_SUPRIM_STATIC_SCORE_PATH")
fi

if [[ "$DDP_DRS_PIPELINE_PHASE" == "all" || "$DDP_DRS_PIPELINE_PHASE" == "cache" ]]; then
  metric_metadata_count=0
  if [[ -d "$DDP_DRS_METRIC_CACHE_ROOT/metadata" ]]; then
    metric_metadata_count="$(find "$DDP_DRS_METRIC_CACHE_ROOT/metadata" -maxdepth 1 -type f -name '*.csv' | wc -l)"
  fi
  if (( metric_metadata_count == 0 )); then
    if [[ "$DDP_DRS_BUILD_NAVSIM_METRIC_CACHE" == "0" ]]; then
      echo "[ddp-drs] NAVSIM metric cache is absent and automatic generation is disabled" >&2
      exit 2
    fi
    echo "[ddp-drs] generating official NAVSIM navtrain metric cache"
    for path in "$DDP_DRS_NAVSIM_LOG_PATH" "$DDP_DRS_NAVSIM_SENSOR_PATH" "$DDP_DRS_MAPS_ROOT"; do
      if [[ ! -d "$path" ]]; then
        echo "[ddp-drs] missing NAVSIM metric-cache input directory: $path" >&2
        exit 2
      fi
    done
    python navsim/navsim/planning/script/run_metric_caching.py \
      train_test_split=navtrain \
      navsim_log_path="$DDP_DRS_NAVSIM_LOG_PATH" \
      original_sensor_path="$DDP_DRS_NAVSIM_SENSOR_PATH" \
      metric_cache_path="$DDP_DRS_METRIC_CACHE_ROOT" \
      worker=ray_distributed_no_torch \
      worker.threads_per_node="${DDP_DRS_METRIC_CACHE_THREADS:-$DDP_DRS_SCORE_WORKERS}" \
      worker.use_distributed=false \
      gpu=false
  fi

  python "$cache_tool" init \
    --cache-root "$DDP_DRS_CACHE_ROOT" \
    --datalist-path "$DDP_DRS_DATALIST_PATH" \
    --split "$DDP_DRS_SPLIT" \
    --config-yaml "$DDP_DRS_CONFIG_YAML" \
    --base-vlm "$DDP_DRS_BASE_VLM" \
    --base-ddp-checkpoint "$DDP_DRS_BASE_DDP_CHECKPOINT" \
    --ddp-checkpoint-sha "$ddp_checkpoint_sha" \
    --vocab-path "$DDP_DRS_STATIC_VOCAB" \
    "${static_score_args[@]}" \
    --seed "$DDP_DRS_SEED" \
    --num-candidates "$DDP_DRS_NUM_DYNAMIC_CANDIDATES"

  proposal_port="${DDP_DRS_PROPOSAL_PORT:-29671}"
  proposal_launch=(
    accelerate launch
    --num_machines 1
    --num_processes "$DDP_DRS_LOCAL_PROCESSES"
    --main_process_port "$proposal_port"
  )
  if (( DDP_DRS_LOCAL_PROCESSES > 1 )); then
    proposal_launch+=(--multi_gpu)
  fi
  "${proposal_launch[@]}" "$cache_tool" proposals \
      --cache-root "$DDP_DRS_CACHE_ROOT" \
      --datalist-path "$DDP_DRS_DATALIST_PATH" \
      --split "$DDP_DRS_SPLIT" \
      --config-yaml "$DDP_DRS_CONFIG_YAML" \
      --base-vlm "$DDP_DRS_BASE_VLM" \
      --base-ddp-checkpoint "$DDP_DRS_BASE_DDP_CHECKPOINT" \
      --processed-data-root "$DDP_DRS_PROCESSED_DATA_ROOT" \
      --ddp-checkpoint-sha "$ddp_checkpoint_sha" \
      --seed "$DDP_DRS_SEED" \
      --num-candidates "$DDP_DRS_NUM_DYNAMIC_CANDIDATES" \
      --attn-implementation "$DDP_DRS_ATTN_IMPLEMENTATION"

  if [[ -n "$DDP_DRS_SUPRIM_STATIC_SCORE_PATH" ]]; then
    echo "[ddp-drs] scoring dynamic candidates with fork-shared official static labels"
    score_log="$DDP_DRS_RUN_ROOT/launcher_logs/${pipeline_id}.score.log"
    if ! python "$cache_tool" score-all \
      --cache-root "$DDP_DRS_CACHE_ROOT" \
      --datalist-path "$DDP_DRS_DATALIST_PATH" \
      --split "$DDP_DRS_SPLIT" \
      --vocab-path "$DDP_DRS_STATIC_VOCAB" \
      --metric-cache-root "$DDP_DRS_METRIC_CACHE_ROOT" \
      --static-score-path "$DDP_DRS_SUPRIM_STATIC_SCORE_PATH" \
      --num-workers "$DDP_DRS_SCORE_WORKERS" \
      > "$score_log" 2>&1; then
      echo "[ddp-drs] PDM score workers failed; final log lines follow" >&2
      tail -n 200 "$score_log" >&2
      exit 1
    fi
  else
    echo "[ddp-drs] no official static labels configured; scoring static and dynamic pools"
    score_pids=()
    for ((worker_index = 0; worker_index < DDP_DRS_SCORE_WORKERS; worker_index++)); do
      python "$cache_tool" score \
        --cache-root "$DDP_DRS_CACHE_ROOT" \
        --datalist-path "$DDP_DRS_DATALIST_PATH" \
        --split "$DDP_DRS_SPLIT" \
        --vocab-path "$DDP_DRS_STATIC_VOCAB" \
        --metric-cache-root "$DDP_DRS_METRIC_CACHE_ROOT" \
        --worker-index "$worker_index" \
        --num-workers "$DDP_DRS_SCORE_WORKERS" \
        > "$DDP_DRS_RUN_ROOT/launcher_logs/${pipeline_id}.score${worker_index}.log" 2>&1 &
      score_pids+=("$!")
    done
    score_failed=0
    for score_pid in "${score_pids[@]}"; do
      if ! wait "$score_pid"; then
        score_failed=1
      fi
    done
    if (( score_failed != 0 )); then
      echo "[ddp-drs] one or more PDM score workers failed" >&2
      for ((worker_index = 0; worker_index < DDP_DRS_SCORE_WORKERS; worker_index++)); do
        tail -n 100 "$DDP_DRS_RUN_ROOT/launcher_logs/${pipeline_id}.score${worker_index}.log" >&2
      done
      exit 1
    fi
  fi
  python "$cache_tool" finalize \
    --cache-root "$DDP_DRS_CACHE_ROOT" \
    --datalist-path "$DDP_DRS_DATALIST_PATH" \
    --split "$DDP_DRS_SPLIT"
fi

manifest_path="$DDP_DRS_CACHE_ROOT/manifest.json"
completion_path="$DDP_DRS_CACHE_ROOT/_SUCCESS.json"
if [[ ! -f "$manifest_path" || ! -f "$completion_path" ]]; then
  echo "[ddp-drs] complete candidate cache is required for training" >&2
  exit 2
fi
generator_config_hash="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["generator_config_hash"])' "$manifest_path")"

run_stage() {
  local stage="$1"
  local run_id="$2"
  local max_steps="$3"
  local learning_rate="$4"
  local scene_checkpoint="$5"
  local scorer_checkpoint="$6"
  local selector_checkpoint="$7"
  local warmup_steps=$((max_steps / 20))
  if (( warmup_steps < 1 )); then warmup_steps=1; fi
  local eval_interval=$((max_steps + 1))
  local port="$8"

  command=(
    accelerate launch
    --config_file "$DDP_DRS_DEEPSPEED_CONFIG"
    --num_machines 1
    --num_processes "$DDP_DRS_LOCAL_PROCESSES"
    --main_process_port "$port"
    starVLA/training/train_starvla.py
    --config_yaml "$DDP_DRS_CONFIG_YAML"
    --run_id "$run_id"
    --run_root_dir "$DDP_DRS_RUN_ROOT"
    --wandb_entity "${WANDB_ENTITY:-local}"
    --wandb_project "$WANDB_PROJECT"
    --framework.qwenvl.base_vlm "$DDP_DRS_BASE_VLM"
    --framework.qwenvl.attn_implementation "$DDP_DRS_ATTN_IMPLEMENTATION"
    --multi_trajectory.enabled true
    --multi_trajectory.training_stage "$stage"
    --multi_trajectory.num_dynamic_candidates "$DDP_DRS_NUM_DYNAMIC_CANDIDATES"
    --multi_trajectory.deterministic_seed "$DDP_DRS_SEED"
    --multi_trajectory.drivor.dynamic_topk "$DDP_DRS_DYNAMIC_TOPK"
    --multi_trajectory.suprim.vocab_path "$DDP_DRS_STATIC_VOCAB"
    --multi_trajectory.cache.root_path "$DDP_DRS_CACHE_ROOT"
    --multi_trajectory.cache.expected_ddp_checkpoint_sha "$ddp_checkpoint_sha"
    --multi_trajectory.cache.expected_generator_config_hash "$generator_config_hash"
    --multi_trajectory.cache.require_complete true
    --multi_trajectory.strict_inference true
    --multi_trajectory.smoke_test_fallback_to_single_ddp false
    --datasets.vla_data.datalist_path "$DDP_DRS_DATALIST_PATH"
    --datasets.vla_data.data_root "$DDP_DRS_PROCESSED_DATA_ROOT"
    --datasets.vla_data.split "$DDP_DRS_SPLIT"
    --datasets.vla_data.per_device_batch_size "$DDP_DRS_PER_DEVICE_BATCH"
    --datasets.vla_data.w_neg_traj null
    --datasets.vla_data.act_norm false
    --datasets.video_data.load_2d_data 0
    --datasets.gs_data.load_3d_data 0
    --datasets.reward_data.load_reward_data 0
    --enable_image_aug 0
    --w_depth 0
    --doing_s2 0
    --vit_pre 0
    --pretrain_model_2d null
    --trainer.pretrained_checkpoint "$DDP_DRS_BASE_DDP_CHECKPOINT"
    --trainer.reload_modules null
    --trainer.resume_ckpt none
    --trainer.max_train_steps "$max_steps"
    --trainer.num_warmup_steps "$warmup_steps"
    --trainer.save_interval "$DDP_DRS_SAVE_INTERVAL"
    --trainer.eval_interval "$eval_interval"
    --trainer.logging_frequency "$DDP_DRS_LOG_INTERVAL"
    --trainer.gradient_accumulation_steps "$DDP_DRS_GRADIENT_ACCUMULATION"
    --trainer.learning_rate.base "$learning_rate"
    --trainer.scheduler_specific_kwargs.min_lr 5e-7
  )
  if [[ -n "$scene_checkpoint" ]]; then
    command+=(--multi_trajectory.scene_compressor.checkpoint_path "$scene_checkpoint")
  else
    command+=(--multi_trajectory.scene_compressor.checkpoint_path null)
  fi
  if [[ -n "$scorer_checkpoint" ]]; then
    command+=(--multi_trajectory.drivor.checkpoint_path "$scorer_checkpoint")
  else
    command+=(--multi_trajectory.drivor.checkpoint_path null)
  fi
  if [[ -n "$selector_checkpoint" ]]; then
    command+=(--multi_trajectory.suprim.checkpoint_path "$selector_checkpoint")
  else
    command+=(--multi_trajectory.suprim.checkpoint_path null)
  fi
  echo "[ddp-drs] starting stage=$stage run_id=$run_id steps=$max_steps lr=$learning_rate"
  "${command[@]}"
}

if [[ "$DDP_DRS_PIPELINE_PHASE" == "all" || "$DDP_DRS_PIPELINE_PHASE" == "train" ]]; then
  drivor_run="${DDP_DRS_DRIVOR_RUN_ID:-${pipeline_id}-drivor}"
  static_run="${DDP_DRS_SUPRIM_STATIC_RUN_ID:-${pipeline_id}-suprim-static}"
  joint_run="${DDP_DRS_SUPRIM_JOINT_RUN_ID:-${pipeline_id}-suprim-joint}"

  run_stage train_drivor "$drivor_run" "$DDP_DRS_DRIVOR_STEPS" \
    "$DDP_DRS_DRIVOR_LR" "" "" "" "${DDP_DRS_TRAIN_PORT:-29681}"
  scene_checkpoint="$DDP_DRS_RUN_ROOT/$drivor_run/final_model/ddp_drs_components/scene_compressor.pt"
  scorer_checkpoint="$DDP_DRS_RUN_ROOT/$drivor_run/final_model/ddp_drs_components/dynamic_scorer.pt"
  if [[ ! -f "$scene_checkpoint" || ! -f "$scorer_checkpoint" ]]; then
    echo "[ddp-drs] train_drivor did not export required components" >&2
    exit 1
  fi

  run_stage train_suprim_static "$static_run" "$DDP_DRS_SUPRIM_STATIC_STEPS" \
    "$DDP_DRS_SUPRIM_LR" "$scene_checkpoint" "" "" "${DDP_DRS_STATIC_PORT:-29682}"
  static_selector="$DDP_DRS_RUN_ROOT/$static_run/final_model/ddp_drs_components/suprim_selector.pt"
  if [[ ! -f "$static_selector" ]]; then
    echo "[ddp-drs] train_suprim_static did not export its selector" >&2
    exit 1
  fi

  run_stage train_suprim_joint "$joint_run" "$DDP_DRS_SUPRIM_JOINT_STEPS" \
    "$DDP_DRS_SUPRIM_LR" "$scene_checkpoint" "$scorer_checkpoint" \
    "$static_selector" "${DDP_DRS_JOINT_PORT:-29683}"
  selector_checkpoint="$DDP_DRS_RUN_ROOT/$joint_run/final_model/ddp_drs_components/suprim_selector.pt"
  if [[ ! -f "$selector_checkpoint" ]]; then
    echo "[ddp-drs] train_suprim_joint did not export its selector" >&2
    exit 1
  fi

  if (( DDP_DRS_JOINT_FINETUNE_STEPS > 0 )); then
    finetune_run="${DDP_DRS_JOINT_FINETUNE_RUN_ID:-${pipeline_id}-joint-finetune}"
    run_stage joint_finetune "$finetune_run" "$DDP_DRS_JOINT_FINETUNE_STEPS" \
      "$DDP_DRS_JOINT_FINETUNE_LR" "$scene_checkpoint" "$scorer_checkpoint" \
      "$selector_checkpoint" "${DDP_DRS_FINETUNE_PORT:-29684}"
    scene_checkpoint="$DDP_DRS_RUN_ROOT/$finetune_run/final_model/ddp_drs_components/scene_compressor.pt"
    scorer_checkpoint="$DDP_DRS_RUN_ROOT/$finetune_run/final_model/ddp_drs_components/dynamic_scorer.pt"
    selector_checkpoint="$DDP_DRS_RUN_ROOT/$finetune_run/final_model/ddp_drs_components/suprim_selector.pt"
  fi

  bundle_path="$DDP_DRS_RUN_ROOT/${pipeline_id}-inference-bundle.json"
  export DDP_DRS_BUNDLE_PATH="$bundle_path"
  export DDP_DRS_FINAL_SCENE_CHECKPOINT="$scene_checkpoint"
  export DDP_DRS_FINAL_SCORER_CHECKPOINT="$scorer_checkpoint"
  export DDP_DRS_FINAL_SELECTOR_CHECKPOINT="$selector_checkpoint"
  export DDP_DRS_FINAL_GENERATOR_HASH="$generator_config_hash"
  export DDP_DRS_FINAL_DDP_SHA="$ddp_checkpoint_sha"
  python - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "baseline": "DDP-DRS",
    "scene_dim": 2048,
    "planning_dim": 256,
    "base_ddp_checkpoint": os.environ["DDP_DRS_BASE_DDP_CHECKPOINT"],
    "base_ddp_checkpoint_sha": os.environ["DDP_DRS_FINAL_DDP_SHA"],
    "scene_compressor_checkpoint": os.environ["DDP_DRS_FINAL_SCENE_CHECKPOINT"],
    "drivor_scorer_checkpoint": os.environ["DDP_DRS_FINAL_SCORER_CHECKPOINT"],
    "suprim_selector_checkpoint": os.environ["DDP_DRS_FINAL_SELECTOR_CHECKPOINT"],
    "static_vocab": os.environ["DDP_DRS_STATIC_VOCAB"],
    "candidate_cache": os.environ["DDP_DRS_CACHE_ROOT"],
    "generator_config_hash": os.environ["DDP_DRS_FINAL_GENERATOR_HASH"],
}
path = Path(os.environ["DDP_DRS_BUNDLE_PATH"])
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
with temporary.open("w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, path)
print(f"[ddp-drs] inference bundle: {path}")
PY
fi

echo "[ddp-drs] pipeline complete"
