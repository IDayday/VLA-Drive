#!/usr/bin/env bash

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

unset NAVSIM_AGENT_DINO_CACHE_ROOT
unset NAVSIM_VGGT_CACHE_ROOT
unset NAVSIM_FEATURE_CACHE_ROOT
export NAVSIM_USE_FEATURE_CACHE=0

fusion_mode="${SQ3DMIX_FUSION_MODE:-gated}"
intervention="${SQ3DMIX_INTERVENTION:-real}"
scene_query_lr="${SCENE_QUERY_LEARNING_RATE:-3e-5}"
gated_fusion_lr="${GATED_FUSION_LEARNING_RATE:-1e-4}"
smoke="${SQ3DMIX_SMOKE:-0}"
dry_run="${SQ3DMIX_DRY_RUN:-${TRAINING_TOPOLOGY_ONLY:-0}}"

for pair in "SQ3DMIX_SMOKE:$smoke" "SQ3DMIX_DRY_RUN:$dry_run"; do
  variable="${pair%%:*}"
  value="${pair#*:}"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "$variable must be 0 or 1, got: $value" >&2
    exit 2
  fi
done

effective_base_vlm="$BASE_VLM"
effective_cache_root="$NAVSIM_VGGT_DENSE_CACHE_ROOT"
effective_action_checkpoint="${ACTION_ONLY_CHECKPOINT:-}"
effective_run_root="$NAVSIM_EXP_ROOT"
effective_run_id="${RUN_ID:-}"
effective_cache_enabled=""
run_root_cli=0
run_id_cli=0
cli_arguments=("$@")
for ((argument_index = 0; argument_index < ${#cli_arguments[@]}; argument_index++)); do
  argument="${cli_arguments[$argument_index]}"
  case "$argument" in
    --framework.qwenvl.base_vlm)
      ((argument_index += 1))
      effective_base_vlm="${cli_arguments[$argument_index]:-}"
      ;;
    --framework.qwenvl.base_vlm=*) effective_base_vlm="${argument#*=}" ;;
    --framework.sq_3d_mix.fusion_mode)
      ((argument_index += 1))
      fusion_mode="${cli_arguments[$argument_index]:-}"
      ;;
    --framework.sq_3d_mix.fusion_mode=*) fusion_mode="${argument#*=}" ;;
    --framework.sq_3d_mix.intervention.mode)
      ((argument_index += 1))
      intervention="${cli_arguments[$argument_index]:-}"
      ;;
    --framework.sq_3d_mix.intervention.mode=*) intervention="${argument#*=}" ;;
    --framework.sq_3d_mix.cache.enabled)
      ((argument_index += 1))
      effective_cache_enabled="${cli_arguments[$argument_index]:-}"
      ;;
    --framework.sq_3d_mix.cache.enabled=*)
      effective_cache_enabled="${argument#*=}"
      ;;
    --framework.sq_3d_mix.cache.root)
      ((argument_index += 1))
      effective_cache_root="${cli_arguments[$argument_index]:-}"
      ;;
    --framework.sq_3d_mix.cache.root=*) effective_cache_root="${argument#*=}" ;;
    --trainer.pretrained_checkpoint)
      ((argument_index += 1))
      effective_action_checkpoint="${cli_arguments[$argument_index]:-}"
      ;;
    --trainer.pretrained_checkpoint=*) effective_action_checkpoint="${argument#*=}" ;;
    --run_root_dir)
      ((argument_index += 1))
      effective_run_root="${cli_arguments[$argument_index]:-}"
      run_root_cli=1
      ;;
    --run_root_dir=*)
      effective_run_root="${argument#*=}"
      run_root_cli=1
      ;;
    --run_id)
      ((argument_index += 1))
      effective_run_id="${cli_arguments[$argument_index]:-}"
      run_id_cli=1
      ;;
    --run_id=*)
      effective_run_id="${argument#*=}"
      run_id_cli=1
      ;;
  esac
done

case "$fusion_mode" in
  scene_only|projected_concat|gated) ;;
  *)
    echo "Effective SQ-3D-Mix fusion mode is invalid: $fusion_mode" >&2
    exit 2
    ;;
esac
case "$intervention" in
  real|zero|gaussian|shuffled) ;;
  *)
    echo "Effective SQ-3D-Mix intervention is invalid: $intervention" >&2
    exit 2
    ;;
esac

if [[ "$fusion_mode" == "scene_only" ]]; then
  required_cache_enabled="false"
else
  required_cache_enabled="true"
fi
if [[ -n "$effective_cache_enabled" ]]; then
  case "${effective_cache_enabled,,}" in
    true|1) effective_cache_enabled="true" ;;
    false|0) effective_cache_enabled="false" ;;
    *)
      echo "framework.sq_3d_mix.cache.enabled must be true or false" >&2
      exit 2
      ;;
  esac
  if [[ "$effective_cache_enabled" != "$required_cache_enabled" ]]; then
    echo "$fusion_mode requires cache.enabled=$required_cache_enabled" >&2
    exit 2
  fi
fi

if [[ ! -d "$effective_base_vlm" ]]; then
  echo "Missing standard action VLM: $effective_base_vlm" >&2
  exit 2
fi
if [[ "$fusion_mode" != "scene_only" ]]; then
  manifest="$effective_cache_root/vggt_dense/manifest.json"
  if [[ ! -f "$manifest" ]]; then
    echo "Missing dense VGGT cache manifest: $manifest" >&2
    echo "Run 11-precompute_vggt_dense_cache.sh first." >&2
    exit 2
  fi
fi
if [[ -n "$effective_action_checkpoint" && ! -f "$effective_action_checkpoint" ]]; then
  echo "Missing action-only initialization checkpoint: $effective_action_checkpoint" >&2
  exit 2
fi

if [[ "$smoke" == "1" ]]; then
  num_machines="${NUM_MACHINES:-1}"
  machine_rank="${MACHINE_RANK:-0}"
  local_num_processes="${LOCAL_NUM_PROCESSES:-1}"
  num_processes="${NUM_PROCESSES:-$((num_machines * local_num_processes))}"
  batch_size="${PER_DEVICE_BATCH_SIZE:-1}"
  gradient_accumulation="${GRADIENT_ACCUMULATION_STEPS:-1}"
  target_effective_batch="${TARGET_EFFECTIVE_BATCH_SIZE:-$((num_processes * batch_size * gradient_accumulation))}"
  max_train_steps="${MAX_TRAIN_STEPS:-2}"
  fm_repeat="${FM_REPEAT:-1}"
  export NAVSIM_NUM_WORKERS="${NAVSIM_NUM_WORKERS:-0}"
  export NAVSIM_PIN_MEMORY="${NAVSIM_PIN_MEMORY:-0}"
  if [[ "$run_root_cli" == "0" ]]; then
    effective_run_root="${SQ3DMIX_SMOKE_ROOT:-${TMPDIR:-/tmp}/sq3dmix-smoke}"
  fi
  if [[ "$run_id_cli" == "0" && -z "$effective_run_id" ]]; then
    effective_run_id="sq3dmix-${fusion_mode}-smoke-$$"
  fi
else
  num_machines="${NUM_MACHINES:-${WORLD_SIZE:-1}}"
  machine_rank="${MACHINE_RANK:-${RANK:-0}}"
  local_num_processes="${LOCAL_NUM_PROCESSES:-${NPROC_PER_NODE:-16}}"
  num_processes="${NUM_PROCESSES:-$((num_machines * local_num_processes))}"
  batch_size="${PER_DEVICE_BATCH_SIZE:-2}"
  gradient_accumulation="${GRADIENT_ACCUMULATION_STEPS:-1}"
  target_effective_batch="${TARGET_EFFECTIVE_BATCH_SIZE:-32}"
  max_train_steps="${MAX_TRAIN_STEPS:-100000}"
  fm_repeat="${FM_REPEAT:-8}"
  timestamp="$(date +'%Y%m%d_%H%M%S')"
  effective_run_id="${effective_run_id:-sq3dmix-${fusion_mode}-${PAI_JOB_ID:-$timestamp}}"
fi

for pair in \
  "NUM_MACHINES:$num_machines" \
  "LOCAL_NUM_PROCESSES:$local_num_processes" \
  "NUM_PROCESSES:$num_processes" \
  "PER_DEVICE_BATCH_SIZE:$batch_size" \
  "GRADIENT_ACCUMULATION_STEPS:$gradient_accumulation" \
  "MAX_TRAIN_STEPS:$max_train_steps" \
  "FM_REPEAT:$fm_repeat"; do
  variable="${pair%%:*}"
  value="${pair#*:}"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$variable must be a positive integer, got: $value" >&2
    exit 2
  fi
done
if (( num_processes != num_machines * local_num_processes )); then
  echo "Invalid topology: processes $num_processes != nodes $num_machines x local $local_num_processes" >&2
  exit 2
fi
effective_batch=$((num_processes * batch_size * gradient_accumulation))
if (( effective_batch != target_effective_batch )); then
  echo "Refusing effective batch $effective_batch; expected $target_effective_batch" >&2
  exit 2
fi

split="${SPLIT:-train}"
datalist="${NAVSIM_DATALIST_PATH:-$DRIVEDREAMER_ROOT/${split}_meta.json}"
if [[ ! -f "$datalist" ]]; then
  echo "Missing NAVSIM datalist: $datalist" >&2
  exit 2
fi
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Missing processed NAVSIM data root: $DATA_ROOT" >&2
  exit 2
fi

base_config="${TRAIN_CONFIG_YAML:-$DRIVEDREAMER_ROOT/starVLA/config/training/cfg_yaw_1225.yaml}"
overlay="$DRIVEDREAMER_ROOT/starVLA/config/training/sq_3d_mix.yaml"
accelerate_config="${TRAIN_ACCELERATE_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}"
visible_devices="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "$visible_devices" ]]; then
  for ((device_index = 0; device_index < local_num_processes; device_index++)); do
    [[ -n "$visible_devices" ]] && visible_devices+=","
    visible_devices+="$device_index"
  done
fi

launch_args=(
  --main_process_port "${MAIN_PROCESS_PORT:-${MASTER_PORT:-29694}}"
  --config_file "$accelerate_config"
  --num_processes "$num_processes"
  --num_machines "$num_machines"
  --machine_rank "$machine_rank"
  --mixed_precision bf16
)
if (( num_machines > 1 )); then
  launch_args+=(
    --main_process_ip "${MAIN_PROCESS_IP:-${MASTER_ADDR:?DLC must provide MASTER_ADDR}}"
    --same_network
  )
fi

training_args=(
  --config_yaml "$base_config"
  --config_overlay "$overlay"
  --framework.qwenvl.base_vlm "$effective_base_vlm"
  --framework.qwenvl.attn_implementation "${VLM_ATTN_IMPLEMENTATION:-flash_attention_2}"
  --framework.sq_3d_mix.fusion_mode "$fusion_mode"
  --framework.sq_3d_mix.intervention.mode "$intervention"
  --framework.sq_3d_mix.cache.enabled "$required_cache_enabled"
  --run_root_dir "$effective_run_root"
  --run_id "$effective_run_id"
  --wandb_project "$WANDB_PROJECT"
  --wandb_entity "$WANDB_ENTITY"
  --datasets.vla_data.datalist_path "$datalist"
  --datasets.vla_data.data_root "$DATA_ROOT"
  --datasets.vla_data.split "$split"
  --datasets.vla_data.per_device_batch_size "$batch_size"
  --datasets.vla_data.load_act_data 1
  --datasets.video_data.load_2d_data 0
  --datasets.gs_data.load_3d_data 0
  --datasets.reward_data.load_reward_data 0
  --w_depth 0
  --rgb_query_loss 0
  --gs_query_loss 0
  --trainer.gradient_accumulation_steps "$gradient_accumulation"
  --trainer.max_train_steps "$max_train_steps"
  --trainer.optimizer.weight_decay "${OPTIMIZER_WEIGHT_DECAY:-1e-3}"
  --trainer.learning_rate.base "${BASE_LEARNING_RATE:-1e-5}"
  --trainer.learning_rate.action_model "${ACTION_LEARNING_RATE:-1e-5}"
  --trainer.learning_rate.scene_query_compressor "$scene_query_lr"
  --trainer.learning_rate.gated_fusion "$gated_fusion_lr"
  --framework.action_model.repeated_diffusion_steps "$fm_repeat"
  --framework.action_model.hidden_size "${ACTION_HIDDEN_SIZE:-1536}"
  --framework.action_model.diffusion_model_cfg.cross_attention_dim "${ACTION_HIDDEN_SIZE:-1536}"
  --framework.action_model.diffusion_model_cfg.output_dim "${ACTION_HIDDEN_SIZE:-1536}"
  --framework.action_model.diffusion_model_cfg.num_layers "${ACTION_LAYERS:-24}"
)
if [[ "$fusion_mode" != "scene_only" ]]; then
  training_args+=(--framework.sq_3d_mix.cache.root "$effective_cache_root")
fi
if [[ -n "$effective_action_checkpoint" ]]; then
  training_args+=(--trainer.pretrained_checkpoint "$effective_action_checkpoint")
fi
training_args+=("$@")

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TRITON_LOCAL_CACHE_ROOT:-/tmp/drivedreamer-triton}/${effective_run_id}/node${machine_rank}}"
echo "SQ-3D-Mix mode: fusion=$fusion_mode intervention=$intervention smoke=$smoke"
echo "SQ-3D-Mix topology: nodes=$num_machines node_rank=$machine_rank local_devices=$local_num_processes global_processes=$num_processes"
echo "SQ-3D-Mix effective batch: $effective_batch (target=$target_effective_batch)"
echo "SQ-3D-Mix optimization: max_steps=$max_train_steps fm_repeat=$fm_repeat scene_lr=$scene_query_lr fusion_lr=$gated_fusion_lr"

if [[ "$dry_run" == "1" ]]; then
  printf '%q ' CUDA_VISIBLE_DEVICES="$visible_devices" accelerate launch "${launch_args[@]}" starVLA/training/train_starvla.py "${training_args[@]}"
  printf '\n'
  exit 0
fi

run_dir="$effective_run_root/$effective_run_id"
if [[ -e "$run_dir" ]]; then
  echo "Refusing to overwrite existing SQ-3D-Mix experiment: $run_dir" >&2
  exit 2
fi
mkdir -p "$TRITON_CACHE_DIR"

set -x
CUDA_VISIBLE_DEVICES="$visible_devices" accelerate launch \
  "${launch_args[@]}" \
  starVLA/training/train_starvla.py \
  "${training_args[@]}"
