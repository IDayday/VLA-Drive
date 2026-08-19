#!/usr/bin/env bash
# Batch inference + NAVSIM v1.1 PDMS evaluation for comparable VGGT AR/query checkpoints.
#
# Defaults compare the common steps under:
#   navsim_exp/vggt_action_AR/checkpoints
#   navsim_exp/vggt_action_query/checkpoints
#
# Examples:
#   bash 9-eval_vggt_compare_pdms.sh
#   GPUS="0,1,2,3,4,5,6,7" STEPS="10000 30000 50000" bash 9-eval_vggt_compare_pdms.sh
#   GPU=1 BATCH_SIZE=4 bash 9-eval_vggt_compare_pdms.sh   # single GPU
#   OVERWRITE=1 SPLIT=test bash 9-eval_vggt_compare_pdms.sh
#   RUN_INFER=0 bash 9-eval_vggt_compare_pdms.sh   # evaluate existing predictions only

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

AR_DIR="${AR_DIR:-${DRIVEDREAMER_ROOT}/navsim_exp/vggt_action_AR}"
QUERY_DIR="${QUERY_DIR:-${DRIVEDREAMER_ROOT}/navsim_exp/vggt_action_query}"
SPLIT="${SPLIT:-test}"
OUT_BASE="${OUT_BASE:-${DRIVEDREAMER_ROOT}/navsim_planning_results/vggt_action_compare_pdms}"
EVAL_BASE="${EVAL_BASE:-${DRIVEDREAMER_ROOT}/navsim_exp/eval_v1.1_vggt_action_compare}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-7}"
GPU="${GPU:-0}"
GPUS="${GPUS:-$GPU}"
QWEN_FORWARD_MODE="${QWEN_FORWARD_MODE:-auto}"
VLM_ATTN_IMPLEMENTATION="${VLM_ATTN_IMPLEMENTATION:-sdpa}"
OVERWRITE="${OVERWRITE:-0}"
RUN_INFER="${RUN_INFER:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

# Teacher VGGT cache is a train-split target. Inference only needs model weights
# and raw test inputs, so keep evaluation independent from feature-cache presence.
unset NAVSIM_FEATURE_CACHE_ROOT
unset NAVSIM_VGGT_CACHE_ROOT
export NAVSIM_USE_FEATURE_CACHE=0
export NAVSIM_VGGT_CACHE_STRICT=0

checkpoint_steps() {
  local exp_dir="$1"
  find "${exp_dir}/checkpoints" -maxdepth 1 -type f -name 'steps_*_pytorch_model.pt' \
    | sed -E 's#.*/steps_([0-9]+)_pytorch_model\.pt#\1#' \
    | sort -n
}

if [[ -n "${STEPS:-}" ]]; then
  read -r -a steps <<< "${STEPS}"
else
  mapfile -t steps < <(comm -12 <(checkpoint_steps "$AR_DIR") <(checkpoint_steps "$QUERY_DIR"))
fi

if [[ "${#steps[@]}" -eq 0 ]]; then
  echo "No comparable checkpoint steps found." >&2
  echo "AR_DIR=${AR_DIR}" >&2
  echo "QUERY_DIR=${QUERY_DIR}" >&2
  exit 1
fi

mkdir -p "$OUT_BASE" "$EVAL_BASE"

echo "Comparing steps: ${steps[*]}"
echo "Prediction root: $OUT_BASE"
echo "Evaluation root:  $EVAL_BASE"
echo "Inference GPUs:  $GPUS"

run_one() {
  local name="$1"
  local model_dir="$2"
  local step="$3"
  local pred_dir="${OUT_BASE}/${name}-step${step}"

  if [[ ! -f "${model_dir}/checkpoints/steps_${step}_pytorch_model.pt" ]]; then
    echo "Missing checkpoint: ${model_dir}/checkpoints/steps_${step}_pytorch_model.pt" >&2
    return 1
  fi

  echo "============================================================"
  echo "[$name step $step] inference"
  echo "============================================================"
  if [[ "$RUN_INFER" == "1" ]]; then
    IFS=',' read -r -a gpu_list <<< "$GPUS"
    local world_size="${#gpu_list[@]}"
    local pids=()
    for rank in "${!gpu_list[@]}"; do
      local gpu="${gpu_list[$rank]}"
      echo "[$name step $step] launch rank ${rank}/${world_size} on GPU ${gpu}"
      MODEL_DIR="$model_dir" \
      MODEL_ITER="$step" \
      OUT_DIR="$OUT_BASE" \
      SPLIT="$SPLIT" \
      BATCH_SIZE="$BATCH_SIZE" \
      NUM_WORKERS="$NUM_WORKERS" \
      GPU="$gpu" \
      RANK="$rank" \
      WORLD_SIZE="$world_size" \
      QWEN_FORWARD_MODE="$QWEN_FORWARD_MODE" \
      VLM_ATTN_IMPLEMENTATION="$VLM_ATTN_IMPLEMENTATION" \
      INFER_VLM_ATTN_IMPLEMENTATION="$VLM_ATTN_IMPLEMENTATION" \
      OVERWRITE="$OVERWRITE" \
      INFER_USE_FEATURE_CACHE=0 \
        bash "$project_root/4-infer.sh" &
      pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        failed=1
      fi
    done
    if [[ "$failed" != "0" ]]; then
      echo "[$name step $step] one or more inference shards failed" >&2
      return 1
    fi
  fi

  echo "============================================================"
  echo "[$name step $step] PDMS eval"
  echo "============================================================"
  if [[ "$RUN_EVAL" == "1" ]]; then
    PRED_DIR="$pred_dir" \
    SPLIT="$SPLIT" \
    NAVSIM_EVAL_ROOT="${EVAL_BASE}/${name}-step${step}" \
      bash "$project_root/5-eval_v1.sh"
  fi
}

for step in "${steps[@]}"; do
  run_one "vggt_action_AR" "$AR_DIR" "$step"
  run_one "vggt_action_query" "$QUERY_DIR" "$step"
done

echo "Done. Predictions are under: $OUT_BASE"
echo "PDMS outputs are under: $EVAL_BASE"
