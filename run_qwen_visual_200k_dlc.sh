#!/usr/bin/env bash
# One-command from-scratch trainable-visual action-only run for PAI-DLC.
# Fixed contract: 1 node x 16 PPU, 200K optimizer steps, no training smoke.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

usage() {
  echo "Usage: bash $project_root/run_qwen_visual_200k_dlc.sh"
}

if (( $# != 0 )); then
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi
  usage >&2
  exit 2
fi

timestamp="$(date +'%Y%m%d_%H%M%S')"

export NUM_MACHINES=1
export MACHINE_RANK=0
export LOCAL_NUM_PROCESSES=16
export NUM_PROCESSES=16
export QWEN_VISUAL_EXPECTED_PPU_COUNT=16
export PER_DEVICE_BATCH_SIZE=2
export GRADIENT_ACCUMULATION_STEPS=1
export TARGET_EFFECTIVE_BATCH_SIZE=32

export QWEN_VISUAL_RESUME_CKPT=none
export QWEN_VISUAL_INITIAL_STEP=0
export QWEN_VISUAL_RESUME_STRICT=0
export MAX_TRAIN_STEPS=200000
export QWEN_VISUAL_LR_DECAY_STEPS=100000
export NUM_WARMUP_STEPS=5000
export QWEN_LEARNING_RATE=1e-5
export VISUAL_LEARNING_RATE=2e-6
export ACTION_LEARNING_RATE=1e-5
export QWEN_VISUAL_CONSTANT_LEARNING_RATE=0

export EXPECTED_TRAIN_SAMPLES=103288
export TRAIN_CONFIG_YAML="$project_root/starVLA/config/training/cfg_yaw_1225.yaml"
export SAVE_INTERVAL=10000
export TRAINING_LOGGING_FREQUENCY=50
export OPTIMIZER_WEIGHT_DECAY=1e-3
export FM_REPEAT=8
export ACTION_HIDDEN_SIZE=1536
export ACTION_LAYERS=24
export QWEN_VISUAL_ATTN_IMPLEMENTATION=sdpa
export QWEN_VISUAL_RUN_SMOKE_BEFORE_FORMAL=0
export QWEN_VISUAL_TUNE_DRY_RUN=0
export QWEN_VISUAL_DLC_PREFLIGHT="${QWEN_VISUAL_DLC_PREFLIGHT:-1}"
export QWEN_VISUAL_DLC_PREFLIGHT_ONLY="${QWEN_VISUAL_DLC_PREFLIGHT_ONLY:-0}"
export QWEN_VISUAL_DLC_DRY_RUN="${QWEN_VISUAL_DLC_DRY_RUN:-0}"
export RUN_ID="qwen-visual-action-only-from-scratch-200k-${PAI_JOB_ID:-dlc}-${timestamp}"

echo "[qwen-visual-200k-dlc] run_id=$RUN_ID"
echo "[qwen-visual-200k-dlc] mode=from-scratch checkpoint=none optimizer=fresh"
echo "[qwen-visual-200k-dlc] topology=1x16 effective_batch=32 training_smoke=disabled"
echo "[qwen-visual-200k-dlc] global_steps=0->200000 lr_schedule=cosine_100k_then_hold"
echo "[qwen-visual-200k-dlc] lr=qwen:1e-5 visual:2e-6 action:1e-5 endpoint=qwen:5e-7 visual:1e-7 action:5e-7"

exec bash "$project_root/8-train_action-only-qwen-visual.sh"
