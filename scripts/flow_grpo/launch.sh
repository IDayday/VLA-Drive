#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/navsim:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export FLASH_ATTENTION_DETERMINISTIC="${FLASH_ATTENTION_DETERMINISTIC:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/ddp-flow-grpo-triton}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/ddp/bin/python}"
COMMAND="${1:-preflight}"
shift || true
CONFIG="${CONFIG:-configs/flow_grpo/action_only_frozen_visual.yaml}"
args=(--config "$CONFIG")
if [[ -n "${SFT_CKPT:-}" ]]; then args+=(--sft-ckpt "$SFT_CKPT"); fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then args+=(--output-dir "$OUTPUT_DIR"); fi
if [[ "$COMMAND" == train ]]; then
  : "${MAX_UPDATES:?Set an explicit bounded MAX_UPDATES}"
  : "${OUTPUT_DIR:?Set a new output directory or explicit --resume}"
  args+=(--max-updates "$MAX_UPDATES")
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="${NUM_GPUS:-1}" \
    -m starVLA.rl.flow_grpo.cli train "${args[@]}" "$@"
else
  "$PYTHON_BIN" -m starVLA.rl.flow_grpo.cli "$COMMAND" "${args[@]}" "$@"
fi
