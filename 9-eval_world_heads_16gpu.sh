#!/usr/bin/env bash
# Run full-test DriveDreamer world-head evaluation on 16 GPUs.
#
# It computes metrics for every test sample, but saves visualizations sparsely:
# global_sample_index % SAVE_VISUAL_EVERY == 0.  The default 100 saves ~1%.
#
# Usage on a 16-card DLC node:
#   cd /mnt/workspace/WM_Group/zhangt_workspace/project/DriveDreamer-Policy
#   bash 9-eval_world_heads_16gpu.sh
#
# Common overrides:
#   NUM_GPUS=8 bash 9-eval_world_heads_16gpu.sh
#   TASKS=depth bash 9-eval_world_heads_16gpu.sh
#   TASKS=video bash 9-eval_world_heads_16gpu.sh
#   OUT_DIR=artifacts/world_head_eval_test_v2 SAVE_VISUAL_EVERY=100 bash 9-eval_world_heads_16gpu.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

NUM_GPUS="${NUM_GPUS:-16}"
SPLIT="${SPLIT:-test}"
TASKS="${TASKS:-video,depth}"
SAVE_VISUAL_EVERY="${SAVE_VISUAL_EVERY:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
QWEN_FORWARD_MODE="${QWEN_FORWARD_MODE:-optimized}"

CKPT_DIR="${CKPT_DIR:-$PROJECT_DIR/navsim_exp/navsim-v1-formal-16gpu-bz2-ga1-flashattn-20260801_073617/final_model/pytorch_model.pt}"
DATALIST_PATH="${DATALIST_PATH:-$PROJECT_DIR/${SPLIT}_meta.json}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/navsim_dataset}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/artifacts/world_head_eval_${SPLIT}}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs/world_head_eval_${SPLIT}}"

BASE_VLM="${BASE_VLM:-$PROJECT_DIR/weights/derived/Qwen3-VL-2B-WorldAction}"
OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/workspace/Public_Space/navsim}"
NAVSIM_SENSOR_BLOBS_ROOT="${NAVSIM_SENSOR_BLOBS_ROOT:-/mnt/workspace/Public_Space/navsim/test_sensor_blobs}"
NAVSIM_VIDEO_SOURCE="${NAVSIM_VIDEO_SOURCE:-images}"
DEPTH_SEMANTICS_PTH="${DEPTH_SEMANTICS_PTH:-/mnt/workspace/VLA_Group/LLM_weight/depth-anything/Depth-Anything-V2-Large/depth_anything_v2_vitl.pth}"
DEPTH_PPD_PTH="${DEPTH_PPD_PTH:-/mnt/workspace/VLA_Group/LLM_weight/gangweix/Pixel-Perfect-Depth/ppd.pth}"
WAN_MODEL_PATH="${WAN_MODEL_PATH:-/mnt/workspace/VLA_Group/LLM_weight/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP}"
VIDEO_ROOT="${VIDEO_ROOT:-$DATA_ROOT/navsim_video}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

printf '[world-eval] split=%s tasks=%s num_gpus=%s save_visual_every=%s\n' \
  "$SPLIT" "$TASKS" "$NUM_GPUS" "$SAVE_VISUAL_EVERY"
printf '[world-eval] out_dir=%s\n' "$OUT_DIR"
printf '[world-eval] log_dir=%s\n' "$LOG_DIR"

for rank in $(seq 0 $((NUM_GPUS - 1))); do
  rank_out="$OUT_DIR/rank_$(printf '%02d' "$rank")"
  log_file="$LOG_DIR/rank_$(printf '%02d' "$rank").log"
  mkdir -p "$rank_out"
  printf '[world-eval] launch rank=%s gpu=%s log=%s\n' "$rank" "$rank" "$log_file"
  (
    BASE_VLM="$BASE_VLM" \
    OPENSCENE_DATA_ROOT="$OPENSCENE_DATA_ROOT" \
    NAVSIM_SENSOR_BLOBS_ROOT="$NAVSIM_SENSOR_BLOBS_ROOT" \
    NAVSIM_VIDEO_SOURCE="$NAVSIM_VIDEO_SOURCE" \
    CUDA_VISIBLE_DEVICES="$rank" \
    python eval_world_heads.py \
      --ckpt_dir "$CKPT_DIR" \
      --datalist_path "$DATALIST_PATH" \
      --data_root "$DATA_ROOT" \
      --out_dir "$rank_out" \
      --split "$SPLIT" \
      --tasks "$TASKS" \
      --max_samples "$MAX_SAMPLES" \
      --batch_size "$BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" \
      --save_visual_every "$SAVE_VISUAL_EVERY" \
      --rank "$rank" \
      --world_size "$NUM_GPUS" \
      --qwen_forward_mode "$QWEN_FORWARD_MODE" \
      --depth_semantics_pth "$DEPTH_SEMANTICS_PTH" \
      --depth_ppd_pth "$DEPTH_PPD_PTH" \
      --wan_model_path "$WAN_MODEL_PATH" \
      --video_root "$VIDEO_ROOT"
  ) >"$log_file" 2>&1 &
done

wait
printf '[world-eval] all ranks finished, merging metrics...\n'

OUT_DIR="$OUT_DIR" SPLIT="$SPLIT" NUM_GPUS="$NUM_GPUS" TASKS="$TASKS" SAVE_VISUAL_EVERY="$SAVE_VISUAL_EVERY" python - <<'PYMERGE'
import json
import os
from pathlib import Path
from numbers import Number

out_dir = Path(os.environ["OUT_DIR"])
num_gpus = int(os.environ["NUM_GPUS"])
video_samples = []
depth_samples = []
rank_metrics = []
missing = []

for rank in range(num_gpus):
    rank_dir = out_dir / f"rank_{rank:02d}"
    metrics_path = rank_dir / "metrics.json"
    if metrics_path.is_file():
        rank_metrics.append(json.loads(metrics_path.read_text()))
    else:
        missing.append(str(metrics_path))
    video_path = rank_dir / "video_metrics_per_sample.json"
    if video_path.is_file():
        video_samples.extend(json.loads(video_path.read_text()).get("samples", []))
    depth_path = rank_dir / "depth_metrics_per_sample.json"
    if depth_path.is_file():
        depth_samples.extend(json.loads(depth_path.read_text()).get("samples", []))

if missing:
    raise SystemExit("missing rank metrics:\n" + "\n".join(missing))

def mean_dict(samples):
    keys = sorted({k for s in samples for k, v in s.items() if isinstance(v, Number)})
    return {k: sum(float(s[k]) for s in samples if k in s) / sum(1 for s in samples if k in s) for k in keys}

summary = {
    "split": os.environ["SPLIT"],
    "tasks": os.environ["TASKS"],
    "world_size": num_gpus,
    "save_visual_every": int(os.environ["SAVE_VISUAL_EVERY"]),
    "num_video_samples": len(video_samples),
    "num_depth_samples": len(depth_samples),
    "video": mean_dict(video_samples),
    "depth_normalized_log": mean_dict(depth_samples),
    "note": "Depth metrics are computed in PPD normalized log-depth latent space. Visuals are stored under rank_XX/ subdirectories for samples whose global index is divisible by save_visual_every.",
    "rank_metrics": rank_metrics,
}
(out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
if video_samples:
    (out_dir / "video_metrics_per_sample.json").write_text(json.dumps({"samples": video_samples}, indent=2, sort_keys=True))
if depth_samples:
    (out_dir / "depth_metrics_per_sample.json").write_text(json.dumps({"samples": depth_samples}, indent=2, sort_keys=True))
print(json.dumps({k: summary[k] for k in ["num_video_samples", "num_depth_samples", "video", "depth_normalized_log"]}, indent=2, sort_keys=True))
PYMERGE

printf '[world-eval] merged metrics: %s\n' "$OUT_DIR/metrics.json"
