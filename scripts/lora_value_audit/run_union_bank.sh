#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 ]]; then
  echo "usage: run_union_bank.sh BANK_NAME BANK_PKL TRUE_MATRIX OUTPUT_DIR [controls] [gpu_offset]" >&2
  exit 2
fi
bank_name="$1"
bank_path="$2"
true_matrix="$3"
output_dir="$4"
controls="${5:-}"
gpu_offset="${6:-4}"
repo_root=/mnt/project/DriveVLA-M0-lora-value-audit-20260902
enhanced=/mnt/project/DriveVLA-M0-stage2/runs/lora_value_audit/drivor25_online_full_20260902_v1
base=/mnt/project/DriveVLA-M0-stage2/runs/drivor_native_proposals/original25_navtest_fp32_v1_4shard/proposal_predictions.pkl
base_matrix=/mnt/project/DriveVLA-M0-stage2/runs/drivor_native_proposals/original25_navtest_fp32_scored_v1/candidate_scores.npz
mkdir -p "$output_dir/logs"
cd /mnt/project

control_args=()
if [[ "$controls" == "controls" ]]; then
  control_args+=(--include-duplicate-controls)
fi
pids=()
for shard in 0 1 2 3; do
  gpu=$((shard + gpu_offset))
  env \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH=/mnt/project/external/DrivoR:/mnt/project/external/DrivoR/nuplan-devkit:"$repo_root" \
    /mnt/project/DriveVLA-M0-env/bin/python -m tools.lora_value_audit.evaluate_union_scorer \
      --enhanced-export-root "$enhanced" \
      --base-proposals "$base" \
      --base-matrix "$base_matrix" \
      --external-bank "$bank_path" \
      --external-matrix "$true_matrix" \
      --bank-name "$bank_name" \
      --drivor-repo /mnt/project/external/DrivoR \
      --checkpoint /mnt/project/external/DrivoR/weights/releases/drivor_Nav1_25epochs.pth \
      --config /mnt/project/external/DrivoR/navsim/planning/script/config/common/agent/drivoR.yaml \
      --dino-weights /mnt/project/external/DrivoR/weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors \
      --output-dir "$output_dir" \
      --shard-count 4 \
      --shard-index "$shard" \
      --batch-size 4 \
      "${control_args[@]}" \
      >"$output_dir/logs/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
if [[ "$status" -ne 0 ]]; then
  for path in "$output_dir"/logs/*.log; do
    echo "FAILED_LOG=$path" >&2
    tail -n 80 "$path" >&2
  done
  exit "$status"
fi

env \
  PYTHONPATH=/mnt/project/external/DrivoR:/mnt/project/external/DrivoR/nuplan-devkit:"$repo_root" \
  /mnt/project/DriveVLA-M0-env/bin/python -m tools.lora_value_audit.evaluate_union_scorer \
    --enhanced-export-root "$enhanced" \
    --base-proposals "$base" \
    --base-matrix "$base_matrix" \
    --external-bank "$bank_path" \
    --external-matrix "$true_matrix" \
    --bank-name "$bank_name" \
    --drivor-repo /mnt/project/external/DrivoR \
    --checkpoint /mnt/project/external/DrivoR/weights/releases/drivor_Nav1_25epochs.pth \
    --config /mnt/project/external/DrivoR/navsim/planning/script/config/common/agent/drivoR.yaml \
    --dino-weights /mnt/project/external/DrivoR/weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors \
    --output-dir "$output_dir" \
    --shard-count 4 \
    --bootstrap-replicates 10000 \
    --aggregate-only
