#!/usr/bin/env bash
set -euo pipefail

audit_root="${1:-/mnt/project/DriveVLA-M0-stage2/runs/lora_value_audit/drivor25_online_full_20260902_v1}"
repo_root=/mnt/project/DriveVLA-M0-lora-value-audit-20260902
python_bin=/mnt/project/DriveVLA-M0-env/bin/python
reference=/mnt/project/DriveVLA-M0-stage2/runs/drivor_native_proposals/original25_navtest_fp32_v1_4shard/proposal_predictions.pkl
matrix=/mnt/project/DriveVLA-M0-stage2/runs/drivor_native_proposals/original25_navtest_fp32_scored_v1/candidate_scores.npz
mkdir -p "$audit_root/logs"
# Avoid the worktree's own ``navsim`` package taking precedence over the
# released DrivoR checkout through Python's empty-path entry.
cd /mnt/project

pids=()
for shard in 0 1 2 3; do
  log_file="$audit_root/logs/shard_${shard}.log"
  env \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    CUDA_VISIBLE_DEVICES="$shard" \
    PYTHONPATH=/mnt/project/external/DrivoR:/mnt/project/external/DrivoR/nuplan-devkit:"$repo_root" \
    "$python_bin" -m tools.lora_value_audit.dump_candidates \
      --reference-proposals "$reference" \
      --candidate-matrix "$matrix" \
      --drivor-repo /mnt/project/external/DrivoR \
      --checkpoint /mnt/project/external/DrivoR/weights/releases/drivor_Nav1_25epochs.pth \
      --config /mnt/project/external/DrivoR/navsim/planning/script/config/common/agent/drivoR.yaml \
      --dino-weights /mnt/project/external/DrivoR/weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors \
      --log-path /mnt/project/DriveDreamer-Policy/navsim_raw/navsim_logs/test \
      --sensor-root /mnt/project/DriveDreamer-Policy/navsim_raw/sensor_blobs/test \
      --output-dir "$audit_root" \
      --shard-count 4 \
      --shard-index "$shard" \
      --batch-size 4 \
      --chunk-size 128 \
      --repeat-parity-scenes 32 \
      >"$log_file" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  for log_file in "$audit_root"/logs/*.log; do
    echo "FAILED_LOG=$log_file" >&2
    tail -n 80 "$log_file" >&2
  done
  exit "$status"
fi

"$python_bin" - "$audit_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifests = [json.loads(path.read_text()) for path in sorted(root.glob("shard_*/manifest.json"))]
if len(manifests) != 4:
    raise SystemExit(f"expected 4 manifests, found {len(manifests)}")
scene_count = sum(int(value["scene_count"]) for value in manifests)
if scene_count != 12146:
    raise SystemExit(f"expected 12146 scenes, found {scene_count}")
if not all(bool(value["parity_passed"]) for value in manifests):
    raise SystemExit("at least one shard failed parity")
print(json.dumps({"scene_count": scene_count, "shards": 4, "all_parity_passed": True}, indent=2))
PY
