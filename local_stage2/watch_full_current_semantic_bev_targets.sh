#!/usr/bin/env bash

# Aggregate and validate all semantic-BEV shards after multi-host generation.

set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
python_bin="${DRIVEVLA_PYTHON:-/mnt/project/DriveVLA-M0-env/bin/python}"
feature_root="${NO_VQA_FEATURE_ROOT:-/root/scorer_pdms93_cache/no_vqa_e35_features_full_v1}"
shard_root="${NO_VQA_SEMANTIC_BEV_SHARD_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_full_current_semantic_bev_targets_v1_shards}"
final_root="${NO_VQA_SEMANTIC_BEV_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_full_current_semantic_bev_targets_v1}"
log_root="${NO_VQA_SEMANTIC_BEV_LOG_ROOT:-/root/scorer_pdms93_logs/no_vqa_e35_full_current_semantic_bev_targets_v1}"
num_shards="${NO_VQA_SEMANTIC_BEV_SHARDS:-24}"

[[ ! -e "${final_root}" ]] || {
  echo "refusing existing semantic-BEV final root: ${final_root}" >&2
  exit 2
}
mkdir -p "${shard_root}" "${log_root}"
export PYTHONPATH="${repo_root}:${repo_root}/nuplan-devkit${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

while true; do
  completed="$(find "${shard_root}" -maxdepth 1 -type f -name "shard_*-of-$(printf '%03d' "${num_shards}").npz" | wc -l)"
  (( completed == num_shards )) && break
  echo "NO_VQA_SEMANTIC_BEV_WAIT completed=${completed}/${num_shards} utc=$(date -u +%FT%TZ)"
  sleep 30
done

"${python_bin}" \
  "${repo_root}/local_stage2/build_full_current_semantic_bev_target_cache.py" \
  --mode aggregate \
  --feature-root "${feature_root}" \
  --output-root "${shard_root}" \
  --final-root "${final_root}" \
  --num-shards "${num_shards}" \
  --expected-scenes 103288 \
  >"${log_root}/aggregate.log" 2>&1

"${python_bin}" - "${final_root}" <<'PY' >"${log_root}/validate.log" 2>&1
import json
import sys
from pathlib import Path

from local_stage2.train_m0_private_residual_scorer import (
    load_semantic_bev_target_table,
)

root = Path(sys.argv[1])
table = load_semantic_bev_target_table(root)
assert len(table.tokens) == 103288
assert len(set(table.tokens)) == 103288
assert tuple(table.map_targets.shape) == (103288, 3, 16, 32)
assert tuple(table.agent_targets.shape) == (103288, 2, 16, 32)
assert bool(table.supervision_valid.all())
assert table.lineage["current_observation_only"] is True
assert table.lineage["depends_on_logged_future"] is False
assert table.lineage["available_as_model_input_at_inference"] is False
print(json.dumps({
    "scene_count": len(table.tokens),
    "valid_scene_count": int(table.supervision_valid.sum()),
    "status": "PASS",
}, sort_keys=True))
PY

touch "${final_root}/.complete"
echo "NO_VQA_SEMANTIC_BEV_CAMPAIGN_COMPLETE root=${final_root}"
