#!/usr/bin/env bash

# Build 100%-coverage current-frame actor supervision for the immutable
# No-VQA 103,288-scene replay inventory.  This is a CPU-only, training-target
# job; it does not read future frames, MetricCache, proposals or PDM values.

set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
python_bin="${DRIVEVLA_PYTHON:-/mnt/project/DriveVLA-M0-env/bin/python}"
feature_root="${NO_VQA_FEATURE_ROOT:-/root/scorer_pdms93_cache/no_vqa_e35_features_full_v1}"
shard_root="${NO_VQA_CURRENT_ACTOR_SHARD_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_full_current_actor_targets_v1_shards}"
final_root="${NO_VQA_CURRENT_ACTOR_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_full_current_actor_targets_v1}"
log_root="${NO_VQA_CURRENT_ACTOR_LOG_ROOT:-/root/scorer_pdms93_logs/no_vqa_e35_full_current_actor_targets_v1}"
num_shards="${NO_VQA_CURRENT_ACTOR_SHARDS:-8}"

for path in "${repo_root}" "${feature_root}" /mnt/navsim/trainval_navsim_logs/trainval /mnt/navsim/maps; do
  [[ -e "${path}" ]] || { echo "missing current-actor input: ${path}" >&2; exit 2; }
done
for path in "${shard_root}" "${final_root}" "${log_root}"; do
  [[ ! -e "${path}" ]] || { echo "refusing existing current-actor output: ${path}" >&2; exit 2; }
done

mkdir -p "${shard_root}" "${log_root}"
export PYTHONPATH="${repo_root}:${repo_root}/nuplan-devkit${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

pids=()
for shard in $(seq 0 $((num_shards - 1))); do
  nice -n 10 "${python_bin}" \
    "${repo_root}/local_stage2/build_full_current_actor_target_cache.py" \
    --mode shard \
    --feature-root "${feature_root}" \
    --output-root "${shard_root}" \
    --final-root "${final_root}" \
    --num-shards "${num_shards}" \
    --shard-index "${shard}" \
    >"${log_root}/shard_${shard}.log" 2>&1 &
  pids+=("$!")
  echo "NO_VQA_CURRENT_ACTOR_STARTED shard=${shard} pid=$!"
done

failure=0
for shard in "${!pids[@]}"; do
  if wait "${pids[${shard}]}"; then
    echo "NO_VQA_CURRENT_ACTOR_COMPLETE shard=${shard}"
  else
    echo "NO_VQA_CURRENT_ACTOR_FAILED shard=${shard}" >&2
    failure=1
  fi
done
(( failure == 0 )) || exit 1

"${python_bin}" "${repo_root}/local_stage2/build_full_current_actor_target_cache.py" \
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

from local_stage2.train_independent_scorer import load_current_actor_target_table

root = Path(sys.argv[1])
table = load_current_actor_target_table(root)
assert len(table.tokens) == 103288
assert len(set(table.tokens)) == 103288
assert tuple(table.actor_states.shape) == (103288, 16, 8)
assert tuple(table.actor_masks.shape) == (103288, 16)
assert bool(table.supervision_valid.all())
assert table.lineage["depends_on_logged_future"] is False
assert table.lineage["available_as_model_input_at_inference"] is False
print(json.dumps({
    "scene_count": len(table.tokens),
    "valid_scene_count": int(table.supervision_valid.sum()),
    "status": "PASS",
}, sort_keys=True))
PY

touch "${final_root}/.complete"
echo "NO_VQA_CURRENT_ACTOR_CAMPAIGN_COMPLETE root=${final_root}"
