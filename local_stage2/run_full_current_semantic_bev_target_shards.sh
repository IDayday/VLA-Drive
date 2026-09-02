#!/usr/bin/env bash

# Build an explicit range of independent current-frame semantic-BEV target
# shards. Multiple hosts may safely write disjoint shard indices to the shared
# output root. This is CPU-only and reads no future or evaluator data.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 START_SHARD END_SHARD_EXCLUSIVE" >&2
  exit 2
fi

repo_root="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
python_bin="${DRIVEVLA_PYTHON:-/mnt/project/DriveVLA-M0-env/bin/python}"
feature_root="${NO_VQA_FEATURE_ROOT:-/root/scorer_pdms93_cache/no_vqa_e35_features_full_v1}"
shard_root="${NO_VQA_SEMANTIC_BEV_SHARD_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_full_current_semantic_bev_targets_v1_shards}"
log_root="${NO_VQA_SEMANTIC_BEV_LOG_ROOT:-/root/scorer_pdms93_logs/no_vqa_e35_full_current_semantic_bev_targets_v1}"
num_shards="${NO_VQA_SEMANTIC_BEV_SHARDS:-24}"
start="$1"
end="$2"

if (( start < 0 || end <= start || end > num_shards )); then
  echo "invalid semantic-BEV shard range [${start},${end})/${num_shards}" >&2
  exit 2
fi
for path in "${repo_root}" "${feature_root}" /mnt/navsim/trainval_navsim_logs/trainval /mnt/navsim/maps; do
  [[ -e "${path}" ]] || { echo "missing semantic-BEV input: ${path}" >&2; exit 2; }
done
mkdir -p "${shard_root}" "${log_root}"
export PYTHONPATH="${repo_root}:${repo_root}/nuplan-devkit${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

pids=()
shards=()
for shard in $(seq "${start}" $((end - 1))); do
  destination="${shard_root}/shard_$(printf '%03d' "${shard}")-of-$(printf '%03d' "${num_shards}").npz"
  [[ ! -e "${destination}" ]] || {
    echo "refusing existing semantic-BEV shard: ${destination}" >&2
    exit 2
  }
  nice -n 10 "${python_bin}" \
    "${repo_root}/local_stage2/build_full_current_semantic_bev_target_cache.py" \
    --mode shard \
    --feature-root "${feature_root}" \
    --output-root "${shard_root}" \
    --num-shards "${num_shards}" \
    --shard-index "${shard}" \
    >"${log_root}/shard_${shard}.log" 2>&1 &
  pids+=("$!")
  shards+=("${shard}")
  echo "NO_VQA_SEMANTIC_BEV_STARTED shard=${shard} pid=$!"
done

failure=0
for index in "${!pids[@]}"; do
  if wait "${pids[${index}]}"; then
    echo "NO_VQA_SEMANTIC_BEV_COMPLETE shard=${shards[${index}]}"
  else
    echo "NO_VQA_SEMANTIC_BEV_FAILED shard=${shards[${index}]}" >&2
    failure=1
  fi
done
exit "${failure}"
