#!/usr/bin/env bash
# Non-interactive Phase-2 frozen baseline draft-cache launcher.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
source "$project_root/env.sh"

checkpoint="${FIELD2PLAN_BASELINE_CHECKPOINT:-$project_root/navsim_exp/navsim-v1-formal-16gpu-bz2-ga1-flashattn-20260801_193514/final_model/pytorch_model.pt}"
datalist="${FIELD2PLAN_DATALIST_PATH:-$project_root/train_meta.json}"
split="${FIELD2PLAN_CACHE_SPLIT:-train}"
cache_root="${FIELD2PLAN_DRAFT_CACHE_ROOT:-$DRIVEDREAMER_SHARED_ROOT/field2plan_cache/baseline_drafts_193514_seed20260808_steps10}"
seed="${FIELD2PLAN_DRAFT_SEED:-20260808}"
steps="${FIELD2PLAN_DRAFT_STEPS:-10}"
candidates="${FIELD2PLAN_DRAFT_CANDIDATES:-1}"
batch_size="${FIELD2PLAN_DRAFT_BATCH_SIZE:-4}"
num_workers="${FIELD2PLAN_DRAFT_NUM_WORKERS:-4}"
max_samples="${FIELD2PLAN_DRAFT_MAX_SAMPLES:-0}"

visible_devices="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
processes="${FIELD2PLAN_DRAFT_PROCESSES:-$visible_devices}"
formal_job=0
if [ -n "${NPROC_PER_NODE:-}" ] || [ -n "${PAI_JOB_ID:-}" ]; then
  formal_job=1
fi
if (( formal_job == 1 )) && (( processes != 16 || visible_devices < 16 )); then
  echo "[field2plan-draft-cache] formal DLC requires one node x 16 devices; processes=$processes visible=$visible_devices" >&2
  exit 2
fi
if (( processes < 1 || processes > visible_devices )); then
  echo "[field2plan-draft-cache] invalid process count: processes=$processes visible=$visible_devices" >&2
  exit 2
fi
for integer in seed steps candidates batch_size num_workers max_samples; do
  value="${!integer}"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "[field2plan-draft-cache] $integer must be non-negative integer, got $value" >&2
    exit 2
  fi
done
if (( seed < 1 || steps < 1 || candidates < 1 || batch_size < 1 )); then
  echo "[field2plan-draft-cache] seed/steps/candidates/batch_size must be positive" >&2
  exit 2
fi
for required in "$checkpoint" "$datalist" "$DATA_ROOT/meta/$split" "$BASE_VLM/config.json"; do
  if [ ! -e "$required" ]; then
    echo "[field2plan-draft-cache] missing required path: $required" >&2
    exit 2
  fi
done
resolved_cache="$(readlink -m "$cache_root")"
resolved_parent="$(readlink -m "$DRIVEDREAMER_SHARED_ROOT/field2plan_cache")"
case "$resolved_cache" in
  "$resolved_parent"/*) ;;
  *)
    echo "[field2plan-draft-cache] cache must be below shared project directory: $resolved_parent" >&2
    exit 2
    ;;
esac
cache_root="$resolved_cache"
mkdir -p "$cache_root/logs"
run_id="${FIELD2PLAN_DRAFT_RUN_ID:-draft-cache-$(date +'%Y%m%d_%H%M%S')}"
log_path="$cache_root/logs/$run_id.log"
export TRITON_CACHE_DIR="${TRITON_LOCAL_CACHE_ROOT:-/tmp/field2plan-draft-triton}/$run_id"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
unset NAVSIM_FEATURE_CACHE_ROOT NAVSIM_CACHE_COMPONENTS

exec > >(tee -a "$log_path") 2>&1
echo "[field2plan-draft-cache] project_root=$project_root"
echo "[field2plan-draft-cache] topology=1node x ${processes}processes visible=$visible_devices"
echo "[field2plan-draft-cache] checkpoint=$checkpoint split=$split datalist=$datalist"
echo "[field2plan-draft-cache] output=$cache_root seed=$seed steps=$steps candidates=$candidates"
echo "[field2plan-draft-cache] batch_per_rank=$batch_size workers_per_rank=$num_workers max_samples=$max_samples"

if [ "${FIELD2PLAN_TOPOLOGY_ONLY:-0}" = "1" ]; then
  exit 0
fi

args=(
  --checkpoint "$checkpoint"
  --datalist "$datalist"
  --data-root "$DATA_ROOT"
  --split "$split"
  --output-dir "$cache_root"
  --seed "$seed"
  --inference-steps "$steps"
  --num-candidates "$candidates"
  --batch-size "$batch_size"
  --num-workers "$num_workers"
  --max-samples "$max_samples"
  --qwen-forward-mode optimized
)
if [ "${FIELD2PLAN_DRAFT_OVERWRITE:-0}" = "1" ]; then
  args+=(--overwrite)
fi
if [ "${FIELD2PLAN_DRAFT_VALIDATE_ONLY:-0}" = "1" ]; then
  args+=(--validate-only)
fi

torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node "$processes" \
  tools/field2plan/cache_baseline_drafts.py "${args[@]}"
