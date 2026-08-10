#!/usr/bin/env bash
# Offline one-node/16-PPU V-JEPA 2.1 dynamics-cache generation.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
source "$project_root/env.sh"

export VJEPA_REPO="${VJEPA_REPO:-$SHARED_WEIGHT_ROOT/facebookresearch/vjepa2}"
export VJEPA_REPO_COMMIT="${VJEPA_REPO_COMMIT:-204698b45b3712590f06245fbfba32d3be539812}"
export VJEPA_CHECKPOINT="${VJEPA_CHECKPOINT:-$SHARED_WEIGHT_ROOT/facebook/VJEPA2.1-ViT-L-384/vjepa2_1_vitl_dist_vitG_384.pt}"
export VJEPA_CHECKPOINT_REVISION="${VJEPA_CHECKPOINT_REVISION:-vjepa2_1_vitl_dist_vitG_384}"
export VJEPA_CHECKPOINT_SHA256="${VJEPA_CHECKPOINT_SHA256:-7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6}"
export VJEPA_MODEL_VARIANT="${VJEPA_MODEL_VARIANT:-vjepa2_1_vit_large_384}"
export FIELD2PLAN_DATALIST_PATH="${FIELD2PLAN_DATALIST_PATH:-$project_root/train_meta.json}"
export VJEPA_SPLIT="${VJEPA_SPLIT:-train}"
export VJEPA_META_ROOT="${VJEPA_META_ROOT:-$DATA_ROOT/meta/$VJEPA_SPLIT}"
export FIELD2PLAN_DYNAMICS_CACHE="${FIELD2PLAN_DYNAMICS_CACHE:-$DRIVEDREAMER_SHARED_ROOT/field2plan_cache/dynamics_vjepa2_1_vitl384_c96_16_v1}"
export VJEPA_CACHE_PROCESSES="${VJEPA_CACHE_PROCESSES:-16}"
export VJEPA_FEATURE_CHANNELS="${VJEPA_FEATURE_CHANNELS:-96}"
export VJEPA_FEATURE_HEIGHT="${VJEPA_FEATURE_HEIGHT:-16}"
export VJEPA_FEATURE_WIDTH="${VJEPA_FEATURE_WIDTH:-16}"
export VJEPA_PROJECTION_SEED="${VJEPA_PROJECTION_SEED:-20260809}"
export VJEPA_VIEW_BATCH_SIZE="${VJEPA_VIEW_BATCH_SIZE:-1}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

for variable in VJEPA_CACHE_PROCESSES VJEPA_FEATURE_CHANNELS VJEPA_FEATURE_HEIGHT VJEPA_FEATURE_WIDTH VJEPA_VIEW_BATCH_SIZE; do
  value="${!variable}"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "[vjepa-dynamics-cache] $variable must be a positive integer, got $value" >&2
    exit 2
  fi
done
if [ "$VJEPA_CACHE_PROCESSES" -ne 16 ] && [ "${VJEPA_ALLOW_NONFORMAL_TOPOLOGY:-0}" != "1" ]; then
  echo "[vjepa-dynamics-cache] formal topology is one node x 16 PPU; got $VJEPA_CACHE_PROCESSES" >&2
  exit 2
fi

required=(
  "$VJEPA_REPO/.git"
  "$VJEPA_CHECKPOINT"
  "$FIELD2PLAN_DATALIST_PATH"
  "$VJEPA_META_ROOT"
  "$OPENSCENE_DATA_ROOT"
  "$NAVSIM_TRAINVAL_SENSOR_ROOT"
)
for path in "${required[@]}"; do
  if [ ! -e "$path" ]; then
    echo "[vjepa-dynamics-cache] missing required local asset: $path" >&2
    exit 2
  fi
done
actual_repo_commit="$(git -C "$VJEPA_REPO" rev-parse HEAD)"
if [ "$actual_repo_commit" != "$VJEPA_REPO_COMMIT" ]; then
  echo "[vjepa-dynamics-cache] repo commit mismatch: expected=$VJEPA_REPO_COMMIT actual=$actual_repo_commit" >&2
  exit 2
fi

resolved_cache="$(readlink -m "$FIELD2PLAN_DYNAMICS_CACHE")"
resolved_parent="$(readlink -m "$DRIVEDREAMER_SHARED_ROOT/field2plan_cache")"
case "$resolved_cache" in
  "$resolved_parent"/*) ;;
  *)
    echo "[vjepa-dynamics-cache] output must be below $resolved_parent" >&2
    exit 2
    ;;
esac
export FIELD2PLAN_DYNAMICS_CACHE="$resolved_cache"

if [ "${VJEPA_TOPOLOGY_ONLY:-0}" = "1" ]; then
  echo "[vjepa-dynamics-cache] topology=1x$VJEPA_CACHE_PROCESSES repo_commit=$actual_repo_commit output=$FIELD2PLAN_DYNAMICS_CACHE"
  exit 0
fi

actual_devices="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if [ "$actual_devices" -lt "$VJEPA_CACHE_PROCESSES" ]; then
  echo "[vjepa-dynamics-cache] need $VJEPA_CACHE_PROCESSES visible accelerators, found $actual_devices" >&2
  exit 2
fi
actual_checkpoint_sha256="$(sha256sum "$VJEPA_CHECKPOINT" | awk '{print $1}')"
if [ "$actual_checkpoint_sha256" != "$VJEPA_CHECKPOINT_SHA256" ]; then
  echo "[vjepa-dynamics-cache] checkpoint checksum mismatch: expected=$VJEPA_CHECKPOINT_SHA256 actual=$actual_checkpoint_sha256" >&2
  exit 2
fi

mkdir -p "$FIELD2PLAN_DYNAMICS_CACHE/logs"
job_tag="${PAI_JOB_ID:-$(date +'%Y%m%d_%H%M%S')}"
launcher_log="$FIELD2PLAN_DYNAMICS_CACHE/logs/vjepa-cache-$job_tag.log"
compiler_cache_root="${VJEPA_COMPILER_CACHE_ROOT:-/tmp/field2plan-vjepa-cache}/$job_tag"
mkdir -p "$compiler_cache_root"
exec > >(tee -a "$launcher_log") 2>&1

common_args=(
  --datalist "$FIELD2PLAN_DATALIST_PATH"
  --split "$VJEPA_SPLIT"
  --meta-root "$VJEPA_META_ROOT"
  --runtime-raw-root "$OPENSCENE_DATA_ROOT"
  --trainval-sensor-root "$NAVSIM_TRAINVAL_SENSOR_ROOT"
  --vjepa-repo "$VJEPA_REPO"
  --checkpoint "$VJEPA_CHECKPOINT"
  --checkpoint-revision "$VJEPA_CHECKPOINT_REVISION"
  --checkpoint-sha256 "$VJEPA_CHECKPOINT_SHA256"
  --model-variant "$VJEPA_MODEL_VARIANT"
  --output-dir "$FIELD2PLAN_DYNAMICS_CACHE"
  --image-size 384
  --feature-channels "$VJEPA_FEATURE_CHANNELS"
  --output-height "$VJEPA_FEATURE_HEIGHT"
  --output-width "$VJEPA_FEATURE_WIDTH"
  --projection-seed "$VJEPA_PROJECTION_SEED"
  --view-batch-size "$VJEPA_VIEW_BATCH_SIZE"
  --compiler-cache-root "$compiler_cache_root"
  --device cuda
  --dtype bfloat16
)
if [ "${VJEPA_MAX_SAMPLES:-0}" -gt 0 ]; then
  common_args+=(--max-samples "$VJEPA_MAX_SAMPLES")
fi
if [ "${VJEPA_OVERWRITE:-0}" = "1" ]; then
  common_args+=(--overwrite)
fi

echo "[vjepa-dynamics-cache] topology=1x$VJEPA_CACHE_PROCESSES visible=$actual_devices independent_workers=true"
echo "[vjepa-dynamics-cache] teacher=$VJEPA_MODEL_VARIANT repo_commit=$actual_repo_commit"
echo "[vjepa-dynamics-cache] checkpoint_sha256=$actual_checkpoint_sha256"
echo "[vjepa-dynamics-cache] tensor=8x3x${VJEPA_FEATURE_CHANNELS}x${VJEPA_FEATURE_HEIGHT}x${VJEPA_FEATURE_WIDTH} output=$FIELD2PLAN_DYNAMICS_CACHE"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="$VJEPA_CACHE_PROCESSES" \
  tools/field2plan/cache_dynamics_vjepa.py \
  "${common_args[@]}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="$VJEPA_CACHE_PROCESSES" \
  tools/field2plan/cache_dynamics_vjepa.py \
  "${common_args[@]}" \
  --validate-only

echo "[vjepa-dynamics-cache] complete manifest=$FIELD2PLAN_DYNAMICS_CACHE/manifest.json"
