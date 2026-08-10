#!/usr/bin/env bash
# Offline one-node/16-PPU cache generation for pinned public VGGT-1B.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
source "$project_root/env.sh"

export VGGT_REPO="${VGGT_REPO:-$SHARED_WEIGHT_ROOT/facebookresearch/vggt}"
export VGGT_CHECKPOINT="${VGGT_CHECKPOINT:-$SHARED_WEIGHT_ROOT/facebook/VGGT-1B/model.safetensors}"
export VGGT_CHECKPOINT_REVISION="${VGGT_CHECKPOINT_REVISION:-860abec7937da0a4c03c41d3c269c366e82abdf9}"
export VGGT_METRICIZATION="${VGGT_METRICIZATION:-da3_scale_anchor}"
export VGGT_SPLIT="${VGGT_SPLIT:-train}"
export VGGT_META_ROOT="${VGGT_META_ROOT:-$DATA_ROOT/meta/$VGGT_SPLIT}"
export VGGT_DA3_ROOT="${VGGT_DA3_ROOT:-$VGGT_META_ROOT}"
export FIELD2PLAN_DATALIST_PATH="${FIELD2PLAN_DATALIST_PATH:-$project_root/train_meta.json}"
export FIELD2PLAN_VGGT_CACHE="${FIELD2PLAN_VGGT_CACHE:-$DRIVEDREAMER_SHARED_ROOT/field2plan_cache/geometry_vggt_1b_da3_anchor_v1}"
export VGGT_NUM_PROCESSES="${VGGT_NUM_PROCESSES:-16}"
export VGGT_OUTPUT_HEIGHT="${VGGT_OUTPUT_HEIGHT:-144}"
export VGGT_OUTPUT_WIDTH="${VGGT_OUTPUT_WIDTH:-256}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TRITON_CACHE_DIR="${TRITON_LOCAL_CACHE_ROOT:-/tmp/field2plan-triton}/vggt-cache-${PAI_JOB_ID:-local}"
mkdir -p "$TRITON_CACHE_DIR" "$FIELD2PLAN_VGGT_CACHE"

if [ "$VGGT_NUM_PROCESSES" -ne 16 ] && [ "${VGGT_ALLOW_NONFORMAL_TOPOLOGY:-0}" != "1" ]; then
  echo "[vggt-cache] formal cache topology is one node x 16 PPU; got $VGGT_NUM_PROCESSES" >&2
  exit 2
fi

required=(
  "$VGGT_REPO/.git"
  "$VGGT_CHECKPOINT"
  "$FIELD2PLAN_DATALIST_PATH"
  "$VGGT_META_ROOT"
  "$OPENSCENE_DATA_ROOT"
)
case "$VGGT_METRICIZATION" in
  da3_scale_anchor)
    required+=("$VGGT_DA3_ROOT")
    ;;
  camera_rig)
    ;;
  *)
    echo "[vggt-cache] VGGT_METRICIZATION must be da3_scale_anchor or camera_rig" >&2
    exit 2
    ;;
esac
for path in "${required[@]}"; do
  if [ ! -e "$path" ]; then
    echo "[vggt-cache] missing required local asset: $path" >&2
    exit 2
  fi
done

actual_repo_commit="$(git -C "$VGGT_REPO" rev-parse HEAD)"
expected_repo_commit="${VGGT_REPO_COMMIT:-a288dd0f14786c93483e45524328726ab7b1b4ce}"
if [ "$actual_repo_commit" != "$expected_repo_commit" ]; then
  echo "[vggt-cache] VGGT repo commit mismatch: expected=$expected_repo_commit actual=$actual_repo_commit" >&2
  exit 2
fi

if [ "${VGGT_TOPOLOGY_ONLY:-0}" = "1" ]; then
  echo "[vggt-cache] topology=$VGGT_NUM_PROCESSES repo_commit=$actual_repo_commit output=$FIELD2PLAN_VGGT_CACHE"
  exit 0
fi

actual_devices="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if [ "$actual_devices" -lt "$VGGT_NUM_PROCESSES" ]; then
  echo "[vggt-cache] need $VGGT_NUM_PROCESSES visible accelerators, found $actual_devices" >&2
  exit 2
fi

launcher_log_dir="$DRIVEDREAMER_SHARED_ROOT/navsim_exp/launcher_logs"
mkdir -p "$launcher_log_dir"
launcher_log="$launcher_log_dir/vggt-cache-${PAI_JOB_ID:-$(date +'%Y%m%d_%H%M%S')}.log"
exec > >(tee -a "$launcher_log") 2>&1

common_args=(
  --datalist "$FIELD2PLAN_DATALIST_PATH"
  --split "$VGGT_SPLIT"
  --meta-root "$VGGT_META_ROOT"
  --runtime-raw-root "$OPENSCENE_DATA_ROOT"
  --trainval-sensor-root "$NAVSIM_TRAINVAL_SENSOR_ROOT"
  --vggt-repo "$VGGT_REPO"
  --checkpoint "$VGGT_CHECKPOINT"
  --checkpoint-revision "$VGGT_CHECKPOINT_REVISION"
  --metricization "$VGGT_METRICIZATION"
  --output-dir "$FIELD2PLAN_VGGT_CACHE"
  --output-height "$VGGT_OUTPUT_HEIGHT"
  --output-width "$VGGT_OUTPUT_WIDTH"
  --frame-index 3
  --device cuda
)
if [ "$VGGT_METRICIZATION" = "da3_scale_anchor" ]; then
  common_args+=(--da3-root "$VGGT_DA3_ROOT")
fi
if [ "${VGGT_MAX_SAMPLES:-0}" -gt 0 ]; then
  common_args+=(--max-samples "$VGGT_MAX_SAMPLES")
fi
if [ "${VGGT_OVERWRITE:-0}" = "1" ]; then
  common_args+=(--overwrite)
fi

echo "[vggt-cache] topology=1x$VGGT_NUM_PROCESSES visible=$actual_devices"
echo "[vggt-cache] repo=$VGGT_REPO commit=$actual_repo_commit"
echo "[vggt-cache] checkpoint=$VGGT_CHECKPOINT revision=$VGGT_CHECKPOINT_REVISION"
echo "[vggt-cache] metricization=$VGGT_METRICIZATION output=$FIELD2PLAN_VGGT_CACHE shape=3x${VGGT_OUTPUT_HEIGHT}x${VGGT_OUTPUT_WIDTH}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="$VGGT_NUM_PROCESSES" \
  tools/field2plan/cache_geometry_vggt.py \
  "${common_args[@]}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="$VGGT_NUM_PROCESSES" \
  tools/field2plan/cache_geometry_vggt.py \
  "${common_args[@]}" \
  --validate-only

echo "[vggt-cache] complete manifest=$FIELD2PLAN_VGGT_CACHE/manifest.json"
