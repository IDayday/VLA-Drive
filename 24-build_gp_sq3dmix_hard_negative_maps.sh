#!/usr/bin/env bash
# Build immutable fixed hard-negative maps for every GP-SQ3D-Mix-v2 split.

set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

source_datalist="${GP_SQ3DMIX_TRAIN_DATALIST:-$NAVSIM_DATALIST_PATH}"
source_cache="${GP_SQ3DMIX_TRAIN_CACHE_ROOT:-$NAVSIM_VGGT_DENSE_CACHE_ROOT}"
test_cache="${GP_SQ3DMIX_TEST_CACHE_ROOT:-${NAVSIM_VGGT_DENSE_TEST_CACHE_ROOT:-}}"
stats_root="${GP_SQ3DMIX_V2_STATS_ROOT:-$DRIVEDREAMER_SHARED_ROOT/navsim_feature_cache/gp_sq3dmix_stage_a_v2_slot_stats}"
output_root="${GP_SQ3DMIX_V2_NEGATIVE_MAP_ROOT:-$DRIVEDREAMER_SHARED_ROOT/navsim_feature_cache/gp_sq3dmix_stage_a_v2_negative_maps}"
processed_root="${DATA_ROOT:?DATA_ROOT must point at processed NAVSIM metadata}"
source_index="${GP_SQ3DMIX_V2_SOURCE_INDEX:-$output_root/source_planning_index.pt}"
source_log_index="${GP_SQ3DMIX_TRAIN_LOG_INDEX:-$processed_root/meta/train_video_log_index.json}"
test_datalist="${NAVSIM_TEST_DATALIST:-$DRIVEDREAMER_SHARED_ROOT/test_meta.json}"
workers="${GP_SQ3DMIX_NEGATIVE_MAP_WORKERS:-16}"
include_full=0
dry_run=0
resume=1
only=""
while (( $# )); do
  case "$1" in
    --source-datalist) source_datalist="${2:?}"; shift 2 ;;
    --source-cache-root) source_cache="${2:?}"; shift 2 ;;
    --test-cache-root) test_cache="${2:?}"; shift 2 ;;
    --stats-root) stats_root="${2:?}"; shift 2 ;;
    --output-root) output_root="${2:?}"; shift 2 ;;
    --processed-root) processed_root="${2:?}"; shift 2 ;;
    --num-workers) workers="${2:?}"; shift 2 ;;
    --only) only="${2:?}"; shift 2 ;;
    --include-full-navtest) include_full=1; shift ;;
    --resume) resume=1; shift ;;
    --no-resume) resume=0; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help)
      echo "Usage: bash $project_root/24-build_gp_sq3dmix_hard_negative_maps.sh [--only NAME] [--include-full-navtest] [--resume|--no-resume] [--dry-run]"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ "$workers" =~ ^[1-9][0-9]*$ ]] || { echo "--num-workers must be positive" >&2; exit 2; }
branch="$(git -C "$project_root" branch --show-current)"
commit="$(git -C "$project_root" rev-parse HEAD)"
[[ "$branch" == "feature/gp-sq-3d-mix-stage-a-v2" ]] || { echo "Wrong DLC-visible branch: $branch" >&2; exit 2; }
[[ -z "$(git -C "$project_root" status --porcelain)" ]] || { echo "Hard-negative maps require a clean DLC-visible worktree" >&2; exit 2; }
for path in \
  "$source_datalist" \
  "$source_cache/vggt_dense/manifest.json" \
  "$stats_root/manifest.json" \
  "$stats_root/pooled_scene_descriptors.pt" \
  "$source_log_index"; do
  [[ -f "$path" ]] || { echo "Missing hard-negative input: $path" >&2; exit 2; }
done

declare -a names datalists caches splits
names=(train_full stage_a_v2_train stage_a_v2_model_selection stage_a_v2_final_gate smoke_train_256 smoke_selection_128 navtest_2k)
datalists=(
  "$source_datalist"
  "$project_root/docs/experiments/splits/gp_sq3dmix_stage_a_v2_train.json"
  "$project_root/docs/experiments/splits/gp_sq3dmix_stage_a_v2_model_selection.json"
  "$project_root/docs/experiments/splits/gp_sq3dmix_stage_a_v2_final_gate.json"
  "$project_root/docs/experiments/splits/gp_sq3dmix_stage_a_v2_smoke_train_256.json"
  "$project_root/docs/experiments/splits/gp_sq3dmix_stage_a_v2_smoke_selection_128.json"
  "$project_root/docs/experiments/splits/gp_sq3dmix_navtest_2k.json"
)
caches=("$source_cache" "$source_cache" "$source_cache" "$source_cache" "$source_cache" "$source_cache" "$test_cache")
splits=(train train train train train train test)
if (( include_full )); then
  names+=(navtest_full)
  datalists+=("$test_datalist")
  caches+=("$test_cache")
  splits+=(test)
fi

echo "[gp-hard-negative-v2] code=$project_root branch=$branch commit=$commit"
echo "[gp-hard-negative-v2] source=$source_datalist stats=$stats_root output=$output_root"
for index in "${!names[@]}"; do
  name="${names[$index]}"
  [[ -z "$only" || "$name" == "$only" ]] || continue
  target_datalist="${datalists[$index]}"
  target_cache="${caches[$index]}"
  target_split="${splits[$index]}"
  [[ -f "$target_datalist" ]] || { echo "Missing target split: $target_datalist" >&2; exit 2; }
  [[ -n "$target_cache" && -f "$target_cache/vggt_dense/manifest.json" ]] || { echo "Missing target dense cache for $name: $target_cache" >&2; exit 2; }
  map_dir="$output_root/$name"
  if [[ -e "$map_dir" ]]; then
    (( resume )) || { echo "Refusing existing map directory: $map_dir" >&2; exit 2; }
    python - "$map_dir" "$target_datalist" "$target_cache/vggt_dense/manifest.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
root=Path(sys.argv[1]); manifest_path=root/'manifest.json'; map_path=root/'hard_negative_map.json'
if not manifest_path.is_file() or not map_path.is_file(): raise SystemExit(f'incomplete existing map: {root}')
manifest=json.load(open(manifest_path))
checks={
 'complete': manifest.get('complete') is True,
 'target': manifest.get('target_split_sha256') == sha(sys.argv[2]),
 'cache': manifest.get('dense_cache_manifest_sha256') == sha(sys.argv[3]),
 'map': manifest.get('map_file_sha256') == sha(map_path),
 'fallback': float(manifest.get('fallback_rate', 1.0)) <= 0.01,
 'self': int(manifest.get('self_donor_count', -1)) == 0,
 'same_log': int(manifest.get('same_log_violation_count', -1)) == 0,
}
if not all(checks.values()): raise SystemExit(f'existing map failed validation: {checks}')
PY
    echo "[gp-hard-negative-v2] resume skip complete map: $name"
    continue
  fi
  args=(
    --source-datalist "$source_datalist"
    --source-cache-root "$source_cache"
    --source-processed-root "$processed_root"
    --source-descriptors "$stats_root/pooled_scene_descriptors.pt"
    --source-stats-manifest "$stats_root/manifest.json"
    --source-index "$source_index"
    --source-log-index "$source_log_index"
    --target-datalist "$target_datalist"
    --target-cache-root "$target_cache"
    --target-processed-root "$processed_root"
    --target-split "$target_split"
    --output-dir "$map_dir"
    --num-workers "$workers"
  )
  (( dry_run )) && args+=(--dry-run)
  printf '[gp-hard-negative-v2] %s command: python %q ' "$name" "$project_root/tools/build_gp_sq3dmix_hard_negative_map.py"
  printf '%q ' "${args[@]}"
  printf '\n'
  PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}" python "$project_root/tools/build_gp_sq3dmix_hard_negative_map.py" "${args[@]}"
done
if [[ -n "$only" ]]; then
  found=0
  for name in "${names[@]}"; do [[ "$name" != "$only" ]] || found=1; done
  (( found )) || { echo "Unknown --only map: $only" >&2; exit 2; }
fi
