#!/usr/bin/env bash
# Build resumable full-train GP-SQ3D-Mix-v2 slot/descriptor statistics on CPU.

set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

cache_root="${GP_SQ3DMIX_TRAIN_CACHE_ROOT:-$NAVSIM_VGGT_DENSE_CACHE_ROOT}"
datalist="${GP_SQ3DMIX_TRAIN_DATALIST:-$NAVSIM_DATALIST_PATH}"
stats_root="${GP_SQ3DMIX_V2_STATS_ROOT:-$DRIVEDREAMER_SHARED_ROOT/navsim_feature_cache/gp_sq3dmix_stage_a_v2_slot_stats}"
num_workers="${GP_SQ3DMIX_STATS_WORKERS:-8}"
num_shards="${GP_SQ3DMIX_STATS_SHARDS:-16}"
shard_id=""
resume=1
merge_only=0
dry_run=0
while (( $# )); do
  case "$1" in
    --cache-root) cache_root="${2:?}"; shift 2 ;;
    --datalist) datalist="${2:?}"; shift 2 ;;
    --stats-root) stats_root="${2:?}"; shift 2 ;;
    --num-workers) num_workers="${2:?}"; shift 2 ;;
    --num-shards) num_shards="${2:?}"; shift 2 ;;
    --shard-id) shard_id="${2:?}"; shift 2 ;;
    --resume) resume=1; shift ;;
    --no-resume) resume=0; shift ;;
    --merge-only) merge_only=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help)
      echo "Usage: bash $project_root/19-compute_gp_sq3dmix_slot_stats.sh [--num-workers N] [--num-shards N] [--shard-id ID] [--resume|--no-resume] [--merge-only] [--dry-run]"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ "$num_workers" =~ ^[1-9][0-9]*$ ]] || { echo "--num-workers must be positive" >&2; exit 2; }
[[ "$num_shards" =~ ^[1-9][0-9]*$ ]] || { echo "--num-shards must be positive" >&2; exit 2; }
if [[ -n "$shard_id" ]]; then
  [[ "$shard_id" =~ ^[0-9]+$ && "$shard_id" -lt "$num_shards" ]] || { echo "Invalid --shard-id" >&2; exit 2; }
fi
(( ! merge_only )) || [[ -z "$shard_id" ]] || { echo "--merge-only cannot use --shard-id" >&2; exit 2; }
branch="$(git -C "$project_root" branch --show-current)"
commit="$(git -C "$project_root" rev-parse HEAD)"
[[ "$branch" == "feature/gp-sq-3d-mix-stage-a-v2" ]] || { echo "Wrong DLC-visible branch: $branch" >&2; exit 2; }
[[ -z "$(git -C "$project_root" status --porcelain)" ]] || { echo "Slot statistics require a clean DLC-visible worktree" >&2; exit 2; }
[[ -f "$cache_root/vggt_dense/manifest.json" ]] || { echo "Missing train dense-cache manifest" >&2; exit 2; }
[[ -f "$datalist" ]] || { echo "Missing train datalist: $datalist" >&2; exit 2; }

args=(--cache-root "$cache_root" --datalist "$datalist" --stats-root "$stats_root" --num-workers "$num_workers" --num-shards "$num_shards")
(( resume )) && args+=(--resume)
(( merge_only )) && args+=(--merge-only)
[[ -z "$shard_id" ]] || args+=(--shard-id "$shard_id")
(( dry_run )) && args+=(--dry-run)

echo "[gp-slot-stats-v2] code=$project_root branch=$branch commit=$commit"
echo "[gp-slot-stats-v2] cache=$cache_root datalist=$datalist output=$stats_root"
echo "[gp-slot-stats-v2] workers=$num_workers shards=$num_shards shard_id=${shard_id:-all} resume=$resume merge_only=$merge_only"
printf '[gp-slot-stats-v2] command: python %q ' "$project_root/tools/compute_gp_sq3dmix_slot_stats.py"
printf '%q ' "${args[@]}"
printf '\n'
PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}" python "$project_root/tools/compute_gp_sq3dmix_slot_stats.py" "${args[@]}"
