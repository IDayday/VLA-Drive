#!/usr/bin/env bash
# Compute immutable full-train pooled VGGT slot statistics.

set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

cache_root="${GP_SQ3DMIX_TRAIN_CACHE_ROOT:-$NAVSIM_VGGT_DENSE_CACHE_ROOT}"
datalist="${GP_SQ3DMIX_TRAIN_DATALIST:-$NAVSIM_DATALIST_PATH}"
stats_root="${GP_SQ3DMIX_STATS_ROOT:-$DRIVEDREAMER_SHARED_ROOT/navsim_feature_cache/gp_sq3dmix_slot_stats}"
dry_run=0
while (( $# )); do
  case "$1" in
    --cache-root) cache_root="${2:?}"; shift 2 ;;
    --datalist) datalist="${2:?}"; shift 2 ;;
    --stats-root) stats_root="${2:?}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) echo "Usage: bash $project_root/19-compute_gp_sq3dmix_slot_stats.sh [--cache-root DIR] [--datalist FILE] [--stats-root DIR] [--dry-run]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

branch="$(git -C "$project_root" branch --show-current)"
[[ "$branch" == "feature/gp-sq-3d-mix" ]] || { echo "Wrong DLC-visible branch: $branch" >&2; exit 2; }
[[ -f "$cache_root/vggt_dense/manifest.json" ]] || { echo "Missing train dense-cache manifest" >&2; exit 2; }
[[ -f "$datalist" ]] || { echo "Missing train datalist: $datalist" >&2; exit 2; }
echo "[gp-slot-stats] code=$project_root branch=$branch"
echo "[gp-slot-stats] cache=$cache_root datalist=$datalist output=$stats_root"
args=(--cache-root "$cache_root" --datalist "$datalist" --stats-root "$stats_root")
if (( dry_run )); then
  args+=(--dry-run)
  printf 'python %q ' "$project_root/tools/compute_gp_sq3dmix_slot_stats.py"
  printf '%q ' "${args[@]}"
  printf '\n'
  exit 0
fi
[[ ! -e "$stats_root" ]] || { echo "Refusing to overwrite stats root: $stats_root" >&2; exit 2; }
python "$project_root/tools/compute_gp_sq3dmix_slot_stats.py" "${args[@]}"
