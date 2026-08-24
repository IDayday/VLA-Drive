#!/usr/bin/env bash
# Launch the matched 10k action-only control and GP pilot sequentially.
set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pair_id="${GP_STAGE_B_PAIR_ID:-gp-sq3dmix-stage-b-${PAI_JOB_ID:-$(date +'%Y%m%d_%H%M%S')}}"
pair_commit="$(git -C "$project_root" rev-parse HEAD)"
bash "$project_root/tools/launch_gp_sq3dmix_training.sh" --stage stage_b --variant control --run-id "${pair_id}-control" "$@"
[[ "$(git -C "$project_root" rev-parse HEAD)" == "$pair_commit" ]] || {
  echo "Code commit changed after the matched control; GP pilot is forbidden" >&2
  exit 2
}
bash "$project_root/tools/launch_gp_sq3dmix_training.sh" --stage stage_b --variant gp --run-id "${pair_id}-gp" "$@"
[[ "$(git -C "$project_root" rev-parse HEAD)" == "$pair_commit" ]] || {
  echo "Code commit changed during the matched Stage-B pair" >&2
  exit 2
}
