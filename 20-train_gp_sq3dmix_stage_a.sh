#!/usr/bin/env bash
set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$project_root/tools/launch_gp_sq3dmix_training.sh" --stage stage_a --variant gp "$@"
