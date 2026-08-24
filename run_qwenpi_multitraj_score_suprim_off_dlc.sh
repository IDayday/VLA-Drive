#!/usr/bin/env bash
# Matched arm A: Qwen + Flow-DiT + 64 proposals + DrivoR, no DriveSuprim.
# The config/intervention are fixed; paths and topology remain env-overridable.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export QDS_CONFIG_YAML="$project_root/starVLA/config/training/qwenpi_multitraj_score_suprim_off.yaml"
export QDS_EXPECT_DRIVESUPRIM=0
export QDS_RUN_ID="${QDS_RUN_ID:-qwenpi-multitraj-score-suprim-off-$(date +'%Y%m%d_%H%M%S')}"

exec bash "$project_root/train_qwenpi_drivor_suprim_dlc.sh" "$@"
