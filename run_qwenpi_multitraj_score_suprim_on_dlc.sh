#!/usr/bin/env bash
# Matched arm B: arm A plus DriveSuprim joint coarse/fine reranking.
# The config/intervention are fixed. The production defaults consume all 16
# DLC accelerators while preserving the original global batch of 64.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export QDS_LOCAL_PROCESSES="${QDS_LOCAL_PROCESSES:-16}"
export VLA_BATCH_SIZE="${VLA_BATCH_SIZE:-4}"
export QDS_TARGET_EFFECTIVE_BATCH="${QDS_TARGET_EFFECTIVE_BATCH:-64}"
# Keep CPU-side process concurrency identical to the OFF arm.
export NAVSIM_NUM_WORKERS="${NAVSIM_NUM_WORKERS:-3}"
export NAVSIM_METRIC_WORKERS="${NAVSIM_METRIC_WORKERS:-4}"
export QDS_CONFIG_YAML="$project_root/starVLA/config/training/qwenpi_multitraj_score_suprim_on.yaml"
export QDS_EXPECT_DRIVESUPRIM=1
export QDS_RUN_ID="${QDS_RUN_ID:-qwenpi-multitraj-score-suprim-on-$(date +'%Y%m%d_%H%M%S')}"

exec bash "$project_root/train_qwenpi_drivor_suprim_dlc.sh" "$@"
