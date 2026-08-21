#!/usr/bin/env bash
# Copy to env.local.sh and set only machine-specific paths. The local file is
# ignored by git. Explicit variables supplied to the DLC job take precedence.

# export SHARED_WEIGHT_ROOT="${SHARED_WEIGHT_ROOT:-/absolute/path/to/model-weights}"
# export NAVSIM_PUBLIC_ROOT="${NAVSIM_PUBLIC_ROOT:-/absolute/path/to/navsim}"
# export DDP_DRS_EXPECTED_BRANCH="${DDP_DRS_EXPECTED_BRANCH:-feature/ddp-drs-scene-2048}"
# export DDP_DRS_BASE_VLM="${DDP_DRS_BASE_VLM:-/absolute/path/to/Qwen3-VL-2B-WorldAction}"
# export DDP_DRS_BASE_DDP_CHECKPOINT="${DDP_DRS_BASE_DDP_CHECKPOINT:-/absolute/path/to/base-ddp/pytorch_model.pt}"
# export DDP_DRS_STATIC_VOCAB="${DDP_DRS_STATIC_VOCAB:-/absolute/path/to/test_8192_kmeans.npy}"
# export DDP_DRS_SUPRIM_STATIC_SCORE_PATH="${DDP_DRS_SUPRIM_STATIC_SCORE_PATH:-/absolute/path/to/DriveSuprim/navtrain.pkl}"
# export DDP_DRS_DATALIST_PATH="${DDP_DRS_DATALIST_PATH:-/absolute/path/to/train_meta.json}"
# export DDP_DRS_PROCESSED_DATA_ROOT="${DDP_DRS_PROCESSED_DATA_ROOT:-/absolute/path/to/processed-navsim}"
# export DDP_DRS_OPENSCENE_DATA_ROOT="${DDP_DRS_OPENSCENE_DATA_ROOT:-/absolute/path/to/navsim}"
# export DDP_DRS_NAVSIM_LOG_PATH="${DDP_DRS_NAVSIM_LOG_PATH:-/absolute/path/to/navsim_logs/trainval}"
# export DDP_DRS_NAVSIM_SENSOR_PATH="${DDP_DRS_NAVSIM_SENSOR_PATH:-/absolute/path/to/sensor_blobs/trainval}"
# export DDP_DRS_MAPS_ROOT="${DDP_DRS_MAPS_ROOT:-/absolute/path/to/nuplan/maps}"
# export DDP_DRS_METRIC_CACHE_ROOT="${DDP_DRS_METRIC_CACHE_ROOT:-/absolute/path/to/navsim-metric-cache}"
# export DDP_DRS_CACHE_ROOT="${DDP_DRS_CACHE_ROOT:-/absolute/path/to/ddp-drs-training-cache}"
# export DDP_DRS_RUN_ROOT="${DDP_DRS_RUN_ROOT:-/absolute/path/to/ddp-drs-runs}"

# QwenPI-DrivoRSuprim one-task training.  Keep these developer/container
# locations out of env.sh and shared YAML; one-shot job variables still win.
# export QWEN_VLM_PATH="${QWEN_VLM_PATH:-/absolute/path/to/Qwen3-VL-2B-WorldAction}"
# export QDS_ASSET_ROOT="${QDS_ASSET_ROOT:-/absolute/path/to/ddp-drs-assets}"
# export SUPRIM_VOCAB_PATH="${SUPRIM_VOCAB_PATH:-/absolute/path/to/test_8192_kmeans.npy}"
# export SUPRIM_STATIC_SCORE_CACHE="${SUPRIM_STATIC_SCORE_CACHE:-/absolute/path/to/static-score-shards}"
# export NAVSIM_DATALIST_PATH="${NAVSIM_DATALIST_PATH:-/absolute/path/to/train_meta.json}"
# export DATA_ROOT="${DATA_ROOT:-/absolute/path/to/processed-navsim}"
# export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/absolute/path/to/raw-navsim}"
# export NAVSIM_METRIC_CACHE_ROOT="${NAVSIM_METRIC_CACHE_ROOT:-/absolute/path/to/navsim-metric-cache}"
# export VLA_OUTPUT_ROOT="${VLA_OUTPUT_ROOT:-/absolute/path/to/training-runs}"
