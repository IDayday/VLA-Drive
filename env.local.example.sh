#!/usr/bin/env bash

# Copy this file to env.local.sh and replace the paths. env.local.sh is ignored
# by git so machine-local mounts and credentials cannot enter commits.
export DRIVEVLA_ASSET_ROOT="${DRIVEVLA_ASSET_ROOT:-/absolute/path/to/drivevla_m0_assets}"
export DRIVEVLA_BASE_CHECKPOINT="${DRIVEVLA_BASE_CHECKPOINT:-${DRIVEVLA_ASSET_ROOT}/base/best-epoch_26-step_174312.server_merged.ckpt}"
export DRIVEVLA_VLM_CONFIG="${DRIVEVLA_VLM_CONFIG:-${DRIVEVLA_ASSET_ROOT}/vlm/InternVL3-2B}"
export DRIVEVLA_DINO_WEIGHTS="${DRIVEVLA_DINO_WEIGHTS:-/absolute/path/to/dinov2/model.safetensors}"

# OPENSCENE_DATA_ROOT must contain meta_datas/test and sensor_blobs/test.
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/absolute/path/to/navsim_dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/absolute/path/to/navsim/maps}"
export METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/absolute/path/to/metric_cache_navtest_v1_1}"

# Full-navtest dual-metric evaluation uses separate official devkits/caches.
# The NAVSIM v2 checkout is external and is never copied into this repository.
export PDMS_METRIC_CACHE_PATH="${PDMS_METRIC_CACHE_PATH:-${METRIC_CACHE_PATH}}"
export EPDMS_METRIC_CACHE_PATH="${EPDMS_METRIC_CACHE_PATH:-/absolute/path/to/navsim_v2_metric_cache_navtest}"
export DRIVEVLA_NAVSIM_V2_ROOT="${DRIVEVLA_NAVSIM_V2_ROOT:-/absolute/path/to/navsim_v2_devkit}"
export DRIVEVLA_NAVTEST_TOKEN_LIST="${DRIVEVLA_NAVTEST_TOKEN_LIST:-/absolute/path/to/test_meta.json}"
export DRIVEVLA_DLC_EVAL_ROOT="${DRIVEVLA_DLC_EVAL_ROOT:-${DRIVEVLA_ASSET_ROOT}/outputs/dlc_navtest_dual_metrics}"

export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${DRIVEVLA_ASSET_ROOT}/outputs}"
export SUBSCORE_PATH="${SUBSCORE_PATH:-${NAVSIM_EXP_ROOT}}"
export PYTHON_BIN="${PYTHON_BIN:-${DRIVEVLA_REPO_ROOT}/.venv/bin/python}"
export NUM_GPUS="${NUM_GPUS:-1}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export NUM_WORKERS="${NUM_WORKERS:-4}"

# DLC submission settings. Keep credentials in the dlc CLI config, never here.
export DLC_PROJECT_ROOT="${DLC_PROJECT_ROOT:-${DRIVEVLA_REPO_ROOT}}"
export DLC_WORKSPACE_ID="${DLC_WORKSPACE_ID:-}"
export DLC_RESOURCE_ID="${DLC_RESOURCE_ID:-}"
export DLC_WORKER_IMAGE="${DLC_WORKER_IMAGE:-}"
export DLC_WORKER_GPU_TYPE="${DLC_WORKER_GPU_TYPE:-PPU}"
export DLC_REGION="${DLC_REGION:-cn-wulanchabu}"
export DLC_ENDPOINT="${DLC_ENDPOINT:-pai-dlc.${DLC_REGION}.aliyuncs.com}"
