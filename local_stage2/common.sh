#!/usr/bin/env bash

set -euo pipefail

script_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRIVEVLA_REPO_ROOT="${DRIVEVLA_REPO_ROOT:-${script_repo_root}}"
DRIVEVLA_PYTHON="${DRIVEVLA_PYTHON:-/mnt/project/DriveVLA-M0-env/bin/python}"
DRIVEVLA_VLM_DIR="${DRIVEVLA_VLM_DIR:-/mnt/project/DriveVLA-M0-models/InternVL3-2B-modelscope}"
DRIVEVLA_DINO_FILE="${DRIVEVLA_DINO_FILE:-/mnt/project/external/DrivoR/weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors}"
DRIVEVLA_PUBLIC_BASE="${DRIVEVLA_PUBLIC_BASE:-/mnt/project/DriveVLA-M0-modelscope/best-epoch_26-step_174312.server_merged.ckpt}"
DRIVEVLA_PUBLIC_BASE_CSV="${DRIVEVLA_PUBLIC_BASE_CSV:-/mnt/project/DriveVLA-M0-runs/ke/public_base_navtest_full/08.27_09.46/2026.08.27.10.03.42.csv}"

DRIVEVLA_DATA_ROOT="${DRIVEVLA_DATA_ROOT:-/mnt/project/DriveDreamer-Policy/navsim_raw}"
DRIVEVLA_SENSOR_ROOT="${DRIVEVLA_SENSOR_ROOT:-/mnt/project/onevl_navsim_data/sensor_blobs}"
DRIVEVLA_MAP_ROOT="${DRIVEVLA_MAP_ROOT:-/mnt/navsim/maps}"
DRIVEVLA_NAVTRAIN_METRIC_CACHE="${DRIVEVLA_NAVTRAIN_METRIC_CACHE:-/mnt/project/DriveVLA-M0-stage2/cache/metric_cache_navtrain_full}"
DRIVEVLA_NAVTRAIN_FEATURE_CACHE="${DRIVEVLA_NAVTRAIN_FEATURE_CACHE:-/mnt/project/DriveVLA-M0-stage2/cache/feature_cache_navtrain_full}"
DRIVEVLA_NAVTRAIN_LONG2_FEATURE_CACHE="${DRIVEVLA_NAVTRAIN_LONG2_FEATURE_CACHE:-/mnt/project/DriveVLA-M0-stage2/cache/feature_cache_navtrain_long2}"
DRIVEVLA_NAVTEST_METRIC_CACHE="${DRIVEVLA_NAVTEST_METRIC_CACHE:-/mnt/project/DriveDreamer-Policy/navsim_exp/eval_v1_1/metric_cache_navtest}"
DRIVEVLA_STAGE2_RUN_ROOT="${DRIVEVLA_STAGE2_RUN_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs}"

export DRIVEVLA_REPO_ROOT DRIVEVLA_PYTHON DRIVEVLA_VLM_DIR
export DRIVEVLA_DINO_FILE DRIVEVLA_PUBLIC_BASE DRIVEVLA_PUBLIC_BASE_CSV
export DRIVEVLA_DATA_ROOT DRIVEVLA_SENSOR_ROOT
export DRIVEVLA_MAP_ROOT DRIVEVLA_NAVTRAIN_METRIC_CACHE
export DRIVEVLA_NAVTRAIN_FEATURE_CACHE DRIVEVLA_NAVTRAIN_LONG2_FEATURE_CACHE
export DRIVEVLA_NAVTEST_METRIC_CACHE
export DRIVEVLA_STAGE2_RUN_ROOT

export PYTHONPATH="${DRIVEVLA_REPO_ROOT}:${DRIVEVLA_REPO_ROOT}/nuplan-devkit${PYTHONPATH:+:${PYTHONPATH}}"
export NUPLAN_MAPS_ROOT="${DRIVEVLA_MAP_ROOT}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export OPENBLAS_CORETYPE="Haswell"
export HYDRA_FULL_ERROR="1"
export DRIVEVLA_VLM_CONFIG="${DRIVEVLA_VLM_DIR}"
export DRIVEVLA_DINO_WEIGHTS="${DRIVEVLA_DINO_FILE}"
export NAVSIM_EXP_ROOT="${DRIVEVLA_STAGE2_RUN_ROOT}"
export SUBSCORE_PATH="${DRIVEVLA_STAGE2_RUN_ROOT}"
