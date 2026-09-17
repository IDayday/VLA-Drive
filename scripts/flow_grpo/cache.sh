#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${CACHE_PATH:?Set a new metric cache output path}"
export NAVSIM_DEVKIT_ROOT="$PWD/navsim"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/project/DriveDreamer-Policy/navsim_raw}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/navsim/maps}"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NAVSIM_EXP_ROOT="$CACHE_PATH"
export PYTHONPATH="$PWD/navsim:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TRAIN_TEST_SPLIT=navtrain
bash navsim/scripts/evaluation/run_metric_caching.sh
