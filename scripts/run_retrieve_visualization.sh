#!/usr/bin/env bash
set -euo pipefail

export OPENBLAS_CORETYPE="${OPENBLAS_CORETYPE:-Haswell}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/GPU_JDTest_fs01/home/zdhs0164/navsim_data/maps}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-/GPU_JDTest_fs01/home/zdhs0164/miniconda3/envs/navsim/bin/python}"

"${PYTHON_BIN}" "${REPO_ROOT}/tools/verify_retrieve_model.py" "$@"
