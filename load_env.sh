#!/usr/bin/env bash

# Source this file from launchers. Values exported by the caller take
# precedence over env.local.sh; env.local.sh takes precedence over defaults.
DRIVEVLA_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DRIVEVLA_REPO_ROOT

DRIVEVLA_ENV_FILE="${DRIVEVLA_ENV_FILE:-${DRIVEVLA_REPO_ROOT}/env.local.sh}"

# Preserve values supplied by the caller so env.local.sh cannot override a
# one-shot launch configuration. DRIVEVLA_REPO_ROOT is intentionally excluded:
# it always identifies this checkout, never an inherited neighboring worktree.
_drivevla_config_variables=(
  DRIVEVLA_ASSET_ROOT
  DRIVEVLA_BASE_CHECKPOINT
  DRIVEVLA_VLM_CONFIG
  DRIVEVLA_DINO_WEIGHTS
  OPENSCENE_DATA_ROOT
  NUPLAN_MAPS_ROOT
  METRIC_CACHE_PATH
  PDMS_METRIC_CACHE_PATH
  EPDMS_METRIC_CACHE_PATH
  DRIVEVLA_NAVSIM_V2_ROOT
  DRIVEVLA_NAVTEST_TOKEN_LIST
  DRIVEVLA_DLC_EVAL_ROOT
  DLC_PROJECT_ROOT
  DLC_OUTPUT_ROOT
  DLC_WORKSPACE_ID
  DLC_RESOURCE_ID
  DLC_WORKER_IMAGE
  DLC_WORKER_SPEC
  DLC_WORKER_GPU_TYPE
  DLC_REGION
  DLC_ENDPOINT
  NAVSIM_ROOT
  NAVSIM_EXP_ROOT
  SUBSCORE_PATH
  PYTHON_BIN
  NUPLAN_MAP_VERSION
  OPENBLAS_CORETYPE
  HYDRA_FULL_ERROR
  HF_HOME
  MPLCONFIGDIR
  TORCH_HOME
  HF_HUB_OFFLINE
  TRANSFORMERS_OFFLINE
  TOKENIZERS_PARALLELISM
  NUM_GPUS
  BATCH_SIZE
  NUM_WORKERS
)
declare -A _drivevla_caller_values=()
declare -A _drivevla_caller_is_set=()
for _drivevla_variable in "${_drivevla_config_variables[@]}"; do
  if [[ -v "${_drivevla_variable}" ]]; then
    _drivevla_caller_is_set["${_drivevla_variable}"]=1
    _drivevla_caller_values["${_drivevla_variable}"]="${!_drivevla_variable}"
  fi
done

if [[ -f "${DRIVEVLA_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${DRIVEVLA_ENV_FILE}"
fi

for _drivevla_variable in "${_drivevla_config_variables[@]}"; do
  if [[ "${_drivevla_caller_is_set[${_drivevla_variable}]:-0}" == "1" ]]; then
    printf -v "${_drivevla_variable}" '%s' \
      "${_drivevla_caller_values[${_drivevla_variable}]}"
    export "${_drivevla_variable}"
  fi
done
unset _drivevla_config_variables _drivevla_caller_values \
  _drivevla_caller_is_set _drivevla_variable

export NAVSIM_ROOT="${NAVSIM_ROOT:-${DRIVEVLA_REPO_ROOT}}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${DRIVEVLA_REPO_ROOT}/outputs}"
export SUBSCORE_PATH="${SUBSCORE_PATH:-${NAVSIM_EXP_ROOT}}"
export PYTHON_BIN="${PYTHON_BIN:-${DRIVEVLA_REPO_ROOT}/.venv/bin/python}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export OPENBLAS_CORETYPE="${OPENBLAS_CORETYPE:-Haswell}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export HF_HOME="${HF_HOME:-${DRIVEVLA_REPO_ROOT}/.cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${DRIVEVLA_REPO_ROOT}/.cache/matplotlib}"
export TORCH_HOME="${TORCH_HOME:-${DRIVEVLA_REPO_ROOT}/.cache/torch}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PDMS_METRIC_CACHE_PATH="${PDMS_METRIC_CACHE_PATH:-${METRIC_CACHE_PATH:-}}"

case ":${PYTHONPATH:-}:" in
  *":${DRIVEVLA_REPO_ROOT}:"*) ;;
  *) export PYTHONPATH="${DRIVEVLA_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
