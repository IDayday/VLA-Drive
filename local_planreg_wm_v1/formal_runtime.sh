#!/usr/bin/env bash

# Host-local /root environments differ between vla-zt and vla-zt2.  Use the
# genuinely shared Python 3.9 executable and shared package tree for every
# formal benchmark and training launcher.
PLANREG_FORMAL_PYTHON_BIN="${PLANREG_FORMAL_PYTHON_BIN:-/mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/envs/navsim_py39_exact/bin/python}"
PLANREG_FORMAL_SITE_PACKAGES="${PLANREG_FORMAL_SITE_PACKAGES:-/mnt/project/DriveVLA-M0-env/lib/python3.9/site-packages}"

planreg_formal_runtime_setup() {
  local repo_root="$1"
  [[ -x "${PLANREG_FORMAL_PYTHON_BIN}" ]] || {
    echo "Formal shared Python is missing: ${PLANREG_FORMAL_PYTHON_BIN}" >&2
    return 2
  }
  [[ -d "${PLANREG_FORMAL_SITE_PACKAGES}" ]] || {
    echo "Formal shared site-packages are missing: ${PLANREG_FORMAL_SITE_PACKAGES}" >&2
    return 2
  }
  PLANREG_FORMAL_PYTHONPATH="${repo_root}:${PLANREG_FORMAL_SITE_PACKAGES}"
  export PLANREG_FORMAL_PYTHON_BIN PLANREG_FORMAL_SITE_PACKAGES
  export PLANREG_FORMAL_PYTHONPATH
  export PYTHONPATH="${PLANREG_FORMAL_PYTHONPATH}"
  export PYTHONNOUSERSITE=1
}

planreg_formal_runtime_audit_local() {
  local repo_root="$1"
  local output="$2"
  repo_root="$(realpath "${repo_root}")"
  output="$(realpath -m "${output}")"
  env PYTHONPATH="${PLANREG_FORMAL_PYTHONPATH}" PYTHONNOUSERSITE=1 \
    "HF_HOME=${HF_HOME}" "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}" \
    "HF_HUB_OFFLINE=${HF_HUB_OFFLINE}" \
    "PLANREG_FORMAL_VLM_PATH=${PLANREG_FORMAL_VLM_PATH:-}" \
    "${PLANREG_FORMAL_PYTHON_BIN}" \
    "${repo_root}/scripts/audit_formal_runtime_environment.py" \
    --repo-root "${repo_root}" --output "${output}" >/dev/null
}

planreg_formal_runtime_audit_remote() {
  local host="$1"
  local repo_root="$2"
  local output="$3"
  repo_root="$(realpath "${repo_root}")"
  output="$(realpath -m "${output}")"
  local command
  printf -v command '%q ' env \
    "PYTHONPATH=${PLANREG_FORMAL_PYTHONPATH}" PYTHONNOUSERSITE=1 \
    "HF_HOME=${HF_HOME}" "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}" \
    "HF_HUB_OFFLINE=${HF_HUB_OFFLINE}" \
    "PLANREG_FORMAL_VLM_PATH=${PLANREG_FORMAL_VLM_PATH:-}" \
    "${PLANREG_FORMAL_PYTHON_BIN}" \
    "${repo_root}/scripts/audit_formal_runtime_environment.py" \
    --repo-root "${repo_root}" --output "${output}"
  ssh -o BatchMode=yes "${host}" "${command}" >/dev/null
}

planreg_formal_runtime_compare() {
  local repo_root="$1"
  local left="$2"
  local right="$3"
  local output="$4"
  repo_root="$(realpath "${repo_root}")"
  left="$(realpath "${left}")"
  right="$(realpath "${right}")"
  output="$(realpath -m "${output}")"
  env PYTHONPATH="${PLANREG_FORMAL_PYTHONPATH}" PYTHONNOUSERSITE=1 \
    "${PLANREG_FORMAL_PYTHON_BIN}" \
    "${repo_root}/scripts/audit_formal_runtime_environment.py" \
    --compare "${left}" "${right}" --output "${output}" >/dev/null
}
