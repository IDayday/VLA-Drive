#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../load_env.sh
source "${REPO_ROOT}/load_env.sh"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  echo "Run ${REPO_ROOT}/scripts/setup_local.sh first." >&2
  exit 2
fi

mkdir -p "${MPLCONFIGDIR}" "${HF_HOME}" "${TORCH_HOME}"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/preflight_base.py" "$@"
