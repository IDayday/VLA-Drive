#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../load_env.sh
source "${REPO_ROOT}/load_env.sh"

VENV_DIR="${DRIVEVLA_VENV:-${REPO_ROOT}/.venv}"
SYSTEM_PYTHON="${SYSTEM_PYTHON:-python}"

# The PPU image provides a vendor-adapted PyTorch stack. Inherit it instead of
# resolving requirements that could replace torch, torchvision, or triton.
"${SYSTEM_PYTHON}" "${SCRIPT_DIR}/verify_runtime_versions.py"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${SYSTEM_PYTHON}" -m venv --system-site-packages "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install \
  --disable-pip-version-check \
  --no-build-isolation \
  --no-deps \
  -e "${REPO_ROOT}"

"${VENV_DIR}/bin/python" "${SCRIPT_DIR}/verify_runtime_versions.py"
"${VENV_DIR}/bin/python" -c "import navsim; print('DriveVLA-M0 editable import:', navsim.__file__)"

echo "Local PPU-compatible environment is ready: ${VENV_DIR}"
