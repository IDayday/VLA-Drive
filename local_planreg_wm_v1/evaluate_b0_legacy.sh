#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
source_checkpoint="${1:-${PLANREG_BASE_CHECKPOINT}}"
student_checkpoint="${PLANREG_B0_STUDENT_CHECKPOINT:-${PLANREG_RUN_ROOT}/exports/b0_legacy_student.ckpt}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  planreg_print_command "${PYTHON_BIN}" \
    "${PLANREG_REPO_ROOT}/scripts/export_planreg_student_checkpoint.py" \
    "${source_checkpoint}" "${student_checkpoint}"
  planreg_print_command bash "${PLANREG_SCRIPT_DIR}/evaluate_all.sh" \
    "b0_legacy_seed0=${student_checkpoint}"
  exit 0
fi
planreg_require_file "${source_checkpoint}"
mkdir -p "$(dirname "${student_checkpoint}")"
"${PYTHON_BIN}" "${PLANREG_REPO_ROOT}/scripts/export_planreg_student_checkpoint.py" \
  "${source_checkpoint}" "${student_checkpoint}"
bash "${PLANREG_SCRIPT_DIR}/evaluate_all.sh" \
  "b0_legacy_seed0=${student_checkpoint}"
