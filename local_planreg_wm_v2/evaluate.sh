#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PLANREG_V2_STUDENT:?student-only checkpoint required}"
: "${PLANREG_V2_EVAL_MANIFEST:?read-only evaluation input manifest required}"
: "${EVAL_OUTPUT:?new report file required}"
export PYTHONPATH="$PWD:/mnt/project/DriveVLA-M0-env/lib/python3.9/site-packages:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHON_BIN=${PYTHON_BIN:-/mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/envs/navsim_py39_exact/bin/python}
exec "$PYTHON_BIN" scripts/evaluate_planreg_v2.py --checkpoint "$PLANREG_V2_STUDENT" --manifest "$PLANREG_V2_EVAL_MANIFEST" --output "$EVAL_OUTPUT" "$@"
