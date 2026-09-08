#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PLANREG_V2_STUDENT:?student-only checkpoint required}"
: "${PLANREG_NAVTEST_LOGS:?official test log directory required}"
: "${PLANREG_NAVTEST_SENSORS:?official test sensor directory required}"
: "${PLANREG_NAVTEST_METRIC_CACHE:?official metric cache required}"
: "${EVAL_OUTPUT:?new output directory required}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD:/mnt/project/DriveVLA-M0-env/lib/python3.9/site-packages:${PYTHONPATH:-}"
PYTHON_BIN=${PYTHON_BIN:-/mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/envs/navsim_py39_exact/bin/python}
COMMAND=("$PYTHON_BIN" navsim/planning/script/run_pdm_score.py
  agent=planreg_wm_v2_student train_test_split=navtest worker=sequential
  "navsim_log_path=$PLANREG_NAVTEST_LOGS" "sensor_blobs_path=$PLANREG_NAVTEST_SENSORS"
  "metric_cache_path=$PLANREG_NAVTEST_METRIC_CACHE" "output_dir=$EVAL_OUTPUT"
  experiment_name=planreg_v2_official_selected)
if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${COMMAND[@]}" "$@"
  printf '\n'
  exit 0
fi
[[ ${LAUNCH_NAVTEST:-0} == 1 ]] || { echo 'Set LAUNCH_NAVTEST=1 explicitly; this task does not authorize automatic full Navtest.' >&2; exit 2; }
[[ ! -e $EVAL_OUTPUT ]] || { echo 'Refusing to reuse an evaluation output directory.' >&2; exit 2; }
exec "${COMMAND[@]}" "$@"
