#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
IQE_REPO=$(pwd)
export PYTHONPATH="${IQE_REPO}:${IQE_REPO}/nuplan-devkit${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IQE_METRIC_PY=${IQE_METRIC_PYTHON:-/root/miniconda3/envs/navsim/bin/python}
"$IQE_METRIC_PY" -c 'import nuplan,pathlib; expected=pathlib.Path("nuplan-devkit/nuplan").resolve(); assert pathlib.Path(nuplan.__file__).parent.resolve()==expected; print("official nuplan:",expected,flush=True)'
exec "$IQE_METRIC_PY" -u -m iqe.navtest score "$@"
