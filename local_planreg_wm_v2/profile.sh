#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PLANREG_V2_MANIFEST:?new progressive-long input cache required}"
: "${PLANREG_V2_SHARED_INIT:?new std0.02 shared parameter bank required}"
: "${PLANREG_V2_NORMALIZER:?new measured raw-GT statistics required}"
: "${OUTPUT_DIR:?new bounded profile output directory required}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD:/mnt/project/DriveVLA-M0-env/lib/python3.9/site-packages:${PYTHONPATH:-}"
PYTHON_BIN=${PYTHON_BIN:-/mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/envs/navsim_py39_exact/bin/python}
# Default is only a CANDIDATE layout; this is not a declaration it fits.
WORLD=${WORLD:-8}
MICRO=${MICRO:-1}
ACCUM=${ACCUM:-16}
[[ $((WORLD * MICRO * ACCUM)) == 128 ]] || { echo 'Profile requires actual GB128' >&2; exit 2; }
exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$WORLD" \
  scripts/train_planreg_v2.py --config "${V2_CONFIG:-navsim/planning/script/config/common/agent/planreg_wm_v2_base.yaml}" \
  --manifest "$PLANREG_V2_MANIFEST" --output "$OUTPUT_DIR" --profile-only \
  --microbatch "$MICRO" --accumulate "$ACCUM" --workers "${WORKERS:-2}" --seed "${SEED:-0}"
