#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PLANREG_V2_MANIFEST:?input-only V2 manifest required}"
: "${PLANREG_V2_LAYOUT:?new V2 layout lock required}"
: "${PLANREG_V2_SHARED_INIT:?new shared trainable artifact required}"
: "${OUTPUT_DIR:?new output directory required}"
: "${LAUNCH_FORMAL:?set LAUNCH_FORMAL=1 for one explicit run}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD:/mnt/project/DriveVLA-M0-env/lib/python3.9/site-packages:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON_BIN=${PYTHON_BIN:-/mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/envs/navsim_py39_exact/bin/python}
read -r GPUS MICRO ACCUM WORKERS < <("$PYTHON_BIN" -c 'import json,os; c=json.load(open(os.environ["PLANREG_V2_LAYOUT"])); print(c["gpus_per_node"],c["microbatch"],c["accumulate"],c["workers"])')
ARGS=(--config "${V2_CONFIG:-navsim/planning/script/config/common/agent/planreg_wm_v2_base.yaml}" --manifest "$PLANREG_V2_MANIFEST" --output "$OUTPUT_DIR" --layout-lock "$PLANREG_V2_LAYOUT" --microbatch "$MICRO" --accumulate "$ACCUM" --workers "$WORKERS" --seed "${SEED:-0}")
if [[ -n ${RESUME_CHECKPOINT:-} ]]; then ARGS+=(--resume "$RESUME_CHECKPOINT"); fi
exec "$PYTHON_BIN" -m torch.distributed.run --nproc_per_node "$GPUS" --nnodes "${NNODES:-1}" --node_rank "${NODE_RANK:-0}" --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29542}" scripts/train_planreg_v2.py "${ARGS[@]}"
