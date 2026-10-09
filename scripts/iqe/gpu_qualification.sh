#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if (( $# != 2 )); then echo 'usage: gpu_qualification.sh EXISTING_SMOKE_CONFIG OUTPUT_DIRECTORY' >&2; exit 2; fi
PYTHON=${IQE_PYTHON:-/root/miniconda3/envs/ddp/bin/python}
"$PYTHON" -c 'from iqe.resources import qualify; print(qualify("cuda"))'
IQE_RUN_GPU_TESTS=1 "$PYTHON" -m pytest tests/iqe/test_training_resume_ddp.py -q -k nccl
PYTHONPATH=. "$PYTHON" scripts/iqe/real_framework_probe.py --config "$1" --output "$2" --device cuda
