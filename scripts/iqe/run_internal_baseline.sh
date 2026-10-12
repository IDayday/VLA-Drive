#!/usr/bin/env bash
set -euo pipefail
cd /mnt/project/DriveVLA-M0-iqe-internal-scorer
IQE_CONFIG=${1:-configs/iqe/internal_baseline/own_wa_v1.yaml}
IQE_PYTHON=/root/miniconda3/envs/ddp/bin/python
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
"$IQE_PYTHON" -m iqe.internal_baseline.cli run --config "$IQE_CONFIG" --device cuda:0 --wait-cache-seconds 10800
