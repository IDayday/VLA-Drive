#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
readonly st_t0_entry="$(dirname -- "$(readlink -f -- "$0")")/prepare_once.py"
exec timeout --signal=TERM --kill-after=40s 1500s taskset -c 12,13 \
  /root/miniconda3/envs/ddp/bin/python -B "$st_t0_entry" "$@"
