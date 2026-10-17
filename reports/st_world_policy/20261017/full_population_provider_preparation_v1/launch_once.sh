#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
readonly st_provider_entry="$(dirname -- "$(readlink -f -- "$0")")/prepare_once.py"
# Covers Python import/bootstrap too; original owner monitor performs cleanup.
exec timeout --signal=TERM --kill-after=40s 1500s taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python -B "$st_provider_entry" "$@"
