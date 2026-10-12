#!/usr/bin/env bash
set -euo pipefail
# Runtime imports may change LC_CTYPE; metadata remains UTF-8 unconditionally.
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false
exec "${IQE_PYTHON:-/root/miniconda3/envs/ddp/bin/python}" -u -m iqe.rolling_cache.cli produce "$@"
