#!/usr/bin/env bash
set -euo pipefail
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false
exec "${IQE_PYTHON:-/root/miniconda3/envs/ddp/bin/python}" -u -m torch.distributed.run \
  --standalone --nproc_per_node=8 -m iqe.rolling_cache.cli "$@"
