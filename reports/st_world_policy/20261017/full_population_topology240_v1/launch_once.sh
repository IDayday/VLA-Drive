#!/bin/bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
exec timeout --signal=TERM --kill-after=40s 1540s taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python -B "$HERE/prepare_once.py" "$@"
