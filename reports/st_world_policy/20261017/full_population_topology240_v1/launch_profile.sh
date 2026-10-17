#!/bin/bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
exec timeout --signal=TERM --kill-after=40s 4000s \
  /root/miniconda3/envs/ddp/bin/python -B "$HERE/controller.py" "$@"
