#!/usr/bin/env bash
set -euo pipefail
exec timeout --signal=TERM --kill-after=40s 1800s /root/miniconda3/envs/ddp/bin/python -B "$(dirname "${BASH_SOURCE[0]}")/control.py" "$@"
