#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
readonly st_t0_profile_entry="$(dirname -- "$(readlink -f -- "$0")")/controller.py"
exec timeout --signal=TERM --kill-after=180s 3900s \
  /root/miniconda3/envs/ddp/bin/python -B "$st_t0_profile_entry" "$@"
