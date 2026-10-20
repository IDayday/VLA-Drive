#!/usr/bin/env bash
set -Eeuo pipefail
trainer_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [[ $# -eq 0 ]]; then
  exec python -u "$trainer_root/entry.py" run
fi
case "$1" in
  plan|check|smoke|run) exec python -u "$trainer_root/entry.py" "$@" ;;
  *) exec python -u "$trainer_root/schedule.py" "$@" ;;
esac
