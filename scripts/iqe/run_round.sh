#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if (( $# < 3 )); then echo 'usage: run_round.sh CONFIG ROUND MAX_SAMPLES [extra CLI arguments]' >&2; exit 2; fi
CONFIG=$1; ROUND=$2; SAMPLES=$3; shift 3
exec "${IQE_PYTHON:-/root/miniconda3/envs/ddp/bin/python}" -m iqe.cli run-round --config "$CONFIG" --round "$ROUND" --mode full --max-samples "$SAMPLES" --device "${IQE_DEVICE:-cuda}" "$@"
