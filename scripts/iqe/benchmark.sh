#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if (( $# < 2 )); then echo 'usage: benchmark.sh CONFIG ROUND [extra CLI arguments]' >&2; exit 2; fi
CONFIG=$1; ROUND=$2; shift 2
exec "${IQE_PYTHON:-/root/miniconda3/envs/ddp/bin/python}" -m iqe.cli benchmark --config "$CONFIG" --round "$ROUND" --mode profile --max-samples 32 --device "${IQE_DEVICE:-cuda}" --expert-counts 1 2 3 5 "$@"
