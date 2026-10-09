#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if (( $# < 4 )); then echo 'usage: evaluate_bundle.sh CONFIG ROUND BUNDLE MAX_SAMPLES' >&2; exit 2; fi
PYTHON=${IQE_PYTHON:-/root/miniconda3/envs/ddp/bin/python}
"$PYTHON" -m iqe.cli load-bundle --config "$1" --bundle "$3" --mode full --max-samples "$4" --device "${IQE_DEVICE:-cuda}"
"$PYTHON" -m iqe.cli evaluate --config "$1" --round "$2" --role dev_report --mode full --max-samples "$4" --device "${IQE_DEVICE:-cuda}"
