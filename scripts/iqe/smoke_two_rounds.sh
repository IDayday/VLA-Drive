#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON=${IQE_PYTHON:-/root/miniconda3/envs/ddp/bin/python}
CONFIG=${1:-configs/iqe/smoke.yaml}
DEVICE=${IQE_DEVICE:-cuda}
STEPS=${IQE_SMOKE_STEPS:-32}
SAMPLES=${IQE_SMOKE_SAMPLES:-32}
if (( STEPS < 1 || STEPS > 32 || SAMPLES < 32 || SAMPLES > 128 )); then
  echo '{"status":"FAILED","error":"smoke budgets: 1..32 updates and 32..128 scenes per role"}' >&2; exit 2
fi
cpu() { "$PYTHON" -m iqe.cli "$@" --config "$CONFIG" --mode smoke --max-samples "$SAMPLES" --device cpu; }
run() { "$PYTHON" -m iqe.cli "$@" --config "$CONFIG" --mode smoke --max-samples "$SAMPLES" --device "$DEVICE"; }
cpu preflight --repo .
cpu build-splits
cpu prepare-metric-contexts --roles incremental_fit stage_val selector_cal
cpu verify-reference
run train-base --max-steps "$STEPS"
BASE=$("$PYTHON" -c 'import sys;from pathlib import Path;from iqe.config import load_config;from iqe.io import read_json;c=load_config(sys.argv[1]);print(read_json(Path(c["output_root"])/"query_base/result.json")["checkpoint"])' "$CONFIG")
cpu freeze-base --checkpoint "$BASE"
run cache-features --roles incremental_fit stage_val selector_cal dev_report
for ROUND in 1 2; do
  run run-round --round "$ROUND" --max-steps "$STEPS" --resume
 done
# Temporary bundles are restored for a real inference compatibility check. No ACTIVE update/navtest.
BUNDLE=$("$PYTHON" -c 'import sys;from iqe.config import load_config;print(load_config(sys.argv[1])["output_root"]+"/rounds/round_002/bundle")' "$CONFIG")
run load-bundle --bundle "$BUNDLE"
