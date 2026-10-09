#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if (( $# != 5 )); then echo 'usage: qualified_base_training.sh CONFIG PROFILE_CONFIG MAX_SAMPLES NPROC MAX_SECONDS_PER_UPDATE' >&2; exit 2; fi
CONFIG=$1; PROFILE=$2; SAMPLES=$3; NPROC=$4; MAX_SECONDS=$5
PYTHON=${IQE_PYTHON:-/root/miniconda3/envs/ddp/bin/python}
# Each CLI checks current AGENTS/policy and refuses occupied GPUs. No evictions/retries.
"$PYTHON" -m iqe.cli preflight --config "$PROFILE" --repo . --device cpu
"$PYTHON" -m iqe.cli build-splits --config "$PROFILE" --mode profile --max-samples 128 --device cpu
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" -m iqe.cli train-base --config "$PROFILE" --mode profile --max-samples 128 --max-steps 8 --device cuda
PROFILE_DIR=$("$PYTHON" -c 'import sys;from iqe.config import load_config;print(load_config(sys.argv[1])["output_root"]+"/query_base")' "$PROFILE")
"$PYTHON" -m iqe.training.efficiency --directory "$PROFILE_DIR" --warmup 3 --max-seconds-per-update "$MAX_SECONDS"
# Profile success qualifies only throughput. The formal job has its own immutable configuration and schedule.
"$PYTHON" -m iqe.cli preflight --config "$CONFIG" --repo . --device cpu
"$PYTHON" -m iqe.cli build-splits --config "$CONFIG" --mode full --max-samples "$SAMPLES" --device cpu
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" -m iqe.cli train-base --config "$CONFIG" --mode full --max-samples "$SAMPLES" --device cuda
