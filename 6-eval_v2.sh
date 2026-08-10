#!/usr/bin/env bash
# Step 6 (eval): Evaluate predictions on NAVSIM v2 (EPDMS).
# Run: source env.sh && bash 6-eval_v2.sh
#
# Optional overrides:
#   SPLIT=test
#   PRED_DIR=/path/to/planning_results/run_name
#   METRIC_CACHE_PATH=/path/to/metric_cache
#   CACHE_WORKERS=2

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/scripts/load_env.sh"

: "${NUPLAN_MAPS_ROOT:?Set NUPLAN_MAPS_ROOT in env.sh}"
: "${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT in env.sh}"
: "${DRIVEDREAMER_ROOT:?Run 'source env.sh' first}"

SPLIT="${SPLIT:-test}"
PRED_DIR="${PRED_DIR:-$DRIVEDREAMER_ROOT/navsim_planning_results/DriveDreamer-Policy}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${NAVSIM_V2_METRIC_CACHE_PATH:-${NAVSIM_V2_METRIC_CACHE_ROOT}/metric_cache_nav${SPLIT}}}"
CACHE_WORKERS="${CACHE_WORKERS:-2}"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NAVSIM_EXP_ROOT="${NAVSIM_EVAL_ROOT:-$DRIVEDREAMER_ROOT/navsim_exp/eval_v2}"
export NAVSIM_DEVKIT_ROOT="$DRIVEDREAMER_ROOT/navsim"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export SPLIT PRED_DIR METRIC_CACHE_PATH

if [[ ! -d "$PRED_DIR/$SPLIT" ]]; then
    echo "Prediction directory does not exist: $PRED_DIR/$SPLIT" >&2
    exit 1
fi

prediction_count="$(find "$PRED_DIR/$SPLIT" -maxdepth 1 -type f -name '*.npy' | wc -l)"
if [[ "$prediction_count" -eq 0 ]]; then
    echo "No predictions found under $PRED_DIR/$SPLIT" >&2
    exit 1
fi

datalist="$DRIVEDREAMER_ROOT/${SPLIT}_meta.json"
if [[ -f "$datalist" ]]; then
    expected_count="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$datalist")"
    if [[ "$prediction_count" -ne "$expected_count" ]]; then
        echo "Prediction count mismatch: found $prediction_count, expected $expected_count" >&2
        exit 1
    fi
fi

cd "$NAVSIM_DEVKIT_ROOT/scripts/evaluation"

if ! find "$METRIC_CACHE_PATH" -type f -print -quit 2>/dev/null | grep -q .; then
    echo "Metric cache is empty; generating nav${SPLIT} cache at $METRIC_CACHE_PATH"
    TRAIN_TEST_SPLIT="nav${SPLIT}" \
    CACHE_PATH="$METRIC_CACHE_PATH" \
    CACHE_WORKERS="$CACHE_WORKERS" \
        bash ./run_metric_caching.sh
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    bash ./run_human_agent_pdm_score_evaluation.sh
