#!/usr/bin/env bash
# Evaluate saved trajectories with the official NAVSIM v1.1 PDM-Score code.
# Example:
#   PRED_DIR=/path/to/run SPLIT=test bash 5-eval_v1.sh

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

: "${NUPLAN_MAPS_ROOT:?Set NUPLAN_MAPS_ROOT in env.sh}"
: "${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT in env.sh}"
: "${DRIVEDREAMER_ROOT:?Run 'source env.sh' first}"

SPLIT="${SPLIT:-test}"
PRED_DIR="${PRED_DIR:-$DRIVEDREAMER_ROOT/navsim_planning_results/pytorch_model.pt}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${NAVSIM_V1_METRIC_CACHE_PATH}}"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NAVSIM_EXP_ROOT="${NAVSIM_EVAL_ROOT:-$DRIVEDREAMER_ROOT/navsim_exp/eval_v1.1}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_V1_DEVKIT_ROOT:-$DRIVEDREAMER_ROOT/navsim_v1.1/navsim}"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"

prediction_dir="$PRED_DIR/test"
if [[ ! -d "$prediction_dir" ]]; then
  echo "Prediction directory does not exist: $prediction_dir" >&2
  exit 1
fi
prediction_count="$(find "$prediction_dir" -maxdepth 1 -type f -name '*.npy' | wc -l)"
expected_count="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$DRIVEDREAMER_ROOT/${SPLIT}_meta.json")"
if [[ "$prediction_count" -ne "$expected_count" ]]; then
  echo "Prediction count mismatch: found $prediction_count, expected $expected_count" >&2
  exit 1
fi
if [[ ! -d "$METRIC_CACHE_PATH/metadata" ]]; then
  echo "NAVSIM v1.1 metric cache is missing: $METRIC_CACHE_PATH" >&2
  exit 1
fi

mkdir -p "$NAVSIM_EXP_ROOT"
python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
  train_test_split="nav${SPLIT}" \
  metric_cache_path="$METRIC_CACHE_PATH" \
  agent=human_agent \
  experiment_name=drivedreamer-policy \
  pred_dir="$PRED_DIR" \
  split="$SPLIT"
