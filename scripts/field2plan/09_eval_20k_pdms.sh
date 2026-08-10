#!/usr/bin/env bash
# Queue-compatible NAVSIM v1.1 PDMS evaluation for Phase-2 step-20k weights.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/env.sh"

SHARED_ROOT="${DRIVEDREAMER_SHARED_ROOT:-/mnt/zhangt_workspace/project/DriveDreamer-Policy}"

MODEL_ITER=20000 \
PRED_ROOT="${PRED_ROOT:-${SHARED_ROOT}/navsim_planning_results/field2plan_step20k_seed20260808}" \
EVAL_ROOT="${EVAL_ROOT:-${SHARED_ROOT}/navsim_exp/eval_field2plan_step20k_pdms_v1_1}" \
LOG_ROOT="${LOG_ROOT:-${SHARED_ROOT}/navsim_exp/field2plan_step20k_pdms_logs}" \
  bash "${SCRIPT_DIR}/08_eval_10k_pdms.sh"
