#!/usr/bin/env bash
# Select exactly one Phase-2 scientific control per non-interactive DLC job.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
experiment="${FIELD2PLAN_EXPERIMENT:-}"
run_seed="${FIELD2PLAN_RUN_SEED:-42}"

if ! [[ "$run_seed" =~ ^[0-9]+$ ]]; then
  echo "[field2plan-matrix] FIELD2PLAN_RUN_SEED must be a non-negative integer" >&2
  exit 2
fi

# The first two digits encode supervision/access.  Every no-teacher arm keeps
# the auxiliary head when needed so parameter capacity remains controlled.
case "$experiment" in
  p2_00_nosup_noaccess)
    teacher_type=da3
    supervision=0
    disable_access=1
    teacher_mode=equal_capacity
    ;;
  p2_10_sup_noaccess_da3)
    teacher_type=da3
    supervision=1
    disable_access=1
    teacher_mode=real
    ;;
  p2_01_nosup_access)
    teacher_type=da3
    supervision=0
    disable_access=0
    teacher_mode=equal_capacity
    ;;
  p2_11_sup_access_da3)
    teacher_type=da3
    supervision=1
    disable_access=0
    teacher_mode=real
    ;;
  p2_11_sup_access_vggt)
    teacher_type=vggt
    supervision=1
    disable_access=0
    teacher_mode=real
    ;;
  p2_random_access_da3)
    teacher_type=da3
    supervision=1
    disable_access=0
    teacher_mode=random
    ;;
  p2_shuffled_access_da3)
    teacher_type=da3
    supervision=1
    disable_access=0
    teacher_mode=shuffled
    ;;
  p2_state_mlp_access)
    teacher_type=da3
    supervision=0
    disable_access=0
    teacher_mode=gt_mlp
    ;;
  *)
    echo "[field2plan-matrix] unsupported FIELD2PLAN_EXPERIMENT=$experiment" >&2
    echo "[field2plan-matrix] choose p2_00_nosup_noaccess, p2_10_sup_noaccess_da3, p2_01_nosup_access, p2_11_sup_access_da3, p2_11_sup_access_vggt, p2_random_access_da3, p2_shuffled_access_da3, or p2_state_mlp_access" >&2
    exit 2
    ;;
esac

export FIELD2PLAN_GEOMETRY_TEACHER_TYPE="$teacher_type"
export FIELD2PLAN_GEOMETRY_SUPERVISION="$supervision"
export FIELD2PLAN_DISABLE_ACCESS="$disable_access"
export FIELD2PLAN_TEACHER_MODE="$teacher_mode"
export FIELD2PLAN_RUN_SEED="$run_seed"
max_train_steps="${MAX_TRAIN_STEPS:-100000}"
export RUN_ID="${RUN_ID:-field2plan-${experiment}-steps${max_train_steps}-seed${run_seed}}"

if [ "${FIELD2PLAN_MATRIX_PRINT_ONLY:-0}" = "1" ]; then
  printf '%s\n' \
    "experiment=$experiment" \
    "teacher_type=$teacher_type" \
    "supervision=$supervision" \
    "disable_access=$disable_access" \
    "teacher_mode=$teacher_mode" \
    "run_seed=$run_seed" \
    "run_id=$RUN_ID"
  exit 0
fi

exec bash "$project_root/scripts/field2plan/05_train_geometry.sh"
