#!/usr/bin/env bash
# Select one Phase-3 dynamics-prior experiment per non-interactive DLC job.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
experiment="${FIELD2PLAN_PHASE3_EXPERIMENT:-}"
run_seed="${FIELD2PLAN_RUN_SEED:-42}"
if ! [[ "$run_seed" =~ ^[0-9]+$ ]]; then
  echo "[field2plan-phase3-matrix] FIELD2PLAN_RUN_SEED must be non-negative" >&2
  exit 2
fi

case "$experiment" in
  p3_dyn_nosup_noaccess)
    geometry_supervision=0
    geometry_access=0
    dynamics_supervision=0
    dynamics_access=0
    dynamics_teacher_mode=equal_capacity
    ;;
  p3_dyn_only_real)
    geometry_supervision=0
    geometry_access=0
    dynamics_supervision=1
    dynamics_access=1
    dynamics_teacher_mode=real
    ;;
  p3_geo_dyn_real)
    geometry_supervision=1
    geometry_access=1
    dynamics_supervision=1
    dynamics_access=1
    dynamics_teacher_mode=real
    ;;
  p3_dyn_sup_noaccess)
    geometry_supervision=0
    geometry_access=0
    dynamics_supervision=1
    dynamics_access=0
    dynamics_teacher_mode=real
    ;;
  p3_dyn_access_nosup)
    geometry_supervision=0
    geometry_access=0
    dynamics_supervision=0
    dynamics_access=1
    dynamics_teacher_mode=equal_capacity
    ;;
  p3_geo_dyn_temporal_shuffle)
    geometry_supervision=1
    geometry_access=1
    dynamics_supervision=1
    dynamics_access=1
    dynamics_teacher_mode=temporal_shuffled
    ;;
  p3_geo_dyn_batch_shuffle)
    geometry_supervision=1
    geometry_access=1
    dynamics_supervision=1
    dynamics_access=1
    dynamics_teacher_mode=batch_shuffled
    ;;
  *)
    echo "[field2plan-phase3-matrix] unsupported FIELD2PLAN_PHASE3_EXPERIMENT=$experiment" >&2
    echo "[field2plan-phase3-matrix] choose p3_dyn_nosup_noaccess, p3_dyn_only_real, p3_geo_dyn_real, p3_dyn_sup_noaccess, p3_dyn_access_nosup, p3_geo_dyn_temporal_shuffle, or p3_geo_dyn_batch_shuffle" >&2
    exit 2
    ;;
esac

export FIELD2PLAN_GEOMETRY_SUPERVISION="$geometry_supervision"
export FIELD2PLAN_GEOMETRY_ACCESS="$geometry_access"
export FIELD2PLAN_DYNAMICS_SUPERVISION="$dynamics_supervision"
export FIELD2PLAN_DYNAMICS_ACCESS="$dynamics_access"
export FIELD2PLAN_DYNAMICS_TEACHER_MODE="$dynamics_teacher_mode"
export FIELD2PLAN_RUN_SEED="$run_seed"
max_train_steps="${MAX_TRAIN_STEPS:-100000}"
export RUN_ID="${RUN_ID:-field2plan-${experiment}-steps${max_train_steps}-seed${run_seed}}"

if [ "${FIELD2PLAN_MATRIX_PRINT_ONLY:-0}" = "1" ]; then
  printf '%s\n' \
    "experiment=$experiment" \
    "geometry_supervision=$geometry_supervision" \
    "geometry_access=$geometry_access" \
    "dynamics_supervision=$dynamics_supervision" \
    "dynamics_access=$dynamics_access" \
    "dynamics_teacher_mode=$dynamics_teacher_mode" \
    "run_seed=$run_seed" \
    "run_id=$RUN_ID"
  exit 0
fi

exec bash "$project_root/scripts/field2plan/13_train_dynamics.sh"
