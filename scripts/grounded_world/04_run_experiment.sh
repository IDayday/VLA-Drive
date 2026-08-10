#!/usr/bin/env bash
# Resolve one revised GroundedWorld B0-B5/control arm, then launch its stage.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
experiment="${GROUNDEDWORLD_EXPERIMENT:-}"
stage="${GROUNDEDWORLD_STAGE:-stage3}"
run_seed="${GROUNDEDWORLD_RUN_SEED:-42}"
stage3_phase="${GROUNDEDWORLD_STAGE3_PHASE:-A}"

if ! [[ "$run_seed" =~ ^[0-9]+$ ]]; then
  echo "[groundedworld-matrix] GROUNDEDWORLD_RUN_SEED must be non-negative" >&2
  exit 2
fi

future_target=student_ema
teacher_mode=real
stage3_direct_init="${GROUNDEDWORLD_STAGE3_DIRECT_INIT:-0}"
case "$experiment" in
  b0_pure_vlm_dit)
    external_prior=none; future_enabled=0; world_access=0; refiner_enabled=0; consequence_enabled=0 ;;
  b1_geometry_aux)
    external_prior=vggt; future_enabled=0; world_access=0; refiner_enabled=0; consequence_enabled=0 ;;
  b2_geometry_access)
    external_prior=vggt; future_enabled=0; world_access=1; refiner_enabled=1; consequence_enabled=0 ;;
  b3_current_world)
    external_prior=vggt_driving_jepa; future_enabled=0; world_access=1; refiner_enabled=1; consequence_enabled=0 ;;
  b4_predictive_world)
    external_prior=vggt_driving_jepa; future_enabled=1; world_access=1; refiner_enabled=1; consequence_enabled=0 ;;
  b5_full)
    external_prior=vggt_driving_jepa; future_enabled=1; world_access=1; refiner_enabled=1; consequence_enabled=1 ;;
  control_real_sup_access)
    external_prior=vggt_driving_jepa; future_enabled=1; world_access=1; refiner_enabled=1; consequence_enabled=1 ;;
  control_no_teacher_same_future)
    external_prior=none; teacher_mode=none; future_enabled=1; world_access=1; refiner_enabled=1; consequence_enabled=1 ;;
  control_scene_shuffled_same_future)
    external_prior=vggt_driving_jepa; teacher_mode=scene_shuffled; future_enabled=1; world_access=1; refiner_enabled=1; consequence_enabled=1 ;;
  control_real_sup_noaccess)
    external_prior=vggt_driving_jepa; future_enabled=1; world_access=0; refiner_enabled=1; consequence_enabled=1 ;;
  control_random_frozen_same_future)
    external_prior=vggt_random_frozen; teacher_mode=random; future_enabled=1; world_access=1; refiner_enabled=1; consequence_enabled=1 ;;
  control_gt_task_mlp_same_future)
    external_prior=vggt_gt_task_mlp; teacher_mode=gt_task_mlp; future_enabled=1; world_access=1; refiner_enabled=1; consequence_enabled=1 ;;
  control_generic_vjepa_same_future)
    external_prior=vggt_generic_vjepa; future_enabled=1; world_access=1; refiner_enabled=1; consequence_enabled=1 ;;
  *)
    echo "[groundedworld-matrix] unsupported GROUNDEDWORLD_EXPERIMENT=$experiment" >&2
    exit 2 ;;
esac

if [ "$stage" = "stage3" ] && [ "$stage3_phase" = "B" ] \
  && [ "$experiment" = "b1_geometry_aux" ] \
  && [ -z "${GROUNDEDWORLD_STAGE3_DIRECT_INIT+x}" ]; then
  stage3_direct_init=1
fi
if [ "$stage3_direct_init" != "0" ] && [ "$stage3_direct_init" != "1" ]; then
  echo "[groundedworld-matrix] GROUNDEDWORLD_STAGE3_DIRECT_INIT must be 0 or 1" >&2
  exit 2
fi

case "$stage" in
  stage1|stage2|stage3) ;;
  *) echo "[groundedworld-matrix] GROUNDEDWORLD_STAGE must be stage1, stage2, or stage3" >&2; exit 2 ;;
esac

run_id="${RUN_ID:-groundedworld-${experiment}-${stage}-seed${run_seed}}"
export GROUNDEDWORLD_EXTERNAL_PRIOR="$external_prior"
export GROUNDEDWORLD_TEACHER_MODE="$teacher_mode"
export GROUNDEDWORLD_FUTURE_ENABLED="$future_enabled"
export GROUNDEDWORLD_WORLD_ACCESS="$world_access"
export GROUNDEDWORLD_REFINER_ENABLED="$refiner_enabled"
export GROUNDEDWORLD_CONSEQUENCE_ENABLED="$consequence_enabled"
export GROUNDEDWORLD_STAGE3_DIRECT_INIT="$stage3_direct_init"
export GROUNDEDWORLD_RUN_SEED="$run_seed"
export RUN_ID="$run_id"

if [ "${GROUNDEDWORLD_MATRIX_PRINT_ONLY:-0}" = "1" ]; then
  printf '%s\n' \
    "experiment=$experiment" \
    "stage=$stage" \
    "external_prior=$external_prior" \
    "teacher_mode=$teacher_mode" \
    "future_enabled=$future_enabled" \
    "future_target=$future_target" \
    "world_access=$world_access" \
    "refiner_enabled=$refiner_enabled" \
    "consequence_enabled=$consequence_enabled" \
    "stage3_direct_init=$stage3_direct_init" \
    "run_seed=$run_seed" \
    "run_id=$run_id"
  exit 0
fi

if [ "$stage" = "stage3" ] && [ "$experiment" = "b0_pure_vlm_dit" ]; then
  echo "[groundedworld-matrix] B0 is the fixed pure baseline: evaluate its supplied checkpoint; do not retrain it through GroundedWorld" >&2
  exit 2
fi
if [ "$stage" = "stage3" ] && [ "$experiment" = "b1_geometry_aux" ] \
  && [ "$stage3_phase" = "A" ]; then
  echo "[groundedworld-matrix] B1 has no reader/refiner parameters for Phase A; launch Phase B with direct_init=1" >&2
  exit 2
fi

exec bash "$project_root/scripts/grounded_world/0${stage#stage}_train_${stage}.sh"
