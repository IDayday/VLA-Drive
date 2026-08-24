#!/usr/bin/env bash
# Launch two matched GP/control Stage-B seeds only after Stage-A-v2 passes.

set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

decision="${GP_STAGE_A_V2_DECISION:-${GP_SQ3DMIX_V2_EVAL_ROOT:-$NAVSIM_EVAL_ROOT/gp_sq3dmix_stage_a_v2_eval}/stage_a_v2/stage_a_v2_decision.json}"
devices="${GP_SQ3DMIX_DEVICES:-1}"
batch="${GP_SQ3DMIX_PER_DEVICE_BATCH:-2}"
accumulation="${GP_SQ3DMIX_GRADIENT_ACCUMULATION:-}"
dry_run=0
while (( $# )); do
  case "$1" in
    --decision-report) decision="${2:?}"; shift 2 ;;
    --devices) devices="${2:?}"; shift 2 ;;
    --per-device-batch) batch="${2:?}"; shift 2 ;;
    --gradient-accumulation) accumulation="${2:?}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) echo "Usage: bash $project_root/28-train_gp_sq3dmix_stage_b_multiseed.sh [--decision-report FILE] [--devices N --per-device-batch N --gradient-accumulation N] [--dry-run]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -f "$decision" ]] || { echo "Missing Stage-A-v2 decision: $decision" >&2; exit 2; }
readarray -t selected < <(python - "$decision" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
if d.get('all_passed') is not True: raise SystemExit('Stage A did not pass; Stage B forbidden')
print(d['selected_variant']); print(d['selected_checkpoint'])
PY
)
selected_variant="${selected[0]}"
stage_a_checkpoint="${selected[1]}"
[[ -f "$stage_a_checkpoint" ]] || { echo "Selected Stage-A checkpoint is missing" >&2; exit 2; }
run_root="${GP_SQ3DMIX_V2_RUN_ROOT:-$NAVSIM_EXP_ROOT/gp_sq3dmix_stage_a_v2}"
for seed in 20260824 20260825; do
  for variant in "$selected_variant" control; do
    command=(bash "$project_root/tools/launch_gp_sq3dmix_training.sh"
      --phase stage_b --variant "$variant" --seed "$seed"
      --decision-report "$decision" --stage-a-checkpoint "$stage_a_checkpoint"
      --devices "$devices" --per-device-batch "$batch"
      --run-root "$run_root" --run-id "gp-sq3dmix-stage-b-${variant}-${seed}")
    [[ -z "$accumulation" ]] || command+=(--gradient-accumulation "$accumulation")
    (( dry_run )) && command+=(--dry-run)
    "${command[@]}"
  done
done
