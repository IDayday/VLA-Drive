#!/usr/bin/env bash
# Train matched projected_residual and gated_residual Stage-A-v2 variants.

set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

devices="${GP_SQ3DMIX_DEVICES:-1}"
batch="${GP_SQ3DMIX_PER_DEVICE_BATCH:-2}"
accumulation="${GP_SQ3DMIX_GRADIENT_ACCUMULATION:-}"
batch_explicit=0
seed=20260824
dry_run=0
while (( $# )); do
  case "$1" in
    --devices) devices="${2:?}"; shift 2 ;;
    --per-device-batch) batch="${2:?}"; batch_explicit=1; shift 2 ;;
    --gradient-accumulation) accumulation="${2:?}"; shift 2 ;;
    --seed) seed="${2:?}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) echo "Usage: bash $project_root/26-train_gp_sq3dmix_stage_a_v2.sh [--devices N --per-device-batch N --gradient-accumulation N] [--dry-run]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
run_root="${GP_SQ3DMIX_V2_RUN_ROOT:-$NAVSIM_EXP_ROOT/gp_sq3dmix_stage_a_v2}"
eval_root="${GP_SQ3DMIX_V2_EVAL_ROOT:-$NAVSIM_EVAL_ROOT/gp_sq3dmix_stage_a_v2_eval}"
for variant in projected_residual gated_residual; do
  report="$eval_root/smoke/$variant/smoke_decision.json"
  [[ -f "$report" ]] || { echo "Missing successful smoke report: $report" >&2; exit 2; }
  python - "$report" "$(git -C "$project_root" rev-parse HEAD)" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
if d.get('all_passed') is not True: raise SystemExit('Stage-A-v2 smoke did not pass')
evaluation=json.load(open(d['evaluation_report']))
if evaluation.get('code_commit') != sys.argv[2]: raise SystemExit('Smoke code commit mismatch')
PY
done
if (( ! batch_explicit )) && [[ "$devices" == 1 && -d "$run_root/gp-sq3dmix-stage-a-v2-smoke-projected_residual-${seed}-batch1" ]]; then
  batch=1
  accumulation=32
  echo "[gp-stage-a-v2] using smoke-confirmed single-PPU batch=1 accumulation=32"
fi
for variant in projected_residual gated_residual; do
  command=(bash "$project_root/tools/launch_gp_sq3dmix_training.sh"
    --phase stage_a_v2 --variant "$variant" --seed "$seed"
    --devices "$devices" --per-device-batch "$batch"
    --run-root "$run_root" --run-id "gp-sq3dmix-stage-a-v2-${variant}-${seed}")
  [[ -z "$accumulation" ]] || command+=(--gradient-accumulation "$accumulation")
  (( dry_run )) && command+=(--dry-run)
  "${command[@]}"
done
