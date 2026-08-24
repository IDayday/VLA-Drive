#!/usr/bin/env bash
# Run and validate matched 100-step projected/gated Stage-A-v2 smoke tests.

set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

devices="${GP_SQ3DMIX_DEVICES:-1}"
batch="${GP_SQ3DMIX_PER_DEVICE_BATCH:-2}"
accumulation="${GP_SQ3DMIX_GRADIENT_ACCUMULATION:-}"
seed=20260824
dry_run=0
while (( $# )); do
  case "$1" in
    --devices) devices="${2:?}"; shift 2 ;;
    --per-device-batch) batch="${2:?}"; shift 2 ;;
    --gradient-accumulation) accumulation="${2:?}"; shift 2 ;;
    --seed) seed="${2:?}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) echo "Usage: bash $project_root/25-smoke_gp_sq3dmix_stage_a_v2.sh [--devices N --per-device-batch N --gradient-accumulation N] [--dry-run]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

run_root="${GP_SQ3DMIX_V2_RUN_ROOT:-$NAVSIM_EXP_ROOT/gp_sq3dmix_stage_a_v2}"
eval_root="${GP_SQ3DMIX_V2_EVAL_ROOT:-$NAVSIM_EVAL_ROOT/gp_sq3dmix_stage_a_v2_eval}/smoke"
for variant in projected_residual gated_residual; do
  run_id="gp-sq3dmix-stage-a-v2-smoke-${variant}-${seed}"
  launch=(bash "$project_root/tools/launch_gp_sq3dmix_training.sh" --phase smoke --variant "$variant" --seed "$seed" --devices "$devices" --per-device-batch "$batch" --run-root "$run_root" --run-id "$run_id")
  [[ -z "$accumulation" ]] || launch+=(--gradient-accumulation "$accumulation")
  (( dry_run )) && launch+=(--dry-run)
  if (( dry_run )); then
    "${launch[@]}"
    continue
  fi
  mkdir -p "$eval_root/$variant"
  log="$eval_root/$variant/training.log"
  [[ ! -e "$log" ]] || { echo "Refusing to overwrite smoke log: $log" >&2; exit 2; }
  set +e
  "${launch[@]}" 2>&1 | tee "$log"
  status="${PIPESTATUS[0]}"
  set -e
  actual_run_id="$run_id"
  if (( status != 0 )); then
    if [[ "$devices" == 1 && "$batch" == 2 ]] && grep -Eqi 'out of memory|resource exhausted|oom' "$log"; then
      echo "[gp-smoke-v2] batch=2 OOM confirmed; retrying batch=1 accumulation=32"
      actual_run_id="${run_id}-batch1"
      bash "$project_root/tools/launch_gp_sq3dmix_training.sh" \
        --phase smoke --variant "$variant" --seed "$seed" \
        --devices 1 --per-device-batch 1 --gradient-accumulation 32 \
        --run-root "$run_root" --run-id "$actual_run_id"
    else
      echo "Stage-A-v2 smoke failed for a non-OOM reason; fallback forbidden" >&2
      exit "$status"
    fi
  fi
  run_dir="$run_root/$actual_run_id"
  checkpoint="$run_dir/final_model/pytorch_model.pt"
  [[ -f "$checkpoint" ]] || { echo "Smoke final checkpoint is missing: $checkpoint" >&2; exit 2; }
  evaluation="$eval_root/$variant/evaluation.json"
  bash "$project_root/27-eval_gp_sq3dmix_stage_a_v2.sh" \
    --run-dir "$run_dir" --checkpoint "$checkpoint" --variant "$variant" \
    --split smoke --bootstrap-draws 1000 --output "$evaluation"
  PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}" python \
    "$project_root/tools/validate_gp_sq3dmix_stage_a_v2_smoke.py" \
    --run-dir "$run_dir" --variant "$variant" \
    --evaluation-report "$evaluation" \
    --output "$eval_root/$variant/smoke_decision.json"
done
