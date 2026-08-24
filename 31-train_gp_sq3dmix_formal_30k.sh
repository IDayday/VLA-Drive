#!/usr/bin/env bash
# Launch Stage-C in a mandatory 0-10k / gated 10-30k sequence.

set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

permission="${GP_FORMAL_PERMISSION_REPORT:-${GP_SQ3DMIX_V2_EVAL_ROOT:-$NAVSIM_EVAL_ROOT/gp_sq3dmix_stage_a_v2_eval}/stage_b_full_navtest/formal_training_permission.json}"
stage_a_decision="${GP_STAGE_A_V2_DECISION:-${GP_SQ3DMIX_V2_EVAL_ROOT:-$NAVSIM_EVAL_ROOT/gp_sq3dmix_stage_a_v2_eval}/stage_a_v2/stage_a_v2_decision.json}"
interim_decision="${GP_FORMAL_INTERIM_DECISION:-}"
prior_seed_decision="${GP_FORMAL_PRIOR_SEED_DECISION:-}"
seed=20260826
segment=10k
devices="${GP_SQ3DMIX_DEVICES:-1}"
batch="${GP_SQ3DMIX_PER_DEVICE_BATCH:-2}"
accumulation="${GP_SQ3DMIX_GRADIENT_ACCUMULATION:-}"
dry_run=0
while (( $# )); do
  case "$1" in
    --permission-report) permission="${2:?}"; shift 2 ;;
    --stage-a-decision) stage_a_decision="${2:?}"; shift 2 ;;
    --interim-decision) interim_decision="${2:?}"; shift 2 ;;
    --prior-seed-decision) prior_seed_decision="${2:?}"; shift 2 ;;
    --seed) seed="${2:?}"; shift 2 ;;
    --segment) segment="${2:?}"; shift 2 ;;
    --devices) devices="${2:?}"; shift 2 ;;
    --per-device-batch) batch="${2:?}"; shift 2 ;;
    --gradient-accumulation) accumulation="${2:?}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) echo "Usage: bash $project_root/31-train_gp_sq3dmix_formal_30k.sh --segment 10k|30k [--seed 20260826|20260827] [--interim-decision JSON] [--dry-run]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$seed" == 20260826 || "$seed" == 20260827 ]] || { echo "Formal seed must be 20260826 or 20260827" >&2; exit 2; }
[[ "$segment" == 10k || "$segment" == 30k ]] || { echo "--segment must be 10k or 30k" >&2; exit 2; }
for path in "$permission" "$stage_a_decision"; do [[ -f "$path" ]] || { echo "Missing formal binding: $path" >&2; exit 2; }; done
python - "$permission" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
if d.get('formal_30k_allowed') is not True: raise SystemExit('formal_30k_allowed is false; Stage C forbidden')
if d.get('formal_100k_allowed') is not False: raise SystemExit('30k launch requires formal_100k_allowed=false')
PY
readarray -t selected < <(python - "$stage_a_decision" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
if d.get('all_passed') is not True: raise SystemExit('Stage A did not pass')
print(d['selected_variant']); print(d['selected_checkpoint'])
PY
)
variant="${selected[0]}"; stage_a_checkpoint="${selected[1]}"
if [[ "$seed" == 20260827 ]]; then
  [[ -n "$prior_seed_decision" && -f "$prior_seed_decision" ]] || { echo "Seed 20260827 requires the completed 20260826 decision" >&2; exit 2; }
  python - "$prior_seed_decision" <<'PY'
import json,sys
if json.load(open(sys.argv[1])).get('all_passed') is not True: raise SystemExit('Seed 20260826 did not pass; second formal seed forbidden')
PY
fi
run_root="${GP_SQ3DMIX_V2_RUN_ROOT:-$NAVSIM_EXP_ROOT/gp_sq3dmix_stage_a_v2}"
max_steps=10000
if [[ "$segment" == 30k ]]; then
  [[ -n "$interim_decision" && -f "$interim_decision" ]] || { echo "10k interim decision is required before 30k continuation" >&2; exit 2; }
  python - "$interim_decision" "$seed" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
if d.get('all_passed') is not True or int(d.get('seed',-1)) != int(sys.argv[2]): raise SystemExit('10k formal gate failed or seed mismatch')
PY
  max_steps=30000
fi
for arm in "$variant" control; do
  run_id="gp-sq3dmix-stage-c-${segment}-${arm}-${seed}"
  command=(bash "$project_root/tools/launch_gp_sq3dmix_training.sh" --phase stage_c_30k --variant "$arm" --seed "$seed" --decision-report "$stage_a_decision" --permission-report "$permission" --stage-a-checkpoint "$stage_a_checkpoint" --devices "$devices" --per-device-batch "$batch" --max-steps "$max_steps" --save-interval 5000 --run-root "$run_root" --run-id "$run_id")
  [[ -z "$accumulation" ]] || command+=(--gradient-accumulation "$accumulation")
  if [[ "$segment" == 30k ]]; then
    checkpoint="$run_root/gp-sq3dmix-stage-c-10k-${arm}-${seed}/checkpoints/steps_10000_pytorch_model.pt"
    [[ -f "$checkpoint" ]] || { echo "Missing matched 10k checkpoint: $checkpoint" >&2; exit 2; }
    command+=(--resume-checkpoint "$checkpoint" --resume-step 10000)
  fi
  (( dry_run )) && command+=(--dry-run)
  "${command[@]}"
done
