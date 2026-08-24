#!/usr/bin/env bash
# Continue both matched Stage-C pairs only when formal_100k_allowed=true.

set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"
permission="${GP_FORMAL_PERMISSION_REPORT:-}"
stage_a_decision="${GP_STAGE_A_V2_DECISION:-}"
devices="${GP_SQ3DMIX_DEVICES:-1}"
batch="${GP_SQ3DMIX_PER_DEVICE_BATCH:-2}"
accumulation="${GP_SQ3DMIX_GRADIENT_ACCUMULATION:-}"
segment=50k
midterm_decision="${GP_FORMAL_50K_DECISION:-}"
dry_run=0
while (( $# )); do
  case "$1" in
    --permission-report) permission="${2:?}"; shift 2 ;;
    --stage-a-decision) stage_a_decision="${2:?}"; shift 2 ;;
    --devices) devices="${2:?}"; shift 2 ;;
    --per-device-batch) batch="${2:?}"; shift 2 ;;
    --gradient-accumulation) accumulation="${2:?}"; shift 2 ;;
    --segment) segment="${2:?}"; shift 2 ;;
    --midterm-decision) midterm_decision="${2:?}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) echo "Usage: bash $project_root/33-continue_gp_sq3dmix_to_100k.sh --segment 50k|100k --permission-report JSON --stage-a-decision JSON [--midterm-decision JSON] [--dry-run]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$segment" == 50k || "$segment" == 100k ]] || { echo "--segment must be 50k or 100k" >&2; exit 2; }
for path in "$permission" "$stage_a_decision"; do [[ -f "$path" ]] || { echo "Missing 100k binding: $path" >&2; exit 2; }; done
python - "$permission" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
required=('stage_a_passed','stage_b_two_seed_passed','stage_b_full_navtest_passed','formal_30k_two_seed_passed','formal_30k_full_navtest_passed','formal_100k_allowed')
missing=[key for key in required if d.get(key) is not True]
if missing: raise SystemExit(f'100k extension permission is incomplete: {missing}')
PY
readarray -t selected < <(python - "$stage_a_decision" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); print(d['selected_variant']); print(d['selected_checkpoint'])
PY
)
variant="${selected[0]}"; stage_a_checkpoint="${selected[1]}"
if [[ "$segment" == 100k ]]; then
  [[ -n "$midterm_decision" && -f "$midterm_decision" ]] || { echo "100k segment requires the 50k midterm decision" >&2; exit 2; }
  python - "$midterm_decision" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
if d.get('stage') != 'formal_100k_50k_midterm' or d.get('all_passed') is not True:
 raise SystemExit('50k midterm did not pass; 100k continuation forbidden')
PY
fi
run_root="${GP_SQ3DMIX_V2_RUN_ROOT:-$NAVSIM_EXP_ROOT/gp_sq3dmix_stage_a_v2}"
for seed in 20260826 20260827; do
  for arm in "$variant" control; do
    resume_step=30000; max_steps=50000; source_segment=30k; run_segment=50k
    if [[ "$segment" == 100k ]]; then
      resume_step=50000; max_steps=100000; source_segment=50k; run_segment=100k
    fi
    checkpoint="$run_root/gp-sq3dmix-stage-c-${source_segment}-${arm}-${seed}/checkpoints/steps_${resume_step}_pytorch_model.pt"
    [[ -f "$checkpoint" ]] || { echo "Missing ${resume_step}-step checkpoint: $checkpoint" >&2; exit 2; }
    command=(bash "$project_root/tools/launch_gp_sq3dmix_training.sh" --phase stage_c_100k --variant "$arm" --seed "$seed" --decision-report "$stage_a_decision" --permission-report "$permission" --stage-a-checkpoint "$stage_a_checkpoint" --resume-checkpoint "$checkpoint" --resume-step "$resume_step" --max-steps "$max_steps" --save-interval 10000 --devices "$devices" --per-device-batch "$batch" --run-root "$run_root" --run-id "gp-sq3dmix-stage-c-${run_segment}-${arm}-${seed}")
    [[ -z "$accumulation" ]] || command+=(--gradient-accumulation "$accumulation")
    (( dry_run )) && command+=(--dry-run)
    "${command[@]}"
  done
done
