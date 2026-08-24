#!/usr/bin/env bash
# Evaluate one Stage-A-v2 checkpoint on an immutable paired split.

set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

run_dir=""
checkpoint=""
variant=""
split=final_gate
output=""
batch_size="${GP_SQ3DMIX_EVAL_BATCH_SIZE:-2}"
num_workers="${GP_SQ3DMIX_EVAL_WORKERS:-2}"
seed=20260824
draws=10000
dry_run=0
all_sweep=0
eval_root="${GP_SQ3DMIX_V2_EVAL_ROOT:-$NAVSIM_EVAL_ROOT/gp_sq3dmix_stage_a_v2_eval}/stage_a_v2"
cache_root="${GP_SQ3DMIX_TRAIN_CACHE_ROOT:-$NAVSIM_VGGT_DENSE_CACHE_ROOT}"
source_cache_root="$cache_root"
stats_root="${GP_SQ3DMIX_V2_STATS_ROOT:-$DRIVEDREAMER_SHARED_ROOT/navsim_feature_cache/gp_sq3dmix_stage_a_v2_slot_stats}"
negative_root="${GP_SQ3DMIX_V2_NEGATIVE_MAP_ROOT:-$DRIVEDREAMER_SHARED_ROOT/navsim_feature_cache/gp_sq3dmix_stage_a_v2_negative_maps}"
source_datalist="${GP_SQ3DMIX_TRAIN_DATALIST:-$NAVSIM_DATALIST_PATH}"
while (( $# )); do
  case "$1" in
    --run-dir) run_dir="${2:?}"; shift 2 ;;
    --checkpoint) checkpoint="${2:?}"; shift 2 ;;
    --variant) variant="${2:?}"; shift 2 ;;
    --split) split="${2:?}"; shift 2 ;;
    --output) output="${2:?}"; shift 2 ;;
    --batch-size) batch_size="${2:?}"; shift 2 ;;
    --num-workers) num_workers="${2:?}"; shift 2 ;;
    --seed) seed="${2:?}"; shift 2 ;;
    --bootstrap-draws) draws="${2:?}"; shift 2 ;;
    --all) all_sweep=1; shift ;;
    --eval-root) eval_root="${2:?}"; shift 2 ;;
    --cache-root) cache_root="${2:?}"; shift 2 ;;
    --stats-root) stats_root="${2:?}"; shift 2 ;;
    --negative-root) negative_root="${2:?}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help)
      echo "Usage: bash $project_root/27-eval_gp_sq3dmix_stage_a_v2.sh --run-dir DIR --checkpoint FILE --variant projected_residual|gated_residual [--split smoke|model_selection|final_gate] --output FILE [--dry-run]"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
if (( all_sweep )); then
  [[ -z "$run_dir$checkpoint$variant$output" ]] || { echo "--all cannot be combined with single-checkpoint arguments" >&2; exit 2; }
  run_root="${GP_SQ3DMIX_V2_RUN_ROOT:-$NAVSIM_EXP_ROOT/gp_sq3dmix_stage_a_v2}"
  for sweep_variant in projected_residual gated_residual; do
    sweep_run="$run_root/gp-sq3dmix-stage-a-v2-${sweep_variant}-${seed}"
    [[ -d "$sweep_run" ]] || { echo "Missing Stage-A-v2 run: $sweep_run" >&2; exit 2; }
    reports=()
    for step in 250 500 750 1000 1250 1500 1750 2000; do
      sweep_checkpoint="$sweep_run/checkpoints/steps_${step}_pytorch_model.pt"
      sweep_output="$eval_root/$sweep_variant/model_selection/step_${step}.json"
      command=(bash "$project_root/27-eval_gp_sq3dmix_stage_a_v2.sh"
        --run-dir "$sweep_run" --checkpoint "$sweep_checkpoint"
        --variant "$sweep_variant" --split model_selection
        --output "$sweep_output" --batch-size "$batch_size"
        --num-workers "$num_workers" --seed "$seed" --bootstrap-draws "$draws")
      (( dry_run )) && command+=(--dry-run)
      "${command[@]}"
      reports+=("$sweep_output")
    done
    if (( dry_run )); then
      echo "[gp-stage-a-v2-eval] dry-run: checkpoint selection/final gate deferred until reports exist"
      continue
    fi
    selection="$eval_root/$sweep_variant/model_selection/selection.json"
    PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}" python \
      "$project_root/tools/select_gp_sq3dmix_stage_a_v2.py" select-checkpoint \
      --reports "${reports[@]}" --output "$selection"
    selected_checkpoint="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_checkpoint"])' "$selection")"
    bash "$project_root/27-eval_gp_sq3dmix_stage_a_v2.sh" \
      --run-dir "$sweep_run" --checkpoint "$selected_checkpoint" \
      --variant "$sweep_variant" --split final_gate \
      --output "$eval_root/$sweep_variant/final_gate.json" \
      --batch-size "$batch_size" --num-workers "$num_workers" \
      --seed "$seed" --bootstrap-draws "$draws"
  done
  (( dry_run )) && exit 0
  PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}" python \
    "$project_root/tools/select_gp_sq3dmix_stage_a_v2.py" decide-variant \
    --projected-report "$eval_root/projected_residual/final_gate.json" \
    --gated-report "$eval_root/gated_residual/final_gate.json" \
    --output "$eval_root/stage_a_v2_decision.json"
  exit 0
fi
[[ -n "$run_dir" && -d "$run_dir" ]] || { echo "--run-dir is required" >&2; exit 2; }
[[ -n "$checkpoint" && -f "$checkpoint" ]] || { echo "--checkpoint is required" >&2; exit 2; }
[[ "$variant" == projected_residual || "$variant" == gated_residual ]] || { echo "Invalid --variant" >&2; exit 2; }
[[ -n "$output" ]] || { echo "--output is required" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "Refusing to overwrite evaluation: $output" >&2; exit 2; }
case "$split" in
  smoke)
    datalist="$project_root/docs/experiments/splits/gp_sq3dmix_stage_a_v2_smoke_selection_128.json"
    map_dir="$negative_root/smoke_selection_128"
    draws="${GP_SQ3DMIX_SMOKE_BOOTSTRAP_DRAWS:-$draws}"
    ;;
  model_selection)
    datalist="$project_root/docs/experiments/splits/gp_sq3dmix_stage_a_v2_model_selection.json"
    map_dir="$negative_root/stage_a_v2_model_selection"
    ;;
  final_gate)
    datalist="$project_root/docs/experiments/splits/gp_sq3dmix_stage_a_v2_final_gate.json"
    map_dir="$negative_root/stage_a_v2_final_gate"
    ;;
  *) echo "Invalid --split: $split" >&2; exit 2 ;;
esac

branch="$(git -C "$project_root" branch --show-current)"
commit="$(git -C "$project_root" rev-parse HEAD)"
[[ "$branch" == feature/gp-sq-3d-mix-stage-a-v2 ]] || { echo "Wrong DLC-visible branch: $branch" >&2; exit 2; }
[[ -z "$(git -C "$project_root" status --porcelain)" ]] || { echo "Stage-A-v2 evaluation requires a clean worktree" >&2; exit 2; }
for path in "$run_dir/config.yaml" "$datalist" "$map_dir/hard_negative_map.json" "$map_dir/manifest.json" "$stats_root/manifest.json" "$cache_root/vggt_dense/manifest.json"; do
  [[ -f "$path" ]] || { echo "Missing evaluation input: $path" >&2; exit 2; }
done

args=(
  --run-dir "$run_dir"
  --checkpoint "$checkpoint"
  --variant "$variant"
  --datalist "$datalist"
  --data-root "$DATA_ROOT"
  --cache-root "$cache_root"
  --stats-root "$stats_root"
  --source-datalist "$source_datalist"
  --source-cache-root "$source_cache_root"
  --negative-map "$map_dir/hard_negative_map.json"
  --negative-map-manifest "$map_dir/manifest.json"
  --output "$output"
  --batch-size "$batch_size"
  --num-workers "$num_workers"
  --seed "$seed"
  --bootstrap-draws "$draws"
)
echo "[gp-stage-a-v2-eval] code=$project_root branch=$branch commit=$commit split=$split variant=$variant"
printf '[gp-stage-a-v2-eval] command: python %q ' "$project_root/tools/evaluate_gp_sq3dmix_stage_a.py"
printf '%q ' "${args[@]}"
printf '\n'
(( dry_run )) && exit 0
PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}" python "$project_root/tools/evaluate_gp_sq3dmix_stage_a.py" "${args[@]}"
