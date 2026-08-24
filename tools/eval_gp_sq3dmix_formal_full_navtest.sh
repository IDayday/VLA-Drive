#!/usr/bin/env bash
# Evaluate both selected Stage-C seeds on the immutable full navtest.

set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_root/load_env.sh"

selection=""
permission_before=""
world_size="${EVAL_DEVICE_COUNT:-1}"
batch_size="${BATCH_SIZE:-2}"
num_workers="${NUM_WORKERS:-2}"
dry_run=0
while (( $# )); do
  case "$1" in
    --selection) selection="${2:?}"; shift 2 ;;
    --permission-before) permission_before="${2:?}"; shift 2 ;;
    --world-size) world_size="${2:?}"; shift 2 ;;
    --batch-size) batch_size="${2:?}"; shift 2 ;;
    --num-workers) num_workers="${2:?}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) echo "Usage: bash $project_root/tools/eval_gp_sq3dmix_formal_full_navtest.sh --selection JSON --permission-before JSON [--dry-run]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
for value in "$world_size" "$batch_size" "$num_workers"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid evaluator topology" >&2; exit 2; }
done
for path in "$selection" "$permission_before"; do
  [[ -f "$path" ]] || { echo "Missing formal full-navtest binding: $path" >&2; exit 2; }
done
branch="$(git -C "$project_root" branch --show-current)"
commit="$(git -C "$project_root" rev-parse HEAD)"
[[ "$branch" == feature/gp-sq-3d-mix-stage-a-v2 ]] || { echo "Wrong DLC-visible branch: $branch" >&2; exit 2; }
[[ -z "$(git -C "$project_root" status --porcelain)" ]] || { echo "Formal full-navtest requires a clean worktree" >&2; exit 2; }
readarray -t implementation_commits < <(python - "$selection" "$permission_before" <<'PY'
import json,sys
selection=json.load(open(sys.argv[1])); permission=json.load(open(sys.argv[2]))
if selection.get('all_passed') is not True: raise SystemExit('formal two-seed 2k gate failed')
if permission.get('formal_30k_allowed') is not True: raise SystemExit('formal_30k_allowed=false')
if permission.get('formal_100k_allowed') is not False: raise SystemExit('formal full-navtest expects formal_100k_allowed=false')
print(selection['code_commit']); print(permission['code_commit'])
PY
)
for bound_commit in "${implementation_commits[@]}"; do
  python "$project_root/tools/validate_gp_sq3dmix_code_commit.py" --repo "$project_root" --bound "$bound_commit" --current "$commit"
done
run_root="${GP_SQ3DMIX_V2_RUN_ROOT:-$NAVSIM_EXP_ROOT/gp_sq3dmix_stage_a_v2}"
eval_base="${GP_SQ3DMIX_V2_EVAL_ROOT:-$NAVSIM_EVAL_ROOT/gp_sq3dmix_stage_a_v2_eval}/formal_30k"
out_root="$eval_base/full_navtest"
datalist="${NAVSIM_TEST_DATALIST:-$DRIVEDREAMER_SHARED_ROOT/test_meta.json}"
negative_root="${GP_SQ3DMIX_V2_NEGATIVE_MAP_ROOT:-$DRIVEDREAMER_SHARED_ROOT/navsim_feature_cache/gp_sq3dmix_stage_a_v2_negative_maps}"
stats_root="${GP_SQ3DMIX_V2_STATS_ROOT:-$DRIVEDREAMER_SHARED_ROOT/navsim_feature_cache/gp_sq3dmix_stage_a_v2_slot_stats}"
source_cache="${GP_SQ3DMIX_TRAIN_CACHE_ROOT:-$NAVSIM_VGGT_DENSE_CACHE_ROOT}"
source_datalist="${GP_SQ3DMIX_TRAIN_DATALIST:-$NAVSIM_DATALIST_PATH}"
dense_cache="${GP_SQ3DMIX_TEST_CACHE_ROOT:-$NAVSIM_VGGT_DENSE_TEST_CACHE_ROOT}"
readarray -t binding < <(python - "$selection" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); print(d['selected_variant'])
for seed in ('20260826','20260827'):
 print(d['seeds'][seed]['selected_step'])
 print(d['seeds'][seed]['selected_checkpoint'])
 print(d['seeds'][seed]['selected_control_checkpoint'])
PY
)
variant="${binding[0]}"
steps=("${binding[1]}" "${binding[4]}")
gp_checkpoints=("${binding[2]}" "${binding[5]}")
control_checkpoints=("${binding[3]}" "${binding[6]}")
if [[ ! -f "$negative_root/navtest_full/manifest.json" ]]; then
  if (( dry_run )); then
    echo "[gp-formal-full] would build the optional immutable full-navtest hard-negative map"
  else
    bash "$project_root/24-build_gp_sq3dmix_hard_negative_maps.sh" --include-full-navtest --only navtest_full
  fi
fi
for path in "$datalist" "$stats_root/manifest.json" "$dense_cache/vggt_dense/manifest.json"; do
  [[ -f "$path" ]] || { echo "Missing formal full-navtest input: $path" >&2; exit 2; }
done
if (( ! dry_run )); then
  for path in "$negative_root/navtest_full/manifest.json" "$negative_root/navtest_full/hard_negative_map.json"; do
    [[ -f "$path" ]] || { echo "Missing full-navtest negative map: $path" >&2; exit 2; }
  done
fi
for index in 0 1; do
  for checkpoint in "${gp_checkpoints[$index]}" "${control_checkpoints[$index]}"; do
    [[ -f "$checkpoint" ]] || { echo "Missing selected formal checkpoint: $checkpoint" >&2; exit 2; }
  done
done
[[ ! -e "$out_root" ]] || { echo "Refusing to overwrite formal full-navtest output: $out_root" >&2; exit 2; }
echo "[gp-formal-full] variant=$variant seeds=20260826,20260827 steps=${steps[*]} scenes=12146 noise=per_token out=$out_root"
if (( dry_run )); then
  printf 'GP_SQ3DMIX_INTERVENTION=hard_shuffled INFER_NOISE_MODE=per_token bash %q\n' "$project_root/4-infer.sh"
  printf 'python %q full --root %q --selection %q --permission-before %q --output-json %q --output-csv %q --permission-json %q\n' \
    "$project_root/tools/summarize_gp_sq3dmix_formal_v2.py" "$out_root" "$selection" "$permission_before" \
    "$out_root/formal_full_navtest_decision.json" "$out_root/gp_sq3dmix_formal_interventions.csv" "$out_root/formal_training_permission.json"
  exit 0
fi

pdms_cache="${PDMS_METRIC_CACHE_PATH:-$DRIVEDREAMER_SHARED_ROOT/navsim_exp/metric_cache_navtest_v1_1}"
epdms_cache="${EPDMS_METRIC_CACHE_PATH:-$DRIVEDREAMER_SHARED_ROOT/navsim_exp/metric_cache_navtest}"
for cache in "$pdms_cache" "$epdms_cache"; do
  [[ -d "$cache/metadata" ]] || { echo "Full metric cache is missing: $cache" >&2; exit 2; }
done
protocol=(python "$project_root/tools/write_gp_sq3dmix_protocol_manifest.py"
  --output "$out_root/protocol.json" --phase formal_30k_full_navtest
  --code-commit "$commit"
  --input "selection=$selection" --input "permission_before=$permission_before"
  --input "datalist=$datalist"
  --input "negative_map=$negative_root/navtest_full/hard_negative_map.json"
  --input "negative_map_manifest=$negative_root/navtest_full/manifest.json"
  --input "slot_stats_manifest=$stats_root/manifest.json"
  --input "source_cache_manifest=$source_cache/vggt_dense/manifest.json"
  --input "target_cache_manifest=$dense_cache/vggt_dense/manifest.json"
  --value "variant=$variant" --value "noise_mode=per_token"
  --value "world_size=$world_size" --value "batch_size=$batch_size"
  --value "num_workers=$num_workers")
for index in 0 1; do
  seed=$((20260826 + index)); step="${steps[$index]}"
  protocol+=(
    --input "gp_${seed}_${step}=${gp_checkpoints[$index]}"
    --input "control_${seed}_${step}=${control_checkpoints[$index]}"
  )
done
"${protocol[@]}"

score_mode() {
  local seed_root="$1" mode="$2" prediction_run="$3"
  local score_root="$seed_root/scores/$mode" pdms_work="$score_root/pdms_work" epdms_work="$score_root/epdms_work" pdms_csv epdms_csv
  mkdir -p "$pdms_work" "$epdms_work" "$seed_root/logs/$mode"
  PRED_DIR="$prediction_run" SPLIT=test DATALIST="$datalist" METRIC_CACHE_PATH="$pdms_cache" NAVSIM_EVAL_ROOT="$pdms_work" bash "$project_root/5-eval_v1.sh" >"$seed_root/logs/$mode/pdms.log" 2>&1
  PRED_DIR="$prediction_run" SPLIT=test DATALIST="$datalist" METRIC_CACHE_PATH="$epdms_cache" NAVSIM_EVAL_ROOT="$epdms_work" bash "$project_root/6-eval_v2.sh" >"$seed_root/logs/$mode/epdms.log" 2>&1
  pdms_csv="$(find "$pdms_work" -type f -name '*.csv' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
  epdms_csv="$(find "$epdms_work" -type f -name '*.csv' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
  [[ -n "$pdms_csv" && -n "$epdms_csv" ]] || { echo "Formal scoring produced no CSV: seed_root=$seed_root mode=$mode" >&2; exit 1; }
  cp "$pdms_csv" "$score_root/pdms.csv"; cp "$epdms_csv" "$score_root/epdms.csv"
}

infer_mode() {
  local seed_root="$1" mode="$2" model_dir="$3" step="$4" prediction_root="$seed_root/predictions/$mode" intervention="$mode" run_name prediction_run count failed
  [[ "$mode" != control ]] || intervention=real
  mkdir -p "$prediction_root" "$seed_root/logs/$mode"; pids=()
  for ((rank=0; rank<world_size; rank++)); do
    (GP_SQ3DMIX_INTERVENTION="$intervention" MODEL_DIR="$model_dir" MODEL_ITER="$step" SPLIT=test DATALIST="$datalist" OUT_DIR="$prediction_root" BATCH_SIZE="$batch_size" NUM_WORKERS="$num_workers" GPU="$rank" RANK="$rank" WORLD_SIZE="$world_size" INFER_SEED=20260824 INFER_NOISE_MODE=per_token INFER_USE_FEATURE_CACHE=0 NAVSIM_VGGT_DENSE_CACHE_ROOT="$dense_cache" GP_SQ3DMIX_STATS_ROOT="$stats_root" GP_SQ3DMIX_SOURCE_DATALIST="$source_datalist" GP_SQ3DMIX_SOURCE_CACHE_MANIFEST="$source_cache/vggt_dense/manifest.json" GP_SQ3DMIX_SOURCE_CACHE_ROOT="$source_cache" GP_SQ3DMIX_NEGATIVE_MAP="$negative_root/navtest_full/hard_negative_map.json" GP_SQ3DMIX_NEGATIVE_MAP_MANIFEST="$negative_root/navtest_full/manifest.json" VLM_ATTN_IMPLEMENTATION=sdpa bash "$project_root/4-infer.sh") >"$seed_root/logs/$mode/infer.rank${rank}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  (( failed == 0 )) || { echo "Formal full inference failed seed_root=$seed_root mode=$mode" >&2; exit 1; }
  run_name="$(basename -- "$model_dir")"; prediction_run="$prediction_root/${run_name}-step${step}"
  count="$(find "$prediction_run/test" -type f -name '*.npy' | wc -l)"
  [[ "$count" == 12146 ]] || { echo "Formal full prediction count $count != 12146" >&2; exit 1; }
  score_mode "$seed_root" "$mode" "$prediction_run"
}

for index in 0 1; do
  seed=$((20260826 + index)); step="${steps[$index]}"; seed_root="$out_root/seed${seed}"
  gp_run="$(dirname -- "$(dirname -- "${gp_checkpoints[$index]}")")"
  control_run="$(dirname -- "$(dirname -- "${control_checkpoints[$index]}")")"
  for mode in real hard_shuffled spatial_shuffled; do infer_mode "$seed_root" "$mode" "$gp_run" "$step"; done
  infer_mode "$seed_root" control "$control_run" "$step"
done
PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}" python "$project_root/tools/summarize_gp_sq3dmix_formal_v2.py" full \
  --root "$out_root" --selection "$selection" --permission-before "$permission_before" \
  --output-json "$out_root/formal_full_navtest_decision.json" \
  --output-csv "$out_root/gp_sq3dmix_formal_interventions.csv" \
  --permission-json "$out_root/formal_training_permission.json"
