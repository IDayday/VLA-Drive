#!/usr/bin/env bash
# Run paired full-navtest only after both Stage-B 2k seeds pass.

set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

stage_b_decision="${GP_STAGE_B_MULTI_SEED_DECISION:-${GP_SQ3DMIX_V2_EVAL_ROOT:-$NAVSIM_EVAL_ROOT/gp_sq3dmix_stage_a_v2_eval}/stage_b_multiseed_2k/stage_b_multiseed_decision.json}"
run_root="${GP_SQ3DMIX_V2_RUN_ROOT:-$NAVSIM_EXP_ROOT/gp_sq3dmix_stage_a_v2}"
out_root="${GP_SQ3DMIX_V2_EVAL_ROOT:-$NAVSIM_EVAL_ROOT/gp_sq3dmix_stage_a_v2_eval}/stage_b_full_navtest"
datalist="${NAVSIM_TEST_DATALIST:-$DRIVEDREAMER_SHARED_ROOT/test_meta.json}"
negative_root="${GP_SQ3DMIX_V2_NEGATIVE_MAP_ROOT:-$DRIVEDREAMER_SHARED_ROOT/navsim_feature_cache/gp_sq3dmix_stage_a_v2_negative_maps}"
stats_root="${GP_SQ3DMIX_V2_STATS_ROOT:-$DRIVEDREAMER_SHARED_ROOT/navsim_feature_cache/gp_sq3dmix_stage_a_v2_slot_stats}"
source_cache="${GP_SQ3DMIX_TRAIN_CACHE_ROOT:-$NAVSIM_VGGT_DENSE_CACHE_ROOT}"
source_datalist="${GP_SQ3DMIX_TRAIN_DATALIST:-$NAVSIM_DATALIST_PATH}"
dense_cache="${GP_SQ3DMIX_TEST_CACHE_ROOT:-$NAVSIM_VGGT_DENSE_TEST_CACHE_ROOT}"
world_size="${EVAL_DEVICE_COUNT:-1}"
batch_size="${BATCH_SIZE:-2}"
num_workers="${NUM_WORKERS:-2}"
dry_run=0
while (( $# )); do
  case "$1" in
    --stage-b-decision) stage_b_decision="${2:?}"; shift 2 ;;
    --run-root) run_root="${2:?}"; shift 2 ;;
    --out-root) out_root="${2:?}"; shift 2 ;;
    --world-size) world_size="${2:?}"; shift 2 ;;
    --batch-size) batch_size="${2:?}"; shift 2 ;;
    --num-workers) num_workers="${2:?}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) echo "Usage: bash $project_root/30-eval_gp_sq3dmix_stage_b_full_navtest.sh [--world-size N --batch-size N] [--dry-run]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
for value in "$world_size" "$batch_size" "$num_workers"; do [[ "$value" =~ ^[1-9][0-9]*$ ]] || exit 2; done
branch="$(git -C "$project_root" branch --show-current)"; commit="$(git -C "$project_root" rev-parse HEAD)"
[[ "$branch" == feature/gp-sq-3d-mix-stage-a-v2 ]] || { echo "Wrong DLC-visible branch" >&2; exit 2; }
[[ -z "$(git -C "$project_root" status --porcelain)" ]] || { echo "Full-navtest requires a clean worktree" >&2; exit 2; }
[[ -f "$stage_b_decision" && -f "$datalist" ]] || { echo "Missing Stage-B decision/full datalist" >&2; exit 2; }
readarray -t binding < <(python - "$stage_b_decision" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
if d.get('all_passed') is not True: raise SystemExit('Two-seed 2k gates failed; full-navtest forbidden')
print(d['selected_variant'])
print(d['code_commit'])
for seed in ('20260824','20260825'): print(d['seeds'][seed]['selected_step'])
PY
)
variant="${binding[0]}"; steps=("${binding[2]}" "${binding[3]}")
python "$project_root/tools/validate_gp_sq3dmix_code_commit.py" --repo "$project_root" --bound "${binding[1]}" --current "$commit"
if [[ ! -f "$negative_root/navtest_full/manifest.json" ]]; then
  (( dry_run )) && { echo "[gp-full-navtest] would build optional full-navtest hard-negative map"; } || \
    bash "$project_root/24-build_gp_sq3dmix_hard_negative_maps.sh" --include-full-navtest --only navtest_full
fi
for path in "$negative_root/navtest_full/manifest.json" "$negative_root/navtest_full/hard_negative_map.json" "$stats_root/manifest.json" "$dense_cache/vggt_dense/manifest.json"; do
  if (( dry_run )) && [[ "$path" == "$negative_root/navtest_full/manifest.json" || "$path" == "$negative_root/navtest_full/hard_negative_map.json" ]]; then continue; fi
  [[ -f "$path" ]] || { echo "Missing full-navtest input: $path" >&2; exit 2; }
done
for index in 0 1; do
  seed=$((20260824 + index)); step="${steps[$index]}"
  for arm in "$variant" control; do
    [[ -f "$run_root/gp-sq3dmix-stage-b-${arm}-${seed}/checkpoints/steps_${step}_pytorch_model.pt" ]] || { echo "Missing selected checkpoint seed=$seed arm=$arm" >&2; exit 2; }
  done
done
[[ ! -e "$out_root" ]] || { echo "Refusing to overwrite full-navtest output" >&2; exit 2; }
echo "[gp-full-navtest] variant=$variant selected_steps=${steps[*]} scenes=12146 noise=per_token out=$out_root"
if (( dry_run )); then
  printf 'GP_SQ3DMIX_INTERVENTION=hard_shuffled GP_SQ3DMIX_NEGATIVE_MAP=%q INFER_NOISE_MODE=per_token bash %q\n' "$negative_root/navtest_full/hard_negative_map.json" "$project_root/4-infer.sh"
  printf 'python %q --root %q --stage-b-decision %q --output-csv %q --decision-json %q --permission-json %q\n' "$project_root/tools/summarize_gp_sq3dmix_full_navtest_v2.py" "$out_root" "$stage_b_decision" "$out_root/gp_sq3dmix_stage_b_full_navtest.csv" "$out_root/full_navtest_decision.json" "$out_root/formal_training_permission.json"
  exit 0
fi

pdms_cache="${PDMS_METRIC_CACHE_PATH:-$DRIVEDREAMER_SHARED_ROOT/navsim_exp/metric_cache_navtest_v1_1}"
epdms_cache="${EPDMS_METRIC_CACHE_PATH:-$DRIVEDREAMER_SHARED_ROOT/navsim_exp/metric_cache_navtest}"
for cache in "$pdms_cache" "$epdms_cache"; do [[ -d "$cache/metadata" ]] || { echo "Full metric cache is missing: $cache" >&2; exit 2; }; done
protocol=(python "$project_root/tools/write_gp_sq3dmix_protocol_manifest.py"
  --output "$out_root/protocol.json" --phase stage_b_full_navtest
  --code-commit "$commit"
  --input "stage_b_decision=$stage_b_decision"
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
  seed=$((20260824 + index)); step="${steps[$index]}"
  protocol+=(
    --input "gp_${seed}_${step}=$run_root/gp-sq3dmix-stage-b-${variant}-${seed}/checkpoints/steps_${step}_pytorch_model.pt"
    --input "control_${seed}_${step}=$run_root/gp-sq3dmix-stage-b-control-${seed}/checkpoints/steps_${step}_pytorch_model.pt"
  )
done
"${protocol[@]}"

score_mode() {
  local seed_root="$1" mode="$2" prediction_run="$3" score_root="$seed_root/scores/$mode" pdms_work="$score_root/pdms_work" epdms_work="$score_root/epdms_work" pdms_csv epdms_csv
  mkdir -p "$pdms_work" "$epdms_work" "$seed_root/logs/$mode"
  PRED_DIR="$prediction_run" SPLIT=test DATALIST="$datalist" METRIC_CACHE_PATH="$pdms_cache" NAVSIM_EVAL_ROOT="$pdms_work" bash "$project_root/5-eval_v1.sh" >"$seed_root/logs/$mode/pdms.log" 2>&1
  PRED_DIR="$prediction_run" SPLIT=test DATALIST="$datalist" METRIC_CACHE_PATH="$epdms_cache" NAVSIM_EVAL_ROOT="$epdms_work" bash "$project_root/6-eval_v2.sh" >"$seed_root/logs/$mode/epdms.log" 2>&1
  pdms_csv="$(find "$pdms_work" -type f -name '*.csv' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"; epdms_csv="$(find "$epdms_work" -type f -name '*.csv' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
  [[ -n "$pdms_csv" && -n "$epdms_csv" ]] || { echo "No scoring CSV for $mode" >&2; exit 1; }
  cp "$pdms_csv" "$score_root/pdms.csv"; cp "$epdms_csv" "$score_root/epdms.csv"
}

infer_mode() {
  local seed_root="$1" mode="$2" model_dir="$3" step="$4" prediction_root="$seed_root/predictions/$mode" intervention="$mode" run_name prediction_run failed count
  [[ "$mode" != control ]] || intervention=real
  mkdir -p "$prediction_root" "$seed_root/logs/$mode"; pids=()
  for ((rank=0; rank<world_size; rank++)); do
    (GP_SQ3DMIX_INTERVENTION="$intervention" MODEL_DIR="$model_dir" MODEL_ITER="$step" SPLIT=test DATALIST="$datalist" OUT_DIR="$prediction_root" BATCH_SIZE="$batch_size" NUM_WORKERS="$num_workers" GPU="$rank" RANK="$rank" WORLD_SIZE="$world_size" INFER_SEED=20260824 INFER_NOISE_MODE=per_token INFER_USE_FEATURE_CACHE=0 NAVSIM_VGGT_DENSE_CACHE_ROOT="$dense_cache" GP_SQ3DMIX_STATS_ROOT="$stats_root" GP_SQ3DMIX_SOURCE_DATALIST="$source_datalist" GP_SQ3DMIX_SOURCE_CACHE_MANIFEST="$source_cache/vggt_dense/manifest.json" GP_SQ3DMIX_SOURCE_CACHE_ROOT="$source_cache" GP_SQ3DMIX_NEGATIVE_MAP="$negative_root/navtest_full/hard_negative_map.json" GP_SQ3DMIX_NEGATIVE_MAP_MANIFEST="$negative_root/navtest_full/manifest.json" VLM_ATTN_IMPLEMENTATION=sdpa bash "$project_root/4-infer.sh") >"$seed_root/logs/$mode/infer.rank${rank}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; (( failed == 0 )) || exit 1
  run_name="$(basename -- "$model_dir")"; prediction_run="$prediction_root/${run_name}-step${step}"; count="$(find "$prediction_run/test" -type f -name '*.npy' | wc -l)"; [[ "$count" == 12146 ]] || { echo "Full prediction count mismatch" >&2; exit 1; }
  score_mode "$seed_root" "$mode" "$prediction_run"
}

for index in 0 1; do
  seed=$((20260824 + index)); step="${steps[$index]}"; seed_root="$out_root/seed${seed}"
  gp_run="$run_root/gp-sq3dmix-stage-b-${variant}-${seed}"; control_run="$run_root/gp-sq3dmix-stage-b-control-${seed}"
  for mode in real hard_shuffled spatial_shuffled; do infer_mode "$seed_root" "$mode" "$gp_run" "$step"; done
  infer_mode "$seed_root" control "$control_run" "$step"
done
PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}" python "$project_root/tools/summarize_gp_sq3dmix_full_navtest_v2.py" --root "$out_root" --stage-b-decision "$stage_b_decision" --output-csv "$out_root/gp_sq3dmix_stage_b_full_navtest.csv" --decision-json "$out_root/full_navtest_decision.json" --permission-json "$out_root/formal_training_permission.json"
