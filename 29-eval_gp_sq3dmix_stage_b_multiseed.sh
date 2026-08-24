#!/usr/bin/env bash
# Evaluate both matched Stage-B seeds at every 2k checkpoint on fixed navtest-2k.

set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

stage_a_decision="${GP_STAGE_A_V2_DECISION:-${GP_SQ3DMIX_V2_EVAL_ROOT:-$NAVSIM_EVAL_ROOT/gp_sq3dmix_stage_a_v2_eval}/stage_a_v2/stage_a_v2_decision.json}"
run_root="${GP_SQ3DMIX_V2_RUN_ROOT:-$NAVSIM_EXP_ROOT/gp_sq3dmix_stage_a_v2}"
out_root="${GP_SQ3DMIX_V2_EVAL_ROOT:-$NAVSIM_EVAL_ROOT/gp_sq3dmix_stage_a_v2_eval}/stage_b_multiseed_2k"
datalist="$project_root/docs/experiments/splits/gp_sq3dmix_navtest_2k.json"
negative_root="${GP_SQ3DMIX_V2_NEGATIVE_MAP_ROOT:-$DRIVEDREAMER_SHARED_ROOT/navsim_feature_cache/gp_sq3dmix_stage_a_v2_negative_maps}"
stats_root="${GP_SQ3DMIX_V2_STATS_ROOT:-$DRIVEDREAMER_SHARED_ROOT/navsim_feature_cache/gp_sq3dmix_stage_a_v2_slot_stats}"
source_cache="${GP_SQ3DMIX_TRAIN_CACHE_ROOT:-$NAVSIM_VGGT_DENSE_CACHE_ROOT}"
source_datalist="${GP_SQ3DMIX_TRAIN_DATALIST:-$NAVSIM_DATALIST_PATH}"
dense_cache="${GP_SQ3DMIX_TEST_CACHE_ROOT:-$NAVSIM_VGGT_DENSE_TEST_CACHE_ROOT}"
world_size="${EVAL_DEVICE_COUNT:-1}"
batch_size="${BATCH_SIZE:-2}"
num_workers="${NUM_WORKERS:-2}"
infer_seed="${INFER_SEED:-20260824}"
dry_run=0
while (( $# )); do
  case "$1" in
    --stage-a-decision) stage_a_decision="${2:?}"; shift 2 ;;
    --run-root) run_root="${2:?}"; shift 2 ;;
    --out-root) out_root="${2:?}"; shift 2 ;;
    --world-size) world_size="${2:?}"; shift 2 ;;
    --batch-size) batch_size="${2:?}"; shift 2 ;;
    --num-workers) num_workers="${2:?}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) echo "Usage: bash $project_root/29-eval_gp_sq3dmix_stage_b_multiseed.sh [--world-size N --batch-size N] [--dry-run]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
for value in "$world_size" "$batch_size" "$num_workers"; do [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Evaluation topology values must be positive" >&2; exit 2; }; done
branch="$(git -C "$project_root" branch --show-current)"
commit="$(git -C "$project_root" rev-parse HEAD)"
[[ "$branch" == feature/gp-sq-3d-mix-stage-a-v2 ]] || { echo "Wrong DLC-visible branch" >&2; exit 2; }
[[ -z "$(git -C "$project_root" status --porcelain)" ]] || { echo "Stage-B evaluation requires a clean worktree" >&2; exit 2; }
for path in "$stage_a_decision" "$datalist" "$negative_root/navtest_2k/manifest.json" "$negative_root/navtest_2k/hard_negative_map.json" "$stats_root/manifest.json" "$source_cache/vggt_dense/manifest.json" "$dense_cache/vggt_dense/manifest.json"; do
  [[ -f "$path" ]] || { echo "Missing Stage-B evaluation input: $path" >&2; exit 2; }
done
readarray -t selected < <(python - "$stage_a_decision" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
if d.get('all_passed') is not True: raise SystemExit('Stage A did not pass')
print(d['selected_variant'])
print(d['code_commit'])
PY
)
variant="${selected[0]}"
python "$project_root/tools/validate_gp_sq3dmix_code_commit.py" --repo "$project_root" --bound "${selected[1]}" --current "$commit"
for seed in 20260824 20260825; do
  for arm in "$variant" control; do
    run="$run_root/gp-sq3dmix-stage-b-${arm}-${seed}"
    [[ -f "$run/config.yaml" ]] || { echo "Missing matched run: $run" >&2; exit 2; }
    for step in 2000 4000 6000 8000 10000; do
      [[ -f "$run/checkpoints/steps_${step}_pytorch_model.pt" ]] || { echo "Missing $arm seed=$seed step=$step" >&2; exit 2; }
    done
  done
done
[[ ! -e "$out_root" ]] || { echo "Refusing to overwrite Stage-B evaluation: $out_root" >&2; exit 2; }
result_csv="$out_root/gp_sq3dmix_stage_b_multiseed.csv"
interventions_csv="$out_root/gp_sq3dmix_stage_b_interventions.csv"
decision_json="$out_root/stage_b_multiseed_decision.json"
echo "[gp-stage-b-v2-eval] code=$project_root commit=$commit variant=$variant out=$out_root"
echo "[gp-stage-b-v2-eval] seeds=20260824,20260825 steps=2k,4k,6k,8k,10k modes=real,hard_shuffled,spatial_shuffled,slot_mean,zero,control noise=per_token"
if (( dry_run )); then
  printf 'GP_SQ3DMIX_INTERVENTION=real GP_SQ3DMIX_NEGATIVE_MAP=%q INFER_NOISE_MODE=per_token bash %q\n' "$negative_root/navtest_2k/hard_negative_map.json" "$project_root/4-infer.sh"
  printf 'python %q --root %q --run-root %q --variant %q --output-csv %q --interventions-csv %q --decision-json %q\n' "$project_root/tools/summarize_gp_sq3dmix_stage_b_v2.py" "$out_root" "$run_root" "$variant" "$result_csv" "$interventions_csv" "$decision_json"
  exit 0
fi

pdms_cache="${PDMS_METRIC_CACHE_PATH:-$DRIVEDREAMER_SHARED_ROOT/navsim_exp/metric_cache_navtest_v1_1}"
epdms_cache="${EPDMS_METRIC_CACHE_PATH:-$DRIVEDREAMER_SHARED_ROOT/navsim_exp/metric_cache_navtest}"
protocol=(python "$project_root/tools/write_gp_sq3dmix_protocol_manifest.py"
  --output "$out_root/protocol.json" --phase stage_b_multiseed_2k
  --code-commit "$commit"
  --input "stage_a_decision=$stage_a_decision"
  --input "datalist=$datalist"
  --input "negative_map=$negative_root/navtest_2k/hard_negative_map.json"
  --input "negative_map_manifest=$negative_root/navtest_2k/manifest.json"
  --input "slot_stats_manifest=$stats_root/manifest.json"
  --input "source_cache_manifest=$source_cache/vggt_dense/manifest.json"
  --input "target_cache_manifest=$dense_cache/vggt_dense/manifest.json"
  --value "variant=$variant" --value "noise_mode=per_token"
  --value "world_size=$world_size" --value "batch_size=$batch_size"
  --value "num_workers=$num_workers")
for seed in 20260824 20260825; do
  for step in 2000 4000 6000 8000 10000; do
    protocol+=(
      --input "gp_${seed}_${step}=$run_root/gp-sq3dmix-stage-b-${variant}-${seed}/checkpoints/steps_${step}_pytorch_model.pt"
      --input "control_${seed}_${step}=$run_root/gp-sq3dmix-stage-b-control-${seed}/checkpoints/steps_${step}_pytorch_model.pt"
    )
  done
done
"${protocol[@]}"
mkdir -p "$out_root/cache_views"
if [[ ! -d "$pdms_cache/metadata" ]]; then
  mkdir -p "$pdms_cache"
  NAVSIM_DEVKIT_ROOT="$NAVSIM_V1_DEVKIT_ROOT" PYTHONPATH="$NAVSIM_V1_DEVKIT_ROOT:${PYTHONPATH:-}" \
    python "$NAVSIM_V1_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
      train_test_split=navtest cache.cache_path="$pdms_cache" cache.force_feature_computation=false \
      navsim_log_path="$NAVSIM_TEST_LOG_ROOT" worker=single_machine_thread_pool worker.max_workers="${CACHE_WORKERS:-8}" worker.use_process_pool=true
fi
if [[ ! -d "$epdms_cache/metadata" ]]; then
  mkdir -p "$epdms_cache"
  python "$project_root/navsim/navsim/planning/script/run_metric_caching.py" \
    train_test_split=navtest metric_cache_path="$epdms_cache" force_feature_computation=false \
    navsim_log_path="$NAVSIM_TEST_LOG_ROOT" original_sensor_path="$NAVSIM_TEST_SENSOR_ROOT" \
    worker=single_machine_thread_pool worker.max_workers="${CACHE_WORKERS:-8}" worker.use_process_pool=true
fi
python "$project_root/tools/filter_navsim_metric_cache.py" --source-root "$pdms_cache" --datalist "$datalist" --output-root "$out_root/cache_views/pdms"
python "$project_root/tools/filter_navsim_metric_cache.py" --source-root "$epdms_cache" --datalist "$datalist" --output-root "$out_root/cache_views/epdms"

score_mode() {
  local step_root="$1" mode="$2" prediction_run="$3"
  local score_root="$step_root/scores/$mode" pdms_work="$score_root/pdms_work" epdms_work="$score_root/epdms_work" pdms_csv epdms_csv
  mkdir -p "$pdms_work" "$epdms_work" "$step_root/logs/$mode"
  PRED_DIR="$prediction_run" SPLIT=test DATALIST="$datalist" METRIC_CACHE_PATH="$out_root/cache_views/pdms" NAVSIM_EVAL_ROOT="$pdms_work" bash "$project_root/5-eval_v1.sh" >"$step_root/logs/$mode/pdms.log" 2>&1
  PRED_DIR="$prediction_run" SPLIT=test DATALIST="$datalist" METRIC_CACHE_PATH="$out_root/cache_views/epdms" NAVSIM_EVAL_ROOT="$epdms_work" bash "$project_root/6-eval_v2.sh" >"$step_root/logs/$mode/epdms.log" 2>&1
  pdms_csv="$(find "$pdms_work" -type f -name '*.csv' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
  epdms_csv="$(find "$epdms_work" -type f -name '*.csv' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
  [[ -n "$pdms_csv" && -n "$epdms_csv" ]] || { echo "Scoring produced no CSV for $mode" >&2; exit 1; }
  cp "$pdms_csv" "$score_root/pdms.csv"
  cp "$epdms_csv" "$score_root/epdms.csv"
}

infer_mode() {
  local step_root="$1" mode="$2" model_dir="$3" step="$4"
  local prediction_root="$step_root/predictions/$mode" intervention="$mode" run_name prediction_run count failed
  [[ "$mode" != control ]] || intervention=real
  mkdir -p "$prediction_root" "$step_root/logs/$mode"
  pids=()
  for ((rank=0; rank<world_size; rank++)); do
    (
      GP_SQ3DMIX_INTERVENTION="$intervention" MODEL_DIR="$model_dir" MODEL_ITER="$step" \
      SPLIT=test DATALIST="$datalist" OUT_DIR="$prediction_root" BATCH_SIZE="$batch_size" NUM_WORKERS="$num_workers" \
      GPU="$rank" RANK="$rank" WORLD_SIZE="$world_size" INFER_SEED="$infer_seed" INFER_NOISE_MODE=per_token \
      VLA_DENSE_CACHE_ALLOW_SUBSET=1 INFER_USE_FEATURE_CACHE=0 \
      NAVSIM_VGGT_DENSE_CACHE_ROOT="$dense_cache" GP_SQ3DMIX_STATS_ROOT="$stats_root" \
      GP_SQ3DMIX_SOURCE_DATALIST="$source_datalist" \
      GP_SQ3DMIX_SOURCE_CACHE_MANIFEST="$source_cache/vggt_dense/manifest.json" \
      GP_SQ3DMIX_SOURCE_CACHE_ROOT="$source_cache" \
      GP_SQ3DMIX_NEGATIVE_MAP="$negative_root/navtest_2k/hard_negative_map.json" \
      GP_SQ3DMIX_NEGATIVE_MAP_MANIFEST="$negative_root/navtest_2k/manifest.json" \
      VLM_ATTN_IMPLEMENTATION=sdpa bash "$project_root/4-infer.sh"
    ) >"$step_root/logs/$mode/infer.rank${rank}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  (( failed == 0 )) || { echo "Inference failed: step=$step mode=$mode" >&2; exit 1; }
  run_name="$(basename -- "$model_dir")"; prediction_run="$prediction_root/${run_name}-step${step}"
  count="$(find "$prediction_run/test" -type f -name '*.npy' | wc -l)"
  [[ "$count" == 2000 ]] || { echo "Prediction count $count != 2000" >&2; exit 1; }
  score_mode "$step_root" "$mode" "$prediction_run"
}

for seed in 20260824 20260825; do
  gp_run="$run_root/gp-sq3dmix-stage-b-${variant}-${seed}"
  control_run="$run_root/gp-sq3dmix-stage-b-control-${seed}"
  for step in 2000 4000 6000 8000 10000; do
    step_root="$out_root/seed${seed}/step${step}"
    for mode in real hard_shuffled spatial_shuffled slot_mean zero; do infer_mode "$step_root" "$mode" "$gp_run" "$step"; done
    infer_mode "$step_root" control "$control_run" "$step"
  done
done
PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}" python "$project_root/tools/summarize_gp_sq3dmix_stage_b_v2.py" \
  --root "$out_root" --run-root "$run_root" --variant "$variant" \
  --output-csv "$result_csv" --interventions-csv "$interventions_csv" --decision-json "$decision_json"
