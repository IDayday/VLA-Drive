#!/usr/bin/env bash
# Evaluate a Stage-C matched checkpoint; a failed 10k result blocks continuation.

set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"
permission="${GP_FORMAL_PERMISSION_REPORT:-}"
stage_a_decision="${GP_STAGE_A_V2_DECISION:-}"
seed=20260826
step=10000
world_size="${EVAL_DEVICE_COUNT:-1}"
batch_size="${BATCH_SIZE:-2}"
num_workers="${NUM_WORKERS:-2}"
dry_run=0
operation=step
selection=""
while (( $# )); do
  case "$1" in
    --permission-report) permission="${2:?}"; shift 2 ;;
    --stage-a-decision) stage_a_decision="${2:?}"; shift 2 ;;
    --seed) seed="${2:?}"; shift 2 ;;
    --step) step="${2:?}"; shift 2 ;;
    --world-size) world_size="${2:?}"; shift 2 ;;
    --batch-size) batch_size="${2:?}"; shift 2 ;;
    --num-workers) num_workers="${2:?}"; shift 2 ;;
    --select-final) operation=select; shift ;;
    --full-navtest) operation=full; shift ;;
    --gate-50k) operation=midterm; shift ;;
    --selection) selection="${2:?}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) echo "Usage: bash $project_root/32-eval_gp_sq3dmix_formal_30k.sh [--seed N --step 10000|20000|30000|50000 | --select-final | --full-navtest --selection JSON | --gate-50k --selection JSON] --permission-report JSON --stage-a-decision JSON [--dry-run]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$seed" == 20260826 || "$seed" == 20260827 ]] || { echo "Invalid formal seed" >&2; exit 2; }
[[ "$step" == 10000 || "$step" == 20000 || "$step" == 30000 || "$step" == 50000 ]] || { echo "Invalid formal evaluation step" >&2; exit 2; }
for value in "$world_size" "$batch_size" "$num_workers"; do [[ "$value" =~ ^[1-9][0-9]*$ ]] || exit 2; done
for path in "$permission" "$stage_a_decision"; do [[ -f "$path" ]] || { echo "Missing formal binding: $path" >&2; exit 2; }; done
python - "$permission" <<'PY'
import json,sys
if json.load(open(sys.argv[1])).get('formal_30k_allowed') is not True: raise SystemExit('formal_30k_allowed=false')
PY
variant="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["all_passed"]; print(d["selected_variant"])' "$stage_a_decision")"
run_root="${GP_SQ3DMIX_V2_RUN_ROOT:-$NAVSIM_EXP_ROOT/gp_sq3dmix_stage_a_v2}"
eval_base="${GP_SQ3DMIX_V2_EVAL_ROOT:-$NAVSIM_EVAL_ROOT/gp_sq3dmix_stage_a_v2_eval}/formal_30k"
branch="$(git -C "$project_root" branch --show-current)"; commit="$(git -C "$project_root" rev-parse HEAD)"; [[ "$branch" == feature/gp-sq-3d-mix-stage-a-v2 ]] || exit 2
[[ -z "$(git -C "$project_root" status --porcelain)" ]] || { echo "Formal evaluation requires a clean worktree" >&2; exit 2; }
if [[ "$operation" == select ]]; then
  selection="${selection:-$eval_base/formal_30k_selection.json}"
  output_csv="$eval_base/gp_sq3dmix_formal_30k.csv"
  command=(python "$project_root/tools/summarize_gp_sq3dmix_formal_v2.py" select
    --root "$eval_base" --run-root "$run_root" --stage-a-decision "$stage_a_decision"
    --output-json "$selection" --output-csv "$output_csv")
  printf '[gp-formal-select] command: PYTHONPATH=%q ' "$project_root${PYTHONPATH:+:$PYTHONPATH}"
  printf '%q ' "${command[@]}"; printf '\n'
  (( dry_run )) && exit 0
  PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}" "${command[@]}"
  exit 0
fi
if [[ "$operation" == full ]]; then
  selection="${selection:-$eval_base/formal_30k_selection.json}"
  command=(bash "$project_root/tools/eval_gp_sq3dmix_formal_full_navtest.sh"
    --selection "$selection" --permission-before "$permission"
    --world-size "$world_size" --batch-size "$batch_size" --num-workers "$num_workers")
  (( dry_run )) && command+=(--dry-run)
  "${command[@]}"
  exit 0
fi
if [[ "$operation" == midterm ]]; then
  selection="${selection:-$eval_base/formal_30k_selection.json}"
  midterm_root="${GP_SQ3DMIX_V2_EVAL_ROOT:-$NAVSIM_EVAL_ROOT/gp_sq3dmix_stage_a_v2_eval}/formal_100k_midterm"
  command=(python "$project_root/tools/summarize_gp_sq3dmix_formal_v2.py" midterm
    --root "$midterm_root" --selection "$selection" --permission "$permission"
    --output-json "$midterm_root/formal_50k_midterm_decision.json")
  printf '[gp-formal-50k] command: PYTHONPATH=%q ' "$project_root${PYTHONPATH:+:$PYTHONPATH}"
  printf '%q ' "${command[@]}"; printf '\n'
  (( dry_run )) && exit 0
  PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}" "${command[@]}"
  exit 0
fi
segment=10k
[[ "$step" != 20000 && "$step" != 30000 ]] || segment=30k
[[ "$step" != 50000 ]] || segment=50k
if [[ "$step" == 50000 ]]; then
  python - "$permission" <<'PY'
import json,sys
if json.load(open(sys.argv[1])).get('formal_100k_allowed') is not True: raise SystemExit('formal_100k_allowed=false; 50k evaluation forbidden')
PY
fi
gp_run="$run_root/gp-sq3dmix-stage-c-${segment}-${variant}-${seed}"
control_run="$run_root/gp-sq3dmix-stage-c-${segment}-control-${seed}"
for run in "$gp_run" "$control_run"; do [[ -f "$run/checkpoints/steps_${step}_pytorch_model.pt" && -f "$run/config.yaml" ]] || { echo "Missing formal checkpoint: $run step=$step" >&2; exit 2; }; done
datalist="$project_root/docs/experiments/splits/gp_sq3dmix_navtest_2k.json"
negative_root="${GP_SQ3DMIX_V2_NEGATIVE_MAP_ROOT:-$DRIVEDREAMER_SHARED_ROOT/navsim_feature_cache/gp_sq3dmix_stage_a_v2_negative_maps}"
stats_root="${GP_SQ3DMIX_V2_STATS_ROOT:-$DRIVEDREAMER_SHARED_ROOT/navsim_feature_cache/gp_sq3dmix_stage_a_v2_slot_stats}"
source_cache="${GP_SQ3DMIX_TRAIN_CACHE_ROOT:-$NAVSIM_VGGT_DENSE_CACHE_ROOT}"; source_datalist="${GP_SQ3DMIX_TRAIN_DATALIST:-$NAVSIM_DATALIST_PATH}"; dense_cache="${GP_SQ3DMIX_TEST_CACHE_ROOT:-$NAVSIM_VGGT_DENSE_TEST_CACHE_ROOT}"
if [[ "$step" == 50000 ]]; then
  out_root="${GP_SQ3DMIX_V2_EVAL_ROOT:-$NAVSIM_EVAL_ROOT/gp_sq3dmix_stage_a_v2_eval}/formal_100k_midterm/seed${seed}/step${step}"
else
  out_root="$eval_base/seed${seed}/step${step}"
fi
[[ ! -e "$out_root" ]] || { echo "Refusing to overwrite formal evaluation" >&2; exit 2; }
for path in "$negative_root/navtest_2k/hard_negative_map.json" "$negative_root/navtest_2k/manifest.json" "$stats_root/manifest.json" "$source_cache/vggt_dense/manifest.json" "$dense_cache/vggt_dense/manifest.json"; do [[ -f "$path" ]] || { echo "Missing formal eval input: $path" >&2; exit 2; }; done
echo "[gp-formal-eval] seed=$seed step=$step variant=$variant modes=real,hard_shuffled,spatial_shuffled,control noise=per_token"
if (( dry_run )); then
  printf 'GP_SQ3DMIX_INTERVENTION=real INFER_NOISE_MODE=per_token bash %q\n' "$project_root/4-infer.sh"
  printf 'python %q --root %q --gp-run %q --variant %q --seed %q --step %q --output-json %q --output-csv %q\n' "$project_root/tools/summarize_gp_sq3dmix_formal_step_v2.py" "$out_root" "$gp_run" "$variant" "$seed" "$step" "$out_root/decision.json" "$out_root/results.csv"
  exit 0
fi
python "$project_root/tools/write_gp_sq3dmix_protocol_manifest.py" \
  --output "$out_root/protocol.json" --phase "formal_step_${step}" \
  --code-commit "$commit" \
  --input "permission=$permission" --input "stage_a_decision=$stage_a_decision" \
  --input "datalist=$datalist" \
  --input "negative_map=$negative_root/navtest_2k/hard_negative_map.json" \
  --input "negative_map_manifest=$negative_root/navtest_2k/manifest.json" \
  --input "slot_stats_manifest=$stats_root/manifest.json" \
  --input "source_cache_manifest=$source_cache/vggt_dense/manifest.json" \
  --input "target_cache_manifest=$dense_cache/vggt_dense/manifest.json" \
  --input "gp_checkpoint=$gp_run/checkpoints/steps_${step}_pytorch_model.pt" \
  --input "control_checkpoint=$control_run/checkpoints/steps_${step}_pytorch_model.pt" \
  --value "variant=$variant" --value "seed=$seed" --value "step=$step" \
  --value "noise_mode=per_token" --value "world_size=$world_size" \
  --value "batch_size=$batch_size" --value "num_workers=$num_workers"
pdms_cache="${PDMS_METRIC_CACHE_PATH:-$DRIVEDREAMER_SHARED_ROOT/navsim_exp/metric_cache_navtest_v1_1}"; epdms_cache="${EPDMS_METRIC_CACHE_PATH:-$DRIVEDREAMER_SHARED_ROOT/navsim_exp/metric_cache_navtest}"
mkdir -p "$out_root/cache_views"
python "$project_root/tools/filter_navsim_metric_cache.py" --source-root "$pdms_cache" --datalist "$datalist" --output-root "$out_root/cache_views/pdms"
python "$project_root/tools/filter_navsim_metric_cache.py" --source-root "$epdms_cache" --datalist "$datalist" --output-root "$out_root/cache_views/epdms"

score_mode() {
  local mode="$1" prediction_run="$2" score_root="$out_root/scores/$mode" pdms_work="$score_root/pdms_work" epdms_work="$score_root/epdms_work" pdms_csv epdms_csv
  mkdir -p "$pdms_work" "$epdms_work" "$out_root/logs/$mode"
  PRED_DIR="$prediction_run" SPLIT=test DATALIST="$datalist" METRIC_CACHE_PATH="$out_root/cache_views/pdms" NAVSIM_EVAL_ROOT="$pdms_work" bash "$project_root/5-eval_v1.sh" >"$out_root/logs/$mode/pdms.log" 2>&1
  PRED_DIR="$prediction_run" SPLIT=test DATALIST="$datalist" METRIC_CACHE_PATH="$out_root/cache_views/epdms" NAVSIM_EVAL_ROOT="$epdms_work" bash "$project_root/6-eval_v2.sh" >"$out_root/logs/$mode/epdms.log" 2>&1
  pdms_csv="$(find "$pdms_work" -type f -name '*.csv' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"; epdms_csv="$(find "$epdms_work" -type f -name '*.csv' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"; cp "$pdms_csv" "$score_root/pdms.csv"; cp "$epdms_csv" "$score_root/epdms.csv"
}
infer_mode() {
  local mode="$1" model_dir="$2" intervention="$mode" prediction_root="$out_root/predictions/$mode" run_name prediction_run failed
  [[ "$mode" != control ]] || intervention=real; mkdir -p "$prediction_root" "$out_root/logs/$mode"; pids=()
  for ((rank=0; rank<world_size; rank++)); do
    (GP_SQ3DMIX_INTERVENTION="$intervention" MODEL_DIR="$model_dir" MODEL_ITER="$step" SPLIT=test DATALIST="$datalist" OUT_DIR="$prediction_root" BATCH_SIZE="$batch_size" NUM_WORKERS="$num_workers" GPU="$rank" RANK="$rank" WORLD_SIZE="$world_size" INFER_SEED=20260824 INFER_NOISE_MODE=per_token INFER_USE_FEATURE_CACHE=0 NAVSIM_VGGT_DENSE_CACHE_ROOT="$dense_cache" GP_SQ3DMIX_STATS_ROOT="$stats_root" GP_SQ3DMIX_SOURCE_DATALIST="$source_datalist" GP_SQ3DMIX_SOURCE_CACHE_MANIFEST="$source_cache/vggt_dense/manifest.json" GP_SQ3DMIX_SOURCE_CACHE_ROOT="$source_cache" GP_SQ3DMIX_NEGATIVE_MAP="$negative_root/navtest_2k/hard_negative_map.json" GP_SQ3DMIX_NEGATIVE_MAP_MANIFEST="$negative_root/navtest_2k/manifest.json" VLM_ATTN_IMPLEMENTATION=sdpa bash "$project_root/4-infer.sh") >"$out_root/logs/$mode/infer.rank${rank}.log" 2>&1 & pids+=("$!")
  done
  failed=0; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; (( failed == 0 )) || exit 1
  run_name="$(basename -- "$model_dir")"; prediction_run="$prediction_root/${run_name}-step${step}"; [[ "$(find "$prediction_run/test" -type f -name '*.npy' | wc -l)" == 2000 ]] || exit 1; score_mode "$mode" "$prediction_run"
}
for mode in real hard_shuffled spatial_shuffled; do infer_mode "$mode" "$gp_run"; done; infer_mode control "$control_run"
PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}" python "$project_root/tools/summarize_gp_sq3dmix_formal_step_v2.py" --root "$out_root" --gp-run "$gp_run" --variant "$variant" --seed "$seed" --step "$step" --output-json "$out_root/decision.json" --output-csv "$out_root/results.csv"
