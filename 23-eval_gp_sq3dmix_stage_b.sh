#!/usr/bin/env bash
# Evaluate matched Stage-B checkpoints every 2k with deterministic interventions.

set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

gp_run="${GP_STAGE_B_RUN_DIR:-}"
control_run="${GP_STAGE_B_CONTROL_RUN_DIR:-}"
gate_report="${GP_STAGE_A_GATE_REPORT:-}"
datalist="${GP_STAGE_B_EVAL_DATALIST:-$project_root/docs/experiments/splits/gp_sq3dmix_navtest_2k.json}"
stats_root="${GP_SQ3DMIX_STATS_ROOT:-$DRIVEDREAMER_SHARED_ROOT/navsim_feature_cache/gp_sq3dmix_slot_stats}"
source_train_cache="${GP_SQ3DMIX_TRAIN_CACHE_ROOT:-$NAVSIM_VGGT_DENSE_CACHE_ROOT}"
source_train_datalist="${GP_SQ3DMIX_TRAIN_DATALIST:-$NAVSIM_DATALIST_PATH}"
out_root="${GP_STAGE_B_EVAL_ROOT:-}"
result_csv="${GP_STAGE_B_RESULT_CSV:-$project_root/docs/experiments/results/gp_sq3dmix_stage_b.csv}"
interventions_csv="${GP_STAGE_B_INTERVENTIONS_CSV:-$project_root/docs/experiments/results/gp_sq3dmix_interventions.csv}"
device_count="${EVAL_DEVICE_COUNT:-2}"
batch_size="${BATCH_SIZE:-4}"
num_workers="${NUM_WORKERS:-2}"
infer_seed="${INFER_SEED:-20260824}"
dry_run=0
while (( $# )); do
  case "$1" in
    --gp-run) gp_run="${2:?}"; shift 2 ;;
    --control-run) control_run="${2:?}"; shift 2 ;;
    --gate-report) gate_report="${2:?}"; shift 2 ;;
    --datalist) datalist="${2:?}"; shift 2 ;;
    --stats-root) stats_root="${2:?}"; shift 2 ;;
    --source-train-cache) source_train_cache="${2:?}"; shift 2 ;;
    --source-train-datalist) source_train_datalist="${2:?}"; shift 2 ;;
    --out-root) out_root="${2:?}"; shift 2 ;;
    --result-csv) result_csv="${2:?}"; shift 2 ;;
    --interventions-csv) interventions_csv="${2:?}"; shift 2 ;;
    --world-size) device_count="${2:?}"; shift 2 ;;
    --batch-size) batch_size="${2:?}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) echo "Usage: bash $project_root/23-eval_gp_sq3dmix_stage_b.sh --gp-run DIR --control-run DIR --gate-report JSON [--dry-run]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$(git -C "$project_root" branch --show-current)" == "feature/gp-sq-3d-mix" ]] || { echo "Wrong DLC-visible branch" >&2; exit 2; }
for path in "$gp_run/config.yaml" "$control_run/config.yaml" "$gate_report" "$datalist" "$stats_root/manifest.json" "$source_train_cache/vggt_dense/manifest.json" "$source_train_datalist"; do
  [[ -e "$path" ]] || { echo "Missing Stage-B evaluation input: $path" >&2; exit 2; }
done
python - "$gate_report" "$(git -C "$project_root" rev-parse HEAD)" <<'PY'
import json,sys
report=json.load(open(sys.argv[1]))
if report.get("all_passed") is not True:
    raise SystemExit("Stage A did not pass; Stage B evaluation is invalid")
if report.get("code_commit") != sys.argv[2]:
    raise SystemExit("Stage A gate report was produced by a different code commit")
PY
for value in "$device_count" "$batch_size" "$num_workers"; do [[ "$value" =~ ^[1-9][0-9]*$ ]] || exit 2; done
steps=(2000 4000 6000 8000 10000)
for step in "${steps[@]}"; do
  [[ -f "$gp_run/checkpoints/steps_${step}_pytorch_model.pt" ]] || { echo "Missing GP step $step" >&2; exit 2; }
  [[ -f "$control_run/checkpoints/steps_${step}_pytorch_model.pt" ]] || { echo "Missing control step $step" >&2; exit 2; }
done
out_root="${out_root:-$NAVSIM_EVAL_ROOT/gp_sq3dmix_stage_b/$(basename -- "$gp_run")-vs-$(basename -- "$control_run")}"
sample_count="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$datalist")"
[[ "$sample_count" == 2000 ]] || { echo "Stage B fixed eval split must contain 2,000 scenes" >&2; exit 2; }
dense_cache="${GP_STAGE_B_DENSE_CACHE_ROOT:-$DRIVEDREAMER_SHARED_ROOT/navsim_feature_cache/vggt_dense_final_crop_navtest}"
pdms_cache="${PDMS_METRIC_CACHE_PATH:-$DRIVEDREAMER_SHARED_ROOT/navsim_exp/metric_cache_navtest_v1_1}"
epdms_cache="${EPDMS_METRIC_CACHE_PATH:-$DRIVEDREAMER_SHARED_ROOT/navsim_exp/metric_cache_navtest}"
echo "[stage-b-eval] code=$project_root gp=$gp_run control=$control_run out=$out_root"
echo "[stage-b-eval] steps=${steps[*]} modes=real,zero,shuffled,slot_mean,control noise=per_token"
if (( dry_run )); then
  printf 'GP_SQ3DMIX_INTERVENTION=real INFER_NOISE_MODE=per_token bash %q\n' "$project_root/4-infer.sh"
  printf 'python %q --root %q --gp-run-dir %q --output-csv %q --interventions-csv %q\n' "$project_root/tools/summarize_gp_sq3dmix_stage_b.py" "$out_root" "$gp_run" "$result_csv" "$interventions_csv"
  exit 0
fi
[[ ! -e "$out_root" ]] || { echo "Refusing to overwrite Stage-B evaluation directory" >&2; exit 2; }
for result_path in "$result_csv" "$interventions_csv"; do
  if [[ -e "$result_path" ]]; then
    python - "$result_path" <<'PY'
import csv,sys
with open(sys.argv[1],newline="",encoding="utf-8") as stream: rows=list(csv.DictReader(stream))
if len(rows) != 1 or rows[0].get("status") != "not_run":
    raise SystemExit(f"Refusing to overwrite completed result CSV: {sys.argv[1]}")
PY
  fi
done
mkdir -p "$out_root/cache_views"
python - "$out_root/run_manifest.json" "$gp_run/config.yaml" "$control_run/config.yaml" \
  "$gate_report" "$datalist" "$stats_root/manifest.json" \
  "$source_train_cache/vggt_dense/manifest.json" "$(git -C "$project_root" rev-parse HEAD)" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
out,gp_cfg,control_cfg,gate,datalist,stats,cache,commit=sys.argv[1:]
payload={
    "schema_version":1,
    "complete":False,
    "commit":commit,
    "noise_mode":"per_token",
    "gp_config_sha256":sha(gp_cfg),
    "control_config_sha256":sha(control_cfg),
    "stage_a_gate_report_sha256":sha(gate),
    "datalist_sha256":sha(datalist),
    "slot_stats_manifest_sha256":sha(stats),
    "source_cache_manifest_sha256":sha(cache),
}
tmp=out+f".tmp-{os.getpid()}"
Path(tmp).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
os.replace(tmp,out)
PY
if [[ ! -f "$dense_cache/vggt_dense/manifest.json" ]]; then
  mkdir -p "$dense_cache"
  SPLIT=test NAVSIM_DATALIST_PATH="$datalist" NAVSIM_TRAINVAL_SENSOR_ROOT="$NAVSIM_TEST_SENSOR_ROOT" \
    NAVSIM_VGGT_DENSE_CACHE_ROOT="$dense_cache" VGGT_DENSE_CACHE_NUM_PROCESSES="$device_count" \
    VGGT_DENSE_CACHE_BATCH_SIZE=1 VGGT_DENSE_CACHE_FULL=1 bash "$project_root/11-precompute_vggt_dense_cache.sh"
fi
[[ -f "$dense_cache/vggt_dense/manifest.json" ]] || {
  echo "Dense evaluation cache generation did not produce a manifest" >&2
  exit 2
}
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
for cache in "$pdms_cache" "$epdms_cache"; do [[ -d "$cache/metadata" ]] || { echo "Metric cache generation failed: $cache" >&2; exit 2; }; done
python "$project_root/tools/filter_navsim_metric_cache.py" --source-root "$pdms_cache" --datalist "$datalist" --output-root "$out_root/cache_views/pdms"
python "$project_root/tools/filter_navsim_metric_cache.py" --source-root "$epdms_cache" --datalist "$datalist" --output-root "$out_root/cache_views/epdms"

score_mode() {
  local step_root="$1" mode="$2" prediction_run="$3"
  local score_root="$step_root/scores/$mode" pdms_work epdms_work pdms_csv epdms_csv
  score_root="$step_root/scores/$mode"; pdms_work="$score_root/pdms_work"; epdms_work="$score_root/epdms_work"
  mkdir -p "$pdms_work" "$epdms_work" "$step_root/logs/$mode"
  PRED_DIR="$prediction_run" SPLIT=test DATALIST="$datalist" METRIC_CACHE_PATH="$out_root/cache_views/pdms" NAVSIM_EVAL_ROOT="$pdms_work" bash "$project_root/5-eval_v1.sh" >"$step_root/logs/$mode/pdms.log" 2>&1
  PRED_DIR="$prediction_run" SPLIT=test DATALIST="$datalist" METRIC_CACHE_PATH="$out_root/cache_views/epdms" NAVSIM_EVAL_ROOT="$epdms_work" bash "$project_root/6-eval_v2.sh" >"$step_root/logs/$mode/epdms.log" 2>&1
  pdms_csv="$(find "$pdms_work" -type f -name '*.csv' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
  epdms_csv="$(find "$epdms_work" -type f -name '*.csv' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
  cp "$pdms_csv" "$score_root/pdms.csv"; cp "$epdms_csv" "$score_root/epdms.csv"
}

infer_mode() {
  local step_root="$1" mode="$2" model_dir="$3" step="$4"
  local prediction_root="$step_root/predictions/$mode" run_name prediction_run count failed intervention
  intervention="$mode"; [[ "$mode" == control ]] && intervention=real
  mkdir -p "$prediction_root" "$step_root/logs/$mode"
  pids=()
  for ((rank=0; rank<device_count; rank++)); do
    (
      GP_SQ3DMIX_INTERVENTION="$intervention" MODEL_DIR="$model_dir" MODEL_ITER="$step" SPLIT=test \
      DATALIST="$datalist" OUT_DIR="$prediction_root" BATCH_SIZE="$batch_size" NUM_WORKERS="$num_workers" \
      GPU="$rank" RANK="$rank" WORLD_SIZE="$device_count" INFER_SEED="$infer_seed" INFER_NOISE_MODE=per_token \
      VLA_TOPOLOGY_INDEPENDENT_SHUFFLE=1 VLA_DENSE_CACHE_ALLOW_SUBSET=1 \
      INFER_USE_FEATURE_CACHE=0 \
      NAVSIM_VGGT_DENSE_CACHE_ROOT="$dense_cache" GP_SQ3DMIX_STATS_ROOT="$stats_root" \
      GP_SQ3DMIX_SOURCE_DATALIST="$source_train_datalist" \
      GP_SQ3DMIX_SOURCE_CACHE_MANIFEST="$source_train_cache/vggt_dense/manifest.json" \
      VLM_ATTN_IMPLEMENTATION=sdpa bash "$project_root/4-infer.sh"
    ) >"$step_root/logs/$mode/infer.rank${rank}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  (( failed == 0 )) || { echo "Inference failed: step=$step mode=$mode" >&2; exit 1; }
  run_name="$(basename -- "$model_dir")"; prediction_run="$prediction_root/${run_name}-step${step}"
  count="$(find "$prediction_run/test" -type f -name '*.npy' | wc -l)"
  [[ "$count" == "$sample_count" ]] || { echo "Prediction count mismatch" >&2; exit 1; }
  score_mode "$step_root" "$mode" "$prediction_run"
}

for step in "${steps[@]}"; do
  step_root="$out_root/step$step"; mkdir -p "$step_root"
  for mode in real zero shuffled slot_mean; do infer_mode "$step_root" "$mode" "$gp_run" "$step"; done
  infer_mode "$step_root" control "$control_run" "$step"
done
python "$project_root/tools/summarize_gp_sq3dmix_stage_b.py" --root "$out_root" --gp-run-dir "$gp_run" \
  --output-csv "$result_csv" --interventions-csv "$interventions_csv" \
  --decision-json "$out_root/go_no_go.json"
python - "$out_root/run_manifest.json" "$dense_cache/vggt_dense/manifest.json" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
path=Path(sys.argv[1]); payload=json.loads(path.read_text()); payload["complete"]=True
payload["evaluation_cache_manifest_sha256"]=hashlib.sha256(
    Path(sys.argv[2]).read_bytes()
).hexdigest()
tmp=path.with_name(path.name+f".tmp-{os.getpid()}")
tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); os.replace(tmp,path)
PY
echo "[stage-b-eval] finished; no 100k job was launched"
