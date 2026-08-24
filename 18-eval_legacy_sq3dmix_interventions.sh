#!/usr/bin/env bash
# Paired real/zero/shuffled evaluation of the legacy 100k SQ-3D-Mix checkpoint.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

dry_run=0
resume=0
full_navtest=0
checkpoint="${LEGACY_SQ3DMIX_CHECKPOINT:-$DRIVEDREAMER_SHARED_ROOT/navsim_exp/sq3dmix-gated-real-100000-dlc-20260821_092515/checkpoints/steps_100000_pytorch_model.pt}"
datalist="${LEGACY_INTERVENTION_DATALIST:-$project_root/docs/experiments/splits/gp_sq3dmix_navtest_2k.json}"
out_root="${LEGACY_INTERVENTION_OUT_ROOT:-$NAVSIM_EVAL_ROOT/legacy_sq3dmix_interventions/gp2k}"
device_count="${EVAL_DEVICE_COUNT:-2}"
batch_size="${BATCH_SIZE:-4}"
num_workers="${NUM_WORKERS:-2}"
infer_seed="${INFER_SEED:-20260824}"

usage() {
  cat <<EOF
Usage: bash $project_root/18-eval_legacy_sq3dmix_interventions.sh [options]
  --checkpoint FILE   legacy SQ-3D-Mix checkpoint
  --datalist FILE     token list (default: fixed GP navtest 2k)
  --out-root DIR      new immutable evaluation directory
  --world-size N      visible accelerator count (default: 2)
  --batch-size N      per-device inference batch (default: 4)
  --num-workers N     dataloader workers per rank (default: 2)
  --full              evaluate all 12,146 navtest scenes
  --resume            continue an incomplete immutable output directory
  --dry-run           print resolved commands without writes/imports
EOF
}

while (( $# )); do
  case "$1" in
    --checkpoint) checkpoint="${2:?}"; shift 2 ;;
    --datalist) datalist="${2:?}"; shift 2 ;;
    --out-root) out_root="${2:?}"; shift 2 ;;
    --world-size) device_count="${2:?}"; shift 2 ;;
    --batch-size) batch_size="${2:?}"; shift 2 ;;
    --num-workers) num_workers="${2:?}"; shift 2 ;;
    --full) full_navtest=1; shift ;;
    --resume) resume=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if (( full_navtest )); then
  datalist="${LEGACY_FULL_DATALIST:-$DRIVEDREAMER_SHARED_ROOT/test_meta.json}"
  out_root="${LEGACY_FULL_INTERVENTION_OUT_ROOT:-$NAVSIM_EVAL_ROOT/legacy_sq3dmix_interventions/full}"
fi
for value in "$device_count" "$batch_size" "$num_workers"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Counts must be positive integers" >&2; exit 2; }
done

branch="$(git -C "$project_root" branch --show-current)"
commit="$(git -C "$project_root" rev-parse HEAD)"
[[ "$branch" == "feature/gp-sq-3d-mix" ]] || {
  echo "Wrong shared checkout branch: $branch (expected feature/gp-sq-3d-mix)" >&2
  exit 2
}
[[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 2; }
[[ -f "$datalist" ]] || { echo "Missing datalist: $datalist" >&2; exit 2; }
[[ -d "$BASE_VLM" ]] || { echo "Missing BASE_VLM: $BASE_VLM" >&2; exit 2; }

sample_count="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$datalist")"
if (( ! full_navtest )) && [[ "$sample_count" != "2000" ]]; then
  echo "Fixed intervention subset must contain exactly 2,000 scenes" >&2
  exit 2
fi
dense_cache="${LEGACY_INTERVENTION_DENSE_CACHE_ROOT:-$DRIVEDREAMER_SHARED_ROOT/navsim_feature_cache/vggt_dense_final_crop_navtest}"
pdms_cache="${PDMS_METRIC_CACHE_PATH:-$DRIVEDREAMER_SHARED_ROOT/navsim_exp/metric_cache_navtest_v1_1}"
epdms_cache="${EPDMS_METRIC_CACHE_PATH:-$DRIVEDREAMER_SHARED_ROOT/navsim_exp/metric_cache_navtest}"
result_csv="${LEGACY_INTERVENTION_RESULT_CSV:-$project_root/docs/experiments/results/legacy_sq3dmix_interventions_${sample_count}.csv}"

echo "[legacy-interventions] code=$project_root"
echo "[legacy-interventions] branch=$branch commit=$commit"
echo "[legacy-interventions] checkpoint=$checkpoint"
echo "[legacy-interventions] datalist=$datalist scenes=$sample_count"
echo "[legacy-interventions] dense_cache=$dense_cache"
echo "[legacy-interventions] out=$out_root noise=per_token seed=$infer_seed"

if (( dry_run )); then
  printf 'SQ3DMIX_INTERVENTION=%q INFER_NOISE_MODE=per_token INFER_SEED=%q bash %q\n' real "$infer_seed" "$project_root/4-infer.sh"
  printf 'SQ3DMIX_INTERVENTION=%q INFER_NOISE_MODE=per_token INFER_SEED=%q bash %q\n' zero "$infer_seed" "$project_root/4-infer.sh"
  printf 'SQ3DMIX_INTERVENTION=%q INFER_NOISE_MODE=per_token INFER_SEED=%q bash %q\n' shuffled "$infer_seed" "$project_root/4-infer.sh"
  printf 'python %q --root %q --datalist %q --output %q\n' "$project_root/tools/summarize_gp_sq3dmix_interventions.py" "$out_root" "$datalist" "$result_csv"
  exit 0
fi

if (( resume )); then
  [[ -d "$out_root" ]] || { echo "Resume output does not exist: $out_root" >&2; exit 2; }
else
  [[ ! -e "$out_root" ]] || { echo "Refusing to overwrite: $out_root" >&2; exit 2; }
fi
[[ ! -e "$result_csv" ]] || { echo "Refusing to overwrite: $result_csv" >&2; exit 2; }
mkdir -p "$out_root" "$out_root/logs" "$out_root/cache_views"

if [[ ! -f "$dense_cache/vggt_dense/manifest.json" ]]; then
  mkdir -p "$dense_cache"
  SPLIT=test NAVSIM_DATALIST_PATH="$datalist" \
    NAVSIM_TRAINVAL_SENSOR_ROOT="$NAVSIM_TEST_SENSOR_ROOT" \
    NAVSIM_VGGT_DENSE_CACHE_ROOT="$dense_cache" \
    VGGT_DENSE_CACHE_NUM_PROCESSES="$device_count" \
    VGGT_DENSE_CACHE_BATCH_SIZE=1 VGGT_DENSE_CACHE_FULL=1 \
    bash "$project_root/11-precompute_vggt_dense_cache.sh"
fi
[[ -f "$dense_cache/vggt_dense/manifest.json" ]] || {
  echo "Dense cache generation did not produce a manifest" >&2; exit 2;
}
if [[ ! -d "$pdms_cache/metadata" ]]; then
  mkdir -p "$pdms_cache"
  (
    export NAVSIM_DEVKIT_ROOT="$NAVSIM_V1_DEVKIT_ROOT"
    export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
    python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
      train_test_split=navtest cache.cache_path="$pdms_cache" \
      cache.force_feature_computation=false navsim_log_path="$NAVSIM_TEST_LOG_ROOT" \
      worker=single_machine_thread_pool worker.max_workers="${CACHE_WORKERS:-8}" worker.use_process_pool=true
  )
fi
if [[ ! -d "$epdms_cache/metadata" ]]; then
  mkdir -p "$epdms_cache"
  python "$project_root/navsim/navsim/planning/script/run_metric_caching.py" \
    train_test_split=navtest metric_cache_path="$epdms_cache" \
    force_feature_computation=false navsim_log_path="$NAVSIM_TEST_LOG_ROOT" \
    original_sensor_path="$NAVSIM_TEST_SENSOR_ROOT" \
    worker=single_machine_thread_pool worker.max_workers="${CACHE_WORKERS:-8}" worker.use_process_pool=true
fi
for cache in "$pdms_cache" "$epdms_cache"; do [[ -d "$cache/metadata" ]] || { echo "Metric cache generation failed: $cache" >&2; exit 2; }; done
if [[ ! -f "$out_root/cache_views/pdms/metadata/cache.csv" ]]; then
  python "$project_root/tools/filter_navsim_metric_cache.py" \
    --source-root "$pdms_cache" --datalist "$datalist" \
    --output-root "$out_root/cache_views/pdms"
fi
if [[ ! -f "$out_root/cache_views/epdms/metadata/cache.csv" ]]; then
  python "$project_root/tools/filter_navsim_metric_cache.py" \
    --source-root "$epdms_cache" --datalist "$datalist" \
    --output-root "$out_root/cache_views/epdms"
fi

mkdir -p "$out_root/predictions" "$out_root/scores"
run_name="$(basename -- "$(dirname -- "$(dirname -- "$checkpoint")")")"
step="$(basename -- "$checkpoint" | sed -nE 's/^steps_([0-9]+)_pytorch_model\.pt$/\1/p')"
[[ -n "$step" ]] || { echo "Checkpoint name does not encode a step" >&2; exit 2; }
for mode in real zero shuffled; do
  mode_prediction_root="$out_root/predictions/$mode"
  mkdir -p "$mode_prediction_root" "$out_root/logs/$mode" "$out_root/scores/$mode"
  pids=()
  for ((rank=0; rank<device_count; rank++)); do
    (
      SQ3DMIX_INTERVENTION="$mode" \
      MODEL_DIR="$(dirname -- "$(dirname -- "$checkpoint")")" MODEL_ITER="$step" \
      SPLIT=test DATALIST="$datalist" OUT_DIR="$mode_prediction_root" \
      BATCH_SIZE="$batch_size" NUM_WORKERS="$num_workers" GPU="$rank" \
      RANK="$rank" WORLD_SIZE="$device_count" OVERWRITE=0 \
      INFER_USE_FEATURE_CACHE=0 INFER_SEED="$infer_seed" \
      INFER_NOISE_MODE=per_token VLA_TOPOLOGY_INDEPENDENT_SHUFFLE=1 \
      VLA_DENSE_CACHE_ALLOW_SUBSET=1 \
      NAVSIM_VGGT_DENSE_CACHE_ROOT="$dense_cache" \
      VLM_ATTN_IMPLEMENTATION=sdpa bash "$project_root/4-infer.sh"
    ) >"$out_root/logs/$mode/infer.rank${rank}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for ((rank=0; rank<device_count; rank++)); do
    wait "${pids[$rank]}" || failed=1
  done
  (( failed == 0 )) || { echo "$mode inference failed" >&2; exit 1; }
  prediction_run="$mode_prediction_root/${run_name}-step${step}"
  count="$(find "$prediction_run/test" -maxdepth 1 -name '*.npy' -type f | wc -l)"
  [[ "$count" == "$sample_count" ]] || { echo "$mode prediction count $count != $sample_count" >&2; exit 1; }

  pdms_work="$out_root/scores/$mode/pdms_work"
  epdms_work="$out_root/scores/$mode/epdms_work"
  mkdir -p "$pdms_work" "$epdms_work"
  if [[ ! -f "$out_root/scores/$mode/pdms.csv" ]]; then
    PRED_DIR="$prediction_run" SPLIT=test DATALIST="$datalist" \
      METRIC_CACHE_PATH="$out_root/cache_views/pdms" NAVSIM_EVAL_ROOT="$pdms_work" \
      bash "$project_root/5-eval_v1.sh" |& tee "$out_root/logs/$mode/pdms.log"
    pdms_csv="$(find "$pdms_work" -type f -name '*.csv' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
    [[ -n "$pdms_csv" ]] || { echo "$mode PDMS scoring produced no CSV" >&2; exit 1; }
    cp "$pdms_csv" "$out_root/scores/$mode/pdms.csv"
  fi
  if [[ ! -f "$out_root/scores/$mode/epdms.csv" ]]; then
    PRED_DIR="$prediction_run" SPLIT=test DATALIST="$datalist" \
      METRIC_CACHE_PATH="$out_root/cache_views/epdms" NAVSIM_EVAL_ROOT="$epdms_work" \
      bash "$project_root/6-eval_v2.sh" |& tee "$out_root/logs/$mode/epdms.log"
    epdms_csv="$(find "$epdms_work" -type f -name '*.csv' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
    [[ -n "$epdms_csv" ]] || { echo "$mode EPDMS scoring produced no CSV" >&2; exit 1; }
    cp "$epdms_csv" "$out_root/scores/$mode/epdms.csv"
  fi
done

python "$project_root/tools/summarize_gp_sq3dmix_interventions.py" \
  --root "$out_root" --datalist "$datalist" --output "$result_csv"
python - "$out_root/run_manifest.json" "$checkpoint" "$datalist" "$dense_cache/vggt_dense/manifest.json" "$commit" "$(dirname -- "$(dirname -- "$checkpoint")")/config.yaml" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
def sha(p):
    digest = hashlib.sha256()
    with Path(p).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
out, ckpt, datalist, cache_manifest, commit, config = sys.argv[1:]
payload = {"complete": True, "commit": commit, "checkpoint": str(Path(ckpt).resolve()),
           "checkpoint_sha256": sha(ckpt), "datalist_sha256": sha(datalist),
           "cache_manifest_sha256": sha(cache_manifest), "resolved_config_sha256": sha(config),
           "noise_mode": "per_token"}
tmp = out + f".tmp-{os.getpid()}"
Path(tmp).write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
os.replace(tmp, out)
PY
echo "[legacy-interventions] complete result=$result_csv"
