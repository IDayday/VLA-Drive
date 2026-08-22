#!/usr/bin/env bash
# Evaluate every existing SQ-3D-Mix gated checkpoint on NAVSIM navtest.
#
# The launcher discovers the newest sq3dmix-gated-real-* run, shards inference
# over configurable accelerators, reuses complete predictions, and computes
# both NAVSIM v1.1 PDMS and NAVSIM v2 EPDMS. Missing dense/metric caches are
# built before inference. All machine paths continue to come from env.local.sh.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

usage() {
  cat <<EOF
Usage: bash $project_root/17-eval_sq3dmix_gated_all_ckpts_dlc.sh [options]

Options:
  --model-dir DIR       SQ-3D-Mix run directory (default: newest gated run)
  --steps "LIST"        Space/comma separated steps (default: every ckpt found)
  --preflight           Print resolved paths/checkpoints without running jobs
  -h, --help            Show this help

Useful environment overrides:
  EVAL_DEVICE_COUNT=16
  EVAL_DEVICE_IDS=0,1,...,15
  BATCH_SIZE=8
  NUM_WORKERS=2
  CACHE_WORKERS=8
  SQ3DMIX_EVAL_OUT_ROOT=/path/to/evaluation-root
  NAVSIM_VGGT_DENSE_TEST_CACHE_ROOT=/path/to/navtest-dense-cache
  PDMS_METRIC_CACHE_PATH=/path/to/v1.1-cache
  EPDMS_METRIC_CACHE_PATH=/path/to/v2-cache
  OVERWRITE=1            Regenerate predictions
  RESCORE=1              Recompute PDMS and EPDMS
EOF
}

model_dir="${MODEL_DIR:-}"
checkpoint_steps="${CHECKPOINT_STEPS:-}"
preflight_only="${EVAL_PREFLIGHT_ONLY:-0}"
while (( $# > 0 )); do
  case "$1" in
    --model-dir)
      model_dir="${2:?--model-dir requires a value}"
      shift 2
      ;;
    --steps)
      checkpoint_steps="${2:?--steps requires a value}"
      shift 2
      ;;
    --preflight)
      preflight_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$model_dir" ]]; then
  model_dir="$(
    find "$NAVSIM_EXP_ROOT" -mindepth 1 -maxdepth 1 -type d \
      -name 'sq3dmix-gated-real-*' -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr | head -n 1 | cut -d' ' -f2-
  )"
fi
if [[ -z "$model_dir" || ! -f "$model_dir/config.yaml" ]]; then
  echo "No SQ-3D-Mix run with config.yaml found under $NAVSIM_EXP_ROOT" >&2
  exit 2
fi
model_dir="$(realpath "$model_dir")"
run_name="$(basename -- "$model_dir")"

framework_name="$(python - "$model_dir/config.yaml" <<'PY'
from omegaconf import OmegaConf
import sys

cfg = OmegaConf.load(sys.argv[1])
print(OmegaConf.select(cfg, "framework.name", default=""))
PY
)"
if [[ "$framework_name" != "QwenOFT_SQ3DMix" ]]; then
  echo "Expected framework.name=QwenOFT_SQ3DMix, found: $framework_name" >&2
  exit 2
fi

if [[ -z "$checkpoint_steps" ]]; then
  mapfile -t steps < <(
    find "$model_dir/checkpoints" -maxdepth 1 -type f \
      -name 'steps_*_pytorch_model.pt' -printf '%f\n' \
      | sed -nE 's/^steps_([0-9]+)_pytorch_model\.pt$/\1/p' \
      | sort -n
  )
else
  checkpoint_steps="${checkpoint_steps//,/ }"
  read -r -a steps <<< "$checkpoint_steps"
  mapfile -t steps < <(printf '%s\n' "${steps[@]}" | sed '/^$/d' | sort -n -u)
fi
if (( ${#steps[@]} == 0 )); then
  echo "No checkpoints found in $model_dir/checkpoints" >&2
  exit 2
fi
for step in "${steps[@]}"; do
  if [[ ! "$step" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid checkpoint step: $step" >&2
    exit 2
  fi
  checkpoint="$model_dir/checkpoints/steps_${step}_pytorch_model.pt"
  if [[ ! -s "$checkpoint" ]]; then
    echo "Missing or empty checkpoint: $checkpoint" >&2
    exit 2
  fi
done

datalist="${DATALIST:-$DRIVEDREAMER_SHARED_ROOT/test_meta.json}"
dense_test_cache="${NAVSIM_VGGT_DENSE_TEST_CACHE_ROOT:-${NAVSIM_VGGT_DENSE_CACHE_ROOT%/}_navtest}"

default_pdms_cache="$NAVSIM_EXP_ROOT/metric_cache_navtest_v1_1"
if [[ -n "${PDMS_METRIC_CACHE_PATH:-}" ]]; then
  pdms_cache="$PDMS_METRIC_CACHE_PATH"
elif [[ -d "${NAVSIM_V1_METRIC_CACHE_PATH:-}/metadata" ]]; then
  pdms_cache="$NAVSIM_V1_METRIC_CACHE_PATH"
else
  pdms_cache="$default_pdms_cache"
fi
epdms_cache="${EPDMS_METRIC_CACHE_PATH:-$NAVSIM_EXP_ROOT/metric_cache_navtest}"

out_root="${SQ3DMIX_EVAL_OUT_ROOT:-${OUT_ROOT:-$NAVSIM_EVAL_ROOT/sq3dmix-navtest-all-ckpts/$run_name}}"
prediction_root="$out_root/predictions"
score_root="$out_root/scores"
log_root="$out_root/logs"
summary_csv="$out_root/summary.csv"

device_count="${EVAL_DEVICE_COUNT:-16}"
device_ids="${EVAL_DEVICE_IDS:-}"
batch_size="${BATCH_SIZE:-8}"
num_workers="${NUM_WORKERS:-2}"
cache_workers="${CACHE_WORKERS:-8}"
overwrite="${OVERWRITE:-0}"
rescore="${RESCORE:-0}"
infer_seed="${INFER_SEED:-42}"
eval_attn_implementation="${EVAL_VLM_ATTN_IMPLEMENTATION:-sdpa}"

for pair in \
  "EVAL_DEVICE_COUNT:$device_count" \
  "BATCH_SIZE:$batch_size" \
  "NUM_WORKERS:$num_workers" \
  "CACHE_WORKERS:$cache_workers"; do
  name="${pair%%:*}"
  value="${pair#*:}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer, got: $value" >&2
    exit 2
  fi
done
if [[ -z "$device_ids" ]]; then
  device_ids="$(seq -s, 0 $((device_count - 1)))"
fi
IFS=',' read -r -a devices <<< "$device_ids"
if (( ${#devices[@]} != device_count )); then
  echo "EVAL_DEVICE_IDS contains ${#devices[@]} devices, expected $device_count" >&2
  exit 2
fi

required_paths=(
  "$datalist"
  "$DATA_ROOT/meta/test"
  "$BASE_VLM/config.json"
  "$NUPLAN_MAPS_ROOT"
  "$NAVSIM_TEST_LOG_ROOT"
  "$NAVSIM_TEST_SENSOR_ROOT"
  "$NAVSIM_DEVKIT_ROOT"
  "$NAVSIM_V1_DEVKIT_ROOT"
)
for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Required navtest asset is missing: $path" >&2
    exit 2
  fi
done
expected_count="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$datalist")"

validate_dense_cache() {
  python - "$dense_test_cache/vggt_dense/manifest.json" "$datalist" "$expected_count" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

manifest_path = Path(sys.argv[1])
datalist_path = Path(sys.argv[2])
expected_count = int(sys.argv[3])
if not manifest_path.is_file():
    raise SystemExit(1)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
digest = hashlib.sha256(datalist_path.read_bytes()).hexdigest()
valid = (
    manifest.get("complete") is True
    and manifest.get("component") == "vggt_dense"
    and manifest.get("datalist_sha256") == digest
    and int(manifest.get("sample_count", -1)) == expected_count
    and int(manifest.get("feature_dim", -1)) == 2048
    and manifest.get("view_order") == ["cam_f0", "cam_l0", "cam_r0"]
)
raise SystemExit(0 if valid else 1)
PY
}

validate_metric_cache() {
  local cache_root="$1"
  python - "$cache_root" "$expected_count" <<'PY'
import csv
from pathlib import Path
import sys

root = Path(sys.argv[1])
expected = int(sys.argv[2])
metadata = sorted((root / "metadata").glob("*.csv"))
if len(metadata) != 1:
    raise SystemExit(1)
with metadata[0].open(newline="", encoding="utf-8") as stream:
    rows = [row[0] for row in csv.reader(stream) if row]
if rows and rows[0] == "file_name":
    rows = rows[1:]
if len(rows) != expected:
    raise SystemExit(1)

# Metadata generated in an older checkout may contain a stale absolute prefix.
# Accept it only when either that path or the same suffix under root exists.
marker = f"/{root.name}/"
for raw in (rows[0], rows[len(rows) // 2], rows[-1]):
    original = Path(raw)
    suffix = raw.split(marker, 1)[1] if marker in raw else None
    relocated = root / suffix if suffix is not None else Path("/__missing__")
    if not relocated.is_file() and not original.is_file():
        raise SystemExit(1)
PY
}

generate_dense_cache() {
  echo "[cache] building dense VGGT navtest cache: $dense_test_cache"
  mkdir -p "$dense_test_cache"
  SPLIT=test \
  NAVSIM_DATALIST_PATH="$datalist" \
  NAVSIM_TRAINVAL_SENSOR_ROOT="$NAVSIM_TEST_SENSOR_ROOT" \
  NAVSIM_VGGT_DENSE_CACHE_ROOT="$dense_test_cache" \
  NUM_MACHINES=1 \
  MACHINE_RANK=0 \
  VGGT_DENSE_CACHE_NUM_PROCESSES="${VGGT_DENSE_CACHE_NUM_PROCESSES:-$device_count}" \
  VGGT_DENSE_CACHE_BATCH_SIZE="${VGGT_DENSE_CACHE_BATCH_SIZE:-1}" \
  VGGT_DENSE_CACHE_FULL=1 \
    bash "$project_root/11-precompute_vggt_dense_cache.sh"
}

generate_pdms_cache() {
  echo "[cache] building NAVSIM v1.1 PDMS cache: $pdms_cache"
  mkdir -p "$pdms_cache"
  (
    export NAVSIM_DEVKIT_ROOT="$NAVSIM_V1_DEVKIT_ROOT"
    export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
    python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
      train_test_split=navtest \
      cache.cache_path="$pdms_cache" \
      cache.force_feature_computation=false \
      navsim_log_path="$NAVSIM_TEST_LOG_ROOT" \
      worker=single_machine_thread_pool \
      worker.max_workers="$cache_workers" \
      worker.use_process_pool=true
  )
}

generate_epdms_cache() {
  echo "[cache] building NAVSIM v2 EPDMS cache: $epdms_cache"
  mkdir -p "$epdms_cache"
  (
    export NAVSIM_DEVKIT_ROOT="$project_root/navsim"
    export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
    python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
      train_test_split=navtest \
      metric_cache_path="$epdms_cache" \
      force_feature_computation=false \
      navsim_log_path="$NAVSIM_TEST_LOG_ROOT" \
      original_sensor_path="$NAVSIM_TEST_SENSOR_ROOT" \
      worker=single_machine_thread_pool \
      worker.max_workers="$cache_workers" \
      worker.use_process_pool=true
  )
}

git_ref="$(git -C "$project_root" branch --show-current)"
if [[ -z "$git_ref" ]]; then
  git_ref="detached@$(git -C "$project_root" rev-parse --short HEAD)"
fi
echo "[eval] code checkout:  $project_root"
echo "[eval] git ref:        $git_ref"
echo "[eval] model run:      $model_dir"
echo "[eval] checkpoints:    ${steps[*]}"
echo "[eval] navtest:        $datalist ($expected_count samples)"
echo "[eval] dense cache:    $dense_test_cache"
echo "[eval] PDMS cache:     $pdms_cache"
echo "[eval] EPDMS cache:    $epdms_cache"
echo "[eval] devices:        ${devices[*]}"
echo "[eval] output:         $out_root"

dense_ready=0
pdms_ready=0
epdms_ready=0
validate_dense_cache && dense_ready=1
validate_metric_cache "$pdms_cache" && pdms_ready=1
validate_metric_cache "$epdms_cache" && epdms_ready=1
echo "[eval] cache status:   dense=$dense_ready pdms=$pdms_ready epdms=$epdms_ready"

if [[ "$preflight_only" == "1" ]]; then
  echo "[eval] preflight complete; missing caches would be generated in a full run"
  exit 0
fi

if (( dense_ready == 0 )); then
  if [[ -f "$dense_test_cache/vggt_dense/manifest.json" ]]; then
    echo "Dense cache manifest exists but does not match navtest: $dense_test_cache" >&2
    echo "Point NAVSIM_VGGT_DENSE_TEST_CACHE_ROOT at a new empty directory." >&2
    exit 2
  fi
  generate_dense_cache
  validate_dense_cache || { echo "Dense cache validation failed" >&2; exit 2; }
fi
if (( pdms_ready == 0 )); then
  generate_pdms_cache
  validate_metric_cache "$pdms_cache" || { echo "PDMS cache validation failed" >&2; exit 2; }
fi
if (( epdms_ready == 0 )); then
  generate_epdms_cache
  validate_metric_cache "$epdms_cache" || { echo "EPDMS cache validation failed" >&2; exit 2; }
fi

visible_count="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if (( visible_count < device_count )); then
  echo "Requested $device_count devices, but torch sees $visible_count" >&2
  exit 2
fi

mkdir -p "$prediction_root" "$score_root" "$log_root"

# Build lightweight metadata views with paths relocated to the active DLC
# mount. Metric payloads remain in place and are not copied.
prepare_metric_cache_view() {
  local source_root="$1"
  local view_root="$2"
  mkdir -p "$view_root/metadata"
  python - "$source_root" "$view_root/metadata/cache.csv" "$expected_count" <<'PY'
import csv
import os
from pathlib import Path
import sys

source = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2])
expected = int(sys.argv[3])
metadata = sorted((source / "metadata").glob("*.csv"))
if len(metadata) != 1:
    raise SystemExit(f"Expected one metadata CSV under {source}/metadata")
with metadata[0].open(newline="", encoding="utf-8") as stream:
    rows = [row[0] for row in csv.reader(stream) if row]
if rows and rows[0] == "file_name":
    rows = rows[1:]
if len(rows) != expected:
    raise SystemExit(f"Metric cache has {len(rows)} records, expected {expected}")
marker = f"/{source.name}/"
resolved = []
for raw in rows:
    suffix = raw.split(marker, 1)[1] if marker in raw else None
    relocated = source / suffix if suffix is not None else None
    if relocated is not None and relocated.is_file():
        resolved.append(str(relocated))
    elif Path(raw).is_file():
        resolved.append(str(Path(raw).resolve()))
    else:
        raise SystemExit(f"Metric cache payload is missing: {raw}")
temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
with temporary.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["file_name"])
    writer.writerows((path,) for path in resolved)
os.replace(temporary, output)
PY
}

pdms_cache_view="$out_root/cache_views/pdms"
epdms_cache_view="$out_root/cache_views/epdms"
prepare_metric_cache_view "$pdms_cache" "$pdms_cache_view"
prepare_metric_cache_view "$epdms_cache" "$epdms_cache_view"

latest_score_csv() {
  local root="$1"
  find "$root" -type f -name '*.csv' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n 1 | cut -d' ' -f2-
}

read_score() {
  local csv_path="$1"
  local row_token="$2"
  python - "$csv_path" "$row_token" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
row = next((item for item in rows if item.get("token") == sys.argv[2]), None)
if row is None or not row.get("score"):
    raise SystemExit(f"Missing {sys.argv[2]!r} score in {sys.argv[1]}")
print(row["score"])
PY
}

write_summary() {
  local temporary
  temporary="$(mktemp "$out_root/summary.csv.tmp.XXXXXX")"
  printf 'step,pdms,epdms,pdms_csv,epdms_csv\n' > "$temporary"
  local step pdms_csv epdms_csv pdms_value epdms_value
  local -a summary_steps=()
  mapfile -t summary_steps < <(
    find "$score_root" -mindepth 1 -maxdepth 1 -type d \
      -name 'step*' -printf '%f\n' 2>/dev/null \
      | sed -nE 's/^step([0-9]+)$/\1/p' \
      | sort -n -u
  )
  for step in "${summary_steps[@]}"; do
    pdms_csv="$score_root/step${step}/pdms.csv"
    epdms_csv="$score_root/step${step}/epdms.csv"
    if [[ -f "$pdms_csv" && -f "$epdms_csv" ]]; then
      pdms_value="$(read_score "$pdms_csv" average)"
      epdms_value="$(read_score "$epdms_csv" average_all_frames)"
      printf '%s,%s,%s,%s,%s\n' \
        "$step" "$pdms_value" "$epdms_value" "$pdms_csv" "$epdms_csv" \
        >> "$temporary"
    fi
  done
  mv "$temporary" "$summary_csv"
}

for step in "${steps[@]}"; do
  prediction_run="$prediction_root/${run_name}-step${step}"
  prediction_dir="$prediction_run/test"
  step_log_root="$log_root/step${step}"
  step_score_root="$score_root/step${step}"
  mkdir -p "$step_log_root" "$step_score_root"

  prediction_count=0
  if [[ -d "$prediction_dir" ]]; then
    prediction_count="$(find "$prediction_dir" -maxdepth 1 -type f -name '*.npy' | wc -l)"
  fi
  if (( prediction_count == expected_count )) && [[ "$overwrite" != "1" ]]; then
    echo "[step $step] reusing $prediction_count predictions"
  else
    echo "[step $step] inference: found=$prediction_count expected=$expected_count"
    pids=()
    for ((rank = 0; rank < device_count; rank++)); do
      device="${devices[$rank]}"
      (
        MODEL_DIR="$model_dir" \
        MODEL_ITER="$step" \
        SPLIT=test \
        DATALIST="$datalist" \
        OUT_DIR="$prediction_root" \
        BATCH_SIZE="$batch_size" \
        NUM_WORKERS="$num_workers" \
        GPU="$device" \
        RANK="$rank" \
        WORLD_SIZE="$device_count" \
        OVERWRITE="$overwrite" \
        INFER_USE_FEATURE_CACHE=0 \
        INFER_SEED="$infer_seed" \
        NAVSIM_VGGT_DENSE_CACHE_ROOT="$dense_test_cache" \
        VLM_ATTN_IMPLEMENTATION="$eval_attn_implementation" \
          bash "$project_root/4-infer.sh"
      ) > "$step_log_root/infer.rank${rank}.log" 2>&1 &
      pids+=("$!")
    done

    failed=0
    for ((rank = 0; rank < device_count; rank++)); do
      if ! wait "${pids[$rank]}"; then
        echo "[step $step] rank $rank failed: $step_log_root/infer.rank${rank}.log" >&2
        failed=1
      fi
    done
    (( failed == 0 )) || exit 1
    prediction_count="$(find "$prediction_dir" -maxdepth 1 -type f -name '*.npy' | wc -l)"
    if (( prediction_count != expected_count )); then
      echo "[step $step] prediction count mismatch: $prediction_count != $expected_count" >&2
      exit 1
    fi
  fi

  pdms_csv="$step_score_root/pdms.csv"
  if [[ -f "$pdms_csv" && "$rescore" != "1" ]]; then
    echo "[step $step] reusing PDMS=$(read_score "$pdms_csv" average)"
  else
    pdms_work="$step_score_root/pdms_work"
    mkdir -p "$pdms_work"
    echo "[step $step] scoring PDMS (NAVSIM v1.1)"
    (
      PRED_DIR="$prediction_run" \
      SPLIT=test \
      DATALIST="$datalist" \
      METRIC_CACHE_PATH="$pdms_cache_view" \
      NAVSIM_EVAL_ROOT="$pdms_work" \
        bash "$project_root/5-eval_v1.sh"
    ) 2>&1 | tee "$step_log_root/pdms.log"
    generated_csv="$(latest_score_csv "$pdms_work")"
    [[ -n "$generated_csv" ]] || { echo "PDMS result CSV was not produced" >&2; exit 1; }
    cp "$generated_csv" "$pdms_csv"
  fi

  epdms_csv="$step_score_root/epdms.csv"
  if [[ -f "$epdms_csv" && "$rescore" != "1" ]]; then
    echo "[step $step] reusing EPDMS=$(read_score "$epdms_csv" average_all_frames)"
  else
    epdms_work="$step_score_root/epdms_work"
    mkdir -p "$epdms_work"
    echo "[step $step] scoring EPDMS (NAVSIM v2)"
    (
      PRED_DIR="$prediction_run" \
      SPLIT=test \
      DATALIST="$datalist" \
      METRIC_CACHE_PATH="$epdms_cache_view" \
      NAVSIM_EVAL_ROOT="$epdms_work" \
      CACHE_WORKERS="$cache_workers" \
        bash "$project_root/6-eval_v2.sh"
    ) 2>&1 | tee "$step_log_root/epdms.log"
    generated_csv="$(latest_score_csv "$epdms_work")"
    [[ -n "$generated_csv" ]] || { echo "EPDMS result CSV was not produced" >&2; exit 1; }
    cp "$generated_csv" "$epdms_csv"
  fi

  echo "[step $step] PDMS=$(read_score "$pdms_csv" average) EPDMS=$(read_score "$epdms_csv" average_all_frames)"
  write_summary
done

echo "[eval] complete: $summary_csv"
cat "$summary_csv"
