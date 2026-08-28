#!/usr/bin/env bash
# Run DriveVLA Base inference once, then score the exact 12,146 trajectories
# with official NAVSIM v1.1 PDMS and official NAVSIM v2 EPDMS.

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
source "$project_root/load_env.sh"

helper="$script_dir/base_navtest_dual_metrics.py"
runtime_check="$script_dir/check_ppu_runtime.py"
runtime_versions="$script_dir/verify_runtime_versions.py"
# The Python helper commits protocol.json and summary.csv/json with os.replace.

usage() {
  cat <<'EOF'
Usage: bash scripts/run_base_navtest_dual_metrics_dlc.sh [options]

Identity/path options (CLI > one-shot environment > env.local.sh):
  --run-id ID                 Unique output identity
  --model-dir PATH            Directory containing the fixed Base checkpoint
  --checkpoint-steps STEP     Fixed checkpoint identity (must be 174312)
  --checkpoint PATH           Fixed Base checkpoint file
  --vlm-config PATH           InternVL config/tokenizer directory
  --dino-weights PATH         DINOv2 weights
  --datalist PATH             Exact 12,146-token navtest JSON list
  --data-root PATH            NAVSIM root with meta_datas/test and sensor_blobs/test
  --maps-root PATH            nuPlan map root
  --pdms-cache PATH           Official NAVSIM v1.1 navtest metric cache
  --epdms-cache PATH          Official NAVSIM v2 navtest metric cache
  --navsim-v2-root PATH       External official NAVSIM v2 devkit checkout
  --output-root PATH          Exact run output directory

Runtime options:
  --num-workers N             v1 inference DataLoader workers (default: 4)
  --preflight-only            Validate assets/runtime; no writes/model load/eval
  --dry-run                   Print the resolved protocol; no imports or writes
  --resume                    Reuse only identity-checked complete artifacts
  --overwrite                 Archive this exact run and restart from scratch
  -h, --help                  Show this help

The launcher is fixed to RANK=0, WORLD_SIZE=1, one visible PPU, batch size 1,
and the released Base/no-memory checkpoint. AUTO_GENERATE_CACHES is disabled.
It never installs or changes torch, triton, or flash-attn. The installed
flash-attn version is checked and remains unused (use_flash_attn=false).
EOF
}

require_value() {
  if (( $# < 2 )) || [[ -z "$2" ]]; then
    echo "Missing value for $1" >&2
    usage >&2
    exit 2
  fi
}

run_id="${RUN_ID:-drivevla-base-navtest-dual-$(date -u +%Y%m%dT%H%M%SZ)}"
checkpoint_steps="${CHECKPOINT_STEPS:-174312}"
checkpoint="${DRIVEVLA_BASE_CHECKPOINT:-}"
model_dir="${MODEL_DIR:-${checkpoint:+$(dirname -- "$checkpoint")}}"
vlm_config="${DRIVEVLA_VLM_CONFIG:-}"
dino_weights="${DRIVEVLA_DINO_WEIGHTS:-}"
datalist="${DRIVEVLA_NAVTEST_TOKEN_LIST:-}"
data_root="${OPENSCENE_DATA_ROOT:-}"
maps_root="${NUPLAN_MAPS_ROOT:-}"
pdms_cache="${PDMS_METRIC_CACHE_PATH:-${METRIC_CACHE_PATH:-}}"
epdms_cache="${EPDMS_METRIC_CACHE_PATH:-}"
navsim_v2_root="${DRIVEVLA_NAVSIM_V2_ROOT:-}"
output_root=""
num_workers="${NUM_WORKERS:-4}"
dry_run="${DRY_RUN:-0}"
preflight_only="${PREFLIGHT_ONLY:-0}"
resume="${RESUME:-0}"
overwrite="${OVERWRITE:-0}"
checkpoint_min_age="${CHECKPOINT_MIN_AGE_SECONDS:-120}"
AUTO_GENERATE_CACHES=0
export AUTO_GENERATE_CACHES

while (( $# > 0 )); do
  case "$1" in
    --run-id) require_value "$@"; run_id="$2"; shift 2 ;;
    --run-id=*) run_id="${1#*=}"; shift ;;
    --model-dir) require_value "$@"; model_dir="$2"; shift 2 ;;
    --model-dir=*) model_dir="${1#*=}"; shift ;;
    --checkpoint-steps) require_value "$@"; checkpoint_steps="$2"; shift 2 ;;
    --checkpoint-steps=*) checkpoint_steps="${1#*=}"; shift ;;
    --checkpoint) require_value "$@"; checkpoint="$2"; shift 2 ;;
    --checkpoint=*) checkpoint="${1#*=}"; shift ;;
    --vlm-config) require_value "$@"; vlm_config="$2"; shift 2 ;;
    --vlm-config=*) vlm_config="${1#*=}"; shift ;;
    --dino-weights) require_value "$@"; dino_weights="$2"; shift 2 ;;
    --dino-weights=*) dino_weights="${1#*=}"; shift ;;
    --datalist) require_value "$@"; datalist="$2"; shift 2 ;;
    --datalist=*) datalist="${1#*=}"; shift ;;
    --data-root) require_value "$@"; data_root="$2"; shift 2 ;;
    --data-root=*) data_root="${1#*=}"; shift ;;
    --maps-root) require_value "$@"; maps_root="$2"; shift 2 ;;
    --maps-root=*) maps_root="${1#*=}"; shift ;;
    --pdms-cache) require_value "$@"; pdms_cache="$2"; shift 2 ;;
    --pdms-cache=*) pdms_cache="${1#*=}"; shift ;;
    --epdms-cache) require_value "$@"; epdms_cache="$2"; shift 2 ;;
    --epdms-cache=*) epdms_cache="${1#*=}"; shift ;;
    --navsim-v2-root) require_value "$@"; navsim_v2_root="$2"; shift 2 ;;
    --navsim-v2-root=*) navsim_v2_root="${1#*=}"; shift ;;
    --output-root)
      require_value "$@"
      output_root="$2"
      shift 2
      ;;
    --output-root=*) output_root="${1#*=}"; shift ;;
    --num-workers) require_value "$@"; num_workers="$2"; shift 2 ;;
    --num-workers=*) num_workers="${1#*=}"; shift ;;
    --dry-run) dry_run=1; shift ;;
    --preflight-only) preflight_only=1; shift ;;
    --resume) resume=1; shift ;;
    --overwrite) overwrite=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$checkpoint" && -n "$model_dir" ]]; then
  checkpoint="$model_dir/best-epoch_26-step_${checkpoint_steps}.server_merged.ckpt"
fi
if [[ -z "$output_root" ]]; then
  output_base="${DRIVEVLA_DLC_EVAL_ROOT:-$NAVSIM_EXP_ROOT/dlc_navtest_dual_metrics}"
  output_root="$output_base/$run_id"
fi

for pair in \
  "DRY_RUN:$dry_run" \
  "PREFLIGHT_ONLY:$preflight_only" \
  "RESUME:$resume" \
  "OVERWRITE:$overwrite"; do
  name="${pair%%:*}"
  value="${pair#*:}"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "$name must be 0 or 1, got: $value" >&2
    exit 2
  fi
done
if [[ "$resume" == "1" && "$overwrite" == "1" ]]; then
  echo "--resume and --overwrite are mutually exclusive" >&2
  exit 2
fi
if [[ "$checkpoint_steps" != "174312" ]]; then
  echo "--checkpoint-steps is fixed to 174312 for the released Base checkpoint" >&2
  exit 2
fi
if ! [[ "$num_workers" =~ ^[0-9]+$ ]]; then
  echo "--num-workers must be a non-negative integer: $num_workers" >&2
  exit 2
fi
if ! [[ "$checkpoint_min_age" =~ ^[0-9]+$ ]]; then
  echo "CHECKPOINT_MIN_AGE_SECONDS must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]]; then
  echo "Invalid --run-id (1-96 safe characters required): $run_id" >&2
  exit 2
fi

RANK="${RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-1}"
if [[ "$RANK" != "0" || "$WORLD_SIZE" != "1" ]]; then
  echo "This evaluator requires RANK=0 and WORLD_SIZE=1; got $RANK/$WORLD_SIZE" >&2
  exit 2
fi
export RANK WORLD_SIZE

for pair in \
  "checkpoint:$checkpoint" \
  "model_dir:$model_dir" \
  "vlm_config:$vlm_config" \
  "dino_weights:$dino_weights" \
  "datalist:$datalist" \
  "data_root:$data_root" \
  "maps_root:$maps_root" \
  "pdms_cache:$pdms_cache" \
  "epdms_cache:$epdms_cache" \
  "navsim_v2_root:$navsim_v2_root"; do
  name="${pair%%:*}"
  value="${pair#*:}"
  if [[ -z "$value" ]]; then
    echo "Required path is unset: $name" >&2
    exit 2
  fi
done

checkpoint="$(realpath -m -- "$checkpoint")"
model_dir="$(realpath -m -- "$model_dir")"
vlm_config="$(realpath -m -- "$vlm_config")"
dino_weights="$(realpath -m -- "$dino_weights")"
datalist="$(realpath -m -- "$datalist")"
data_root="$(realpath -m -- "$data_root")"
maps_root="$(realpath -m -- "$maps_root")"
pdms_cache="$(realpath -m -- "$pdms_cache")"
epdms_cache="$(realpath -m -- "$epdms_cache")"
navsim_v2_root="$(realpath -m -- "$navsim_v2_root")"
output_root="$(realpath -m -- "$output_root")"

preflight_command=(
  "$PYTHON_BIN" "$helper" preflight
  --repo-root "$project_root"
  --checkpoint "$checkpoint"
  --checkpoint-step "$checkpoint_steps"
  --checkpoint-min-age "$checkpoint_min_age"
  --vlm-config "$vlm_config"
  --dino-weights "$dino_weights"
  --datalist "$datalist"
  --data-root "$data_root"
  --maps-root "$maps_root"
  --pdms-cache "$pdms_cache"
  --epdms-cache "$epdms_cache"
  --navsim-v2-root "$navsim_v2_root"
)

if [[ "$dry_run" == "1" ]]; then
  printf '%s\n' "DRY RUN: full navtest, 12,146 scenes, RANK=0, WORLD_SIZE=1"
  printf '%s\n' "Model: DriveVLA-M0 Base/no-memory checkpoint step $checkpoint_steps"
  printf '%s\n' "PDMS: official NAVSIM v1.1 / aggregate row average"
  printf '%s\n' "EPDMS: official NAVSIM v2 / aggregate row average_all_frames"
  printf '%s\n' "Inference: one pass; v2 reuses the exact [8,3] float32 trajectories"
  printf '%s\n' "Runtime: existing PPU packages only; flash-attn is checked, never changed, and disabled in Base"
  printf '%s\n' "Caches: AUTO_GENERATE_CACHES=0; separate read-only metadata views"
  printf 'Preflight command:'
  printf ' %q' "${preflight_command[@]}"
  printf '\nOutput root: %s\n' "$output_root"
  printf '%s\n' "No files were written and no Python module was imported."
  exit 0
fi

if [[ "$preflight_only" == "1" ]]; then
  "$PYTHON_BIN" "$runtime_versions"
  EXPECTED_VISIBLE_DEVICES=1 "$PYTHON_BIN" "$runtime_check"
  "${preflight_command[@]}"
  printf '%s\n' "PREFLIGHT PASS: no output was written and no model/evaluator was loaded."
  exit 0
fi

summary_csv="$output_root/summary.csv"
summary_json="$output_root/summary.json"
results_root="$output_root/results"
stable_pdms_csv="$results_root/pdms.csv"
stable_epdms_csv="$results_root/epdms.csv"
prediction_root="$output_root/predictions"
work_root="$output_root/work"
inference_manifest_pattern="inference_manifest.rank*.json"

is_complete=0
if [[ -f "$summary_csv" && -f "$summary_json" && -f "$stable_pdms_csv" && -f "$stable_epdms_csv" ]]; then
  if "$PYTHON_BIN" "$helper" validate-score \
      --csv "$stable_pdms_csv" --aggregate-token average --datalist "$datalist" \
      >/dev/null 2>&1 && \
    "$PYTHON_BIN" "$helper" validate-score \
      --csv "$stable_epdms_csv" --aggregate-token average_all_frames --datalist "$datalist" \
      >/dev/null 2>&1; then
    is_complete=1
  fi
fi

if [[ -e "$output_root" && "$overwrite" == "0" && "$resume" == "0" ]]; then
  if [[ "$is_complete" == "1" ]]; then
    echo "Complete metric files exist; validating the immutable protocol before reuse: $output_root"
    resume=1
  else
    echo "Refusing to overwrite an existing incomplete run: $output_root" >&2
    echo "Use --resume for verified artifacts or --overwrite only to archive and restart this exact run." >&2
    exit 2
  fi
fi

if [[ -e "$output_root" && "$overwrite" == "1" ]]; then
  output_parent="$(dirname -- "$output_root")"
  archive_root="$output_parent/.superseded"
  archive_target="$archive_root/${run_id}-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p -- "$archive_root"
  if [[ -e "$archive_target" ]]; then
    echo "Refusing archive collision: $archive_target" >&2
    exit 2
  fi
  mv -- "$output_root" "$archive_target"
  echo "Archived the previous exact run to: $archive_target"
fi

mkdir -p -- "$output_root" "$results_root" "$work_root" "$output_root/logs"
launcher_log="$output_root/logs/launcher.log"
exec > >(tee -a "$launcher_log") 2>&1

phase="formal preflight"
on_error() {
  status=$?
  echo "FAILED phase=$phase status=$status output=$output_root" >&2
  exit "$status"
}
trap on_error ERR

echo "[dual-navtest] run_id=$run_id output=$output_root"
echo "[dual-navtest] fixed topology RANK=$RANK WORLD_SIZE=$WORLD_SIZE visible_ppus=1"
echo "[dual-navtest] flash-attn will not be installed, changed, or enabled"
"$PYTHON_BIN" "$runtime_versions"
EXPECTED_VISIBLE_DEVICES=1 "$PYTHON_BIN" "$runtime_check"
"${preflight_command[@]}" --json-output "$output_root/protocol.json"

phase="read-only metric-cache views"
pdms_cache_view="$output_root/cache_views/pdms"
epdms_cache_view="$output_root/cache_views/epdms"
"$PYTHON_BIN" "$helper" prepare-cache-view \
  --source "$pdms_cache" --target "$pdms_cache_view" --datalist "$datalist"
"$PYTHON_BIN" "$helper" prepare-cache-view \
  --source "$epdms_cache" --target "$epdms_cache_view" --datalist "$datalist"

checkpoint_sha="7f7ce61b9492936d9af15c7bb68dc57d53c6629fccd23ac5ae530058286a807d"
prediction_complete=0
if [[ -d "$prediction_root" ]]; then
  if "$PYTHON_BIN" "$helper" validate-predictions \
      --prediction-root "$prediction_root" \
      --datalist "$datalist" \
      --checkpoint-sha256 "$checkpoint_sha" >/dev/null; then
    prediction_complete=1
    echo "[dual-navtest] reusing 12,146 verified predictions ($inference_manifest_pattern)"
  else
    echo "Refusing unverifiable prediction artifact: $prediction_root" >&2
    echo "Use --overwrite only to archive and restart this exact run." >&2
    exit 2
  fi
fi

prediction_pickle=""
if [[ "$prediction_complete" == "0" ]]; then
  prediction_pickle="$({ "$PYTHON_BIN" "$helper" find-pickle \
      --root "$work_root" --datalist "$datalist"; } 2>/dev/null || true)"
  if [[ -z "$prediction_pickle" ]]; then
    phase="full navtest Base inference and official v1.1 PDMS"
    attempt_id="$(date -u +%Y%m%dT%H%M%SZ)"
    v1_attempt="$work_root/v1-$attempt_id"
    mkdir -p -- "$v1_attempt"
    echo "[dual-navtest] starting one-pass Base inference for all 12,146 navtest scenes"
    (
      unset LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK NODE_RANK MASTER_ADDR MASTER_PORT
      unset RANK WORLD_SIZE MAX_SCENES
      export NAVSIM_EXP_ROOT="$v1_attempt/navsim_exp"
      export SUBSCORE_PATH="$v1_attempt"
      export METRIC_CACHE_PATH="$pdms_cache_view"
      export DRIVEVLA_BASE_CHECKPOINT="$checkpoint"
      export DRIVEVLA_VLM_CONFIG="$vlm_config"
      export DRIVEVLA_DINO_WEIGHTS="$dino_weights"
      export OPENSCENE_DATA_ROOT="$data_root"
      export NUPLAN_MAPS_ROOT="$maps_root"
      export NUM_GPUS=1
      export BATCH_SIZE=1
      export NUM_WORKERS="$num_workers"
      export RUN_PREFLIGHT=0
      export EXPERIMENT_NAME="drivevla_base_navtest_${run_id}"
      bash "$script_dir/run_base_pdms.sh" "experiment_uid=$attempt_id"
    ) | tee "$output_root/logs/v1-inference-pdms-$attempt_id.log"
    prediction_pickle="$({ "$PYTHON_BIN" "$helper" find-pickle \
        --root "$v1_attempt" --datalist "$datalist"; } 2>/dev/null || true)"
    if [[ -z "$prediction_pickle" ]]; then
      echo "Base inference did not produce an exact 12,146-scene pickle" >&2
      exit 2
    fi
  else
    echo "[dual-navtest] resuming from verified prediction pickle: $prediction_pickle"
  fi

  phase="atomic prediction conversion"
  "$PYTHON_BIN" "$helper" convert \
    --pickle "$prediction_pickle" \
    --prediction-root "$prediction_root" \
    --datalist "$datalist" \
    --checkpoint-sha256 "$checkpoint_sha" \
    --checkpoint-step "$checkpoint_steps"
fi

phase="official v1.1 PDMS result validation"
pdms_valid=0
if [[ -f "$stable_pdms_csv" ]] && \
  "$PYTHON_BIN" "$helper" validate-score \
    --csv "$stable_pdms_csv" --aggregate-token average --datalist "$datalist" \
    >/dev/null; then
  pdms_valid=1
  echo "[dual-navtest] reusing verified PDMS: $stable_pdms_csv"
fi
if [[ "$pdms_valid" == "0" ]]; then
  produced_pdms="$({ "$PYTHON_BIN" "$helper" find-score \
      --root "$work_root" --aggregate-token average --datalist "$datalist"; } \
      2>/dev/null || true)"
  if [[ -z "$produced_pdms" ]]; then
    phase="official v1.1 PDMS rescore from exact submission"
    pdms_rescore="$work_root/pdms-rescore-$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p -- "$pdms_rescore"
    "$PYTHON_BIN" "$project_root/navsim/planning/script/run_pdm_score_from_submission.py" \
      "submission_file_path=$prediction_root/submission.pkl" \
      "metric_cache_path=$pdms_cache_view" \
      "output_dir=$pdms_rescore"
    produced_pdms="$({ "$PYTHON_BIN" "$helper" find-score \
        --root "$pdms_rescore" --aggregate-token average --datalist "$datalist"; } \
        2>/dev/null || true)"
  fi
  if [[ -z "$produced_pdms" ]]; then
    echo "No complete official NAVSIM v1.1 PDMS CSV was produced" >&2
    exit 2
  fi
  "$PYTHON_BIN" "$helper" atomic-copy \
    --source "$produced_pdms" --destination "$stable_pdms_csv"
fi
"$PYTHON_BIN" "$helper" validate-score \
  --csv "$stable_pdms_csv" --aggregate-token average --datalist "$datalist"

phase="official NAVSIM v2 EPDMS"
epdms_valid=0
if [[ -f "$stable_epdms_csv" ]] && \
  "$PYTHON_BIN" "$helper" validate-score \
    --csv "$stable_epdms_csv" --aggregate-token average_all_frames --datalist "$datalist" \
    >/dev/null; then
  epdms_valid=1
  echo "[dual-navtest] reusing verified EPDMS: $stable_epdms_csv"
fi
if [[ "$epdms_valid" == "0" ]]; then
  produced_epdms="$({ "$PYTHON_BIN" "$helper" find-score \
      --root "$work_root" --aggregate-token average_all_frames --datalist "$datalist"; } \
      2>/dev/null || true)"
  if [[ -z "$produced_epdms" ]]; then
    v2_attempt="$work_root/v2-$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p -- "$v2_attempt"
    echo "[dual-navtest] scoring the same trajectories with official NAVSIM v2 EPDMS"
    (
      export NAVSIM_EXP_ROOT="$v2_attempt"
      export NAVSIM_DEVKIT_ROOT="$navsim_v2_root"
      export PYTHONPATH="$navsim_v2_root${PYTHONPATH:+:$PYTHONPATH}"
      export NUPLAN_MAPS_ROOT="$maps_root"
      export OPENSCENE_DATA_ROOT="$data_root"
      "$PYTHON_BIN" "$navsim_v2_root/navsim/planning/script/run_pdm_score_one_stage.py" \
        train_test_split=navtest \
        agent=human_agent \
        experiment_name="drivevla_base_navtest_${run_id}" \
        experiment_uid=official \
        metric_cache_path="$epdms_cache_view" \
        pred_dir="$prediction_root" \
        split=test \
        navsim_log_path="$data_root/meta_datas/test" \
        original_sensor_path="$data_root/sensor_blobs/test"
    ) | tee "$output_root/logs/epdms.log"
    produced_epdms="$({ "$PYTHON_BIN" "$helper" find-score \
        --root "$v2_attempt" --aggregate-token average_all_frames --datalist "$datalist"; } \
        2>/dev/null || true)"
  else
    echo "[dual-navtest] adopting a verified EPDMS result from prior work"
  fi
  if [[ -z "$produced_epdms" ]]; then
    echo "No complete official NAVSIM v2 EPDMS CSV was produced" >&2
    exit 2
  fi
  "$PYTHON_BIN" "$helper" atomic-copy \
    --source "$produced_epdms" --destination "$stable_epdms_csv"
fi
"$PYTHON_BIN" "$helper" validate-score \
  --csv "$stable_epdms_csv" --aggregate-token average_all_frames --datalist "$datalist"

phase="atomic summary"
"$PYTHON_BIN" "$helper" summarize \
  --run-id "$run_id" \
  --datalist "$datalist" \
  --pdms-csv "$stable_pdms_csv" \
  --epdms-csv "$stable_epdms_csv" \
  --output-root "$output_root"

trap - ERR
echo "[dual-navtest] PASS summary.csv=$summary_csv"
"$PYTHON_BIN" - "$summary_json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"PDMS={data['PDMS']['score']:.6f}")
print(f"EPDMS={data['EPDMS']['score']:.6f}")
print(f"scenarios={data['scenarios']}")
PY
