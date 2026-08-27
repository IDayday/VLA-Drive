#!/usr/bin/env bash
# Diagnose Register64 geometry vs selector quality with official navtest PDMS.
#
# GPU: export all 64 proposals and the DrivoR-selected index once.
# CPU: score the full proposal pool with exact NAVSIM-v1.1 per-candidate
#      semantics, select Oracle@64, and verify it with the official scorer.
# This intentionally does not claim an EPDMS Oracle: v2 extended comfort links
# adjacent frames, so a true upper bound requires sequence-level optimization
# rather than an independent per-scene argmax.

set -Eeuo pipefail

dry_run=0
preflight_only=0
resume=0
overwrite=0
cli_source_run=""
cli_output_root=""
cli_run_id=""
cli_datalist=""
cli_data_root=""
cli_metric_cache=""
cli_workers=""
checkpoint_steps="best"
while (( $# )); do
  case "$1" in
    --dry-run) dry_run=1 ;;
    --preflight-only) preflight_only=1 ;;
    --resume) resume=1 ;;
    --overwrite) overwrite=1 ;;
    --source-run|--model-dir|--output-root|--run-id|--datalist|--data-root|--metric-cache|--workers|--checkpoint-steps)
      if (( $# < 2 )) || [[ -z "$2" ]]; then
        echo "[oracle64] missing value for $1" >&2
        exit 2
      fi
      case "$1" in
        --source-run|--model-dir) cli_source_run="$2" ;;
        --output-root) cli_output_root="$2" ;;
        --run-id) cli_run_id="$2" ;;
        --datalist) cli_datalist="$2" ;;
        --data-root) cli_data_root="$2" ;;
        --metric-cache) cli_metric_cache="$2" ;;
        --workers) cli_workers="$2" ;;
        --checkpoint-steps) checkpoint_steps="$2" ;;
      esac
      shift
      ;;
    --help|-h)
      cat <<'EOF'
Usage: run_register64_oracle64_pdms_dlc.sh [OPTIONS]

Options:
  --dry-run                  Print the resolved contract; write nothing.
  --preflight-only           Validate DLC runtime and all inputs; write nothing.
  --resume                   Resume only identity-compatible partial outputs.
  --overwrite                Move the exact old run to a recoverable backup.
  --source-run PATH          Completed Register64 OFF pipeline run.
  --model-dir PATH           Alias for --source-run.
  --checkpoint-steps best    Fixed stable component-checkpoint snapshot.
  --output-root PATH         Dedicated Oracle evaluation root.
  --run-id ID                Output namespace below OUTPUT_ROOT.
  --datalist PATH            Full navtest token JSON.
  --data-root PATH           Processed NAVSIM root containing meta/test.
  --metric-cache PATH        Complete NAVSIM-v1.1 navtest metric cache.
  --workers N                CPU workers for pool and official scoring.
EOF
      exit 0
      ;;
    *) echo "[oracle64] unsupported argument: $1" >&2; exit 2 ;;
  esac
  shift
done

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"
export DRIVEDREAMER_ROOT="$project_root"
cd "$project_root"

source_run_default="${REGISTER64_ORACLE_SOURCE_RUN_ROOT:-${NAVSIM_EXP_ROOT:-$project_root/navsim_exp}/register64_complete_pipeline/register64-r64d-off-formal-v2}"
output_root_default="${REGISTER64_ORACLE_OUTPUT_ROOT:-${NAVSIM_EXP_ROOT:-$project_root/navsim_exp}/register64_oracle_evaluation}"
run_id_default="${REGISTER64_ORACLE_RUN_ID:-register64-r64d-off-formal-v2-oracle64-pdms}"
source_datalist_default="${NAVSIM_DATALIST_PATH:-$project_root/train_meta.json}"
datalist_default="${REGISTER64_NAVTEST_DATALIST:-$(dirname -- "$source_datalist_default")/test_meta.json}"
data_root_default="${DATA_ROOT:-$project_root/navsim_dataset}"

source_run="${cli_source_run:-$source_run_default}"
output_root="${cli_output_root:-$output_root_default}"
run_id="${cli_run_id:-$run_id_default}"
navtest_datalist="${cli_datalist:-$datalist_default}"
data_root="${cli_data_root:-$data_root_default}"
pdms_cache_default="${REGISTER64_ORACLE_PDMS_CACHE_ROOT:-$source_run/caches/navtest_v1_1}"
pdms_metric_cache="${cli_metric_cache:-$pdms_cache_default}"
oracle_workers="${cli_workers:-${REGISTER64_ORACLE_WORKERS:-96}}"

export REGISTER64_ORACLE_EXPECTED_BRANCH="${REGISTER64_ORACLE_EXPECTED_BRANCH:-feature/ddp-drs-scene-2048}"
export REGISTER64_ORACLE_ALLOW_DIRTY="${REGISTER64_ORACLE_ALLOW_DIRTY:-0}"
export LOCAL_NUM_PROCESSES="${LOCAL_NUM_PROCESSES:-16}"
export NUM_MACHINES="${NUM_MACHINES:-1}"
export MACHINE_RANK="${MACHINE_RANK:-0}"
export NUM_PROCESSES="${NUM_PROCESSES:-$((LOCAL_NUM_PROCESSES * NUM_MACHINES))}"
export REGISTER64_ORACLE_MAIN_PROCESS_PORT="${REGISTER64_ORACLE_MAIN_PROCESS_PORT:-29861}"
export REGISTER64_ORACLE_INFER_BATCH_SIZE="${REGISTER64_ORACLE_INFER_BATCH_SIZE:-4}"
export REGISTER64_ORACLE_INFER_WORKERS="${REGISTER64_ORACLE_INFER_WORKERS:-3}"
export CHECKPOINT_MIN_AGE_SECONDS="${CHECKPOINT_MIN_AGE_SECONDS:-60}"
export AUTO_GENERATE_CACHES=0
export QWEN_VLM_PATH="${QWEN_VLM_PATH:-${BASE_VLM:-}}"
# The PPU image's validated CUDA-compatible runtime uses SDPA.
export VLM_ATTN_IMPLEMENTATION=sdpa
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$project_root/navsim_dataset_raw}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$OPENSCENE_DATA_ROOT/maps}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export REGISTER64_NAVTEST_LOG_ROOT="${REGISTER64_NAVTEST_LOG_ROOT:-$OPENSCENE_DATA_ROOT/navsim_logs/test}"

source_run="$(realpath -m "$source_run")"
output_root="$(realpath -m "$output_root")"
navtest_datalist="$(realpath -m "$navtest_datalist")"
data_root="$(realpath -m "$data_root")"
pdms_metric_cache="$(realpath -m "$pdms_metric_cache")"
run_root="$output_root/$run_id"

generator_config="$project_root/starVLA/config/training/register64_inference.yaml"
generator_checkpoint="$source_run/stages/qwen_register64_generator/best_minade_generator.pt"
generator_complete="$source_run/stages/qwen_register64_generator/training_complete.json"
drivor_checkpoint="$source_run/stages/register64_drivor_scorer/best_regret.pt"
drivor_complete="$source_run/stages/register64_drivor_scorer/training_complete.json"
source_summary="$source_run/summary/summary.csv"
prediction_root="$run_root/predictions"
prediction_dir="$prediction_root/test"
candidate_dir="$prediction_dir/candidates"
oracle_selection_root="$run_root/oracle_selection"
oracle_prediction_root="$oracle_selection_root/oracle_predictions"
official_results="$run_root/evaluation/pdms_v1_1_oracle64"
summary_root="$run_root/summary"
launcher_log="$output_root/launcher_logs/$run_id.log"

print_command() {
  printf '[oracle64] command:'
  printf ' %q' "$@"
  printf '\n'
}

candidate_command=(
  accelerate launch --multi_gpu --mixed_precision bf16
  --num_machines 1 --num_processes "$NUM_PROCESSES"
  --machine_rank 0 --main_process_port "$REGISTER64_ORACLE_MAIN_PROCESS_PORT"
  --num_cpu_threads_per_process 1
  "$project_root/starVLA/training/export_register_navtest_predictions.py"
  --config "$generator_config"
  --datalist "$navtest_datalist"
  --data-root "$data_root"
  --output-dir "$prediction_dir"
  --split test
  --batch-size "$REGISTER64_ORACLE_INFER_BATCH_SIZE"
  --num-workers "$REGISTER64_ORACLE_INFER_WORKERS"
  --generator-checkpoint "$generator_checkpoint"
  --drivor-checkpoint "$drivor_checkpoint"
  --export-candidates
)
oracle_command=(
  python "$project_root/tools/score_register64_oracle_pdms.py"
  --candidate-dir "$candidate_dir"
  --prediction-manifest "$prediction_dir/prediction_manifest.json"
  --metric-cache-root "$pdms_metric_cache"
  --datalist "$navtest_datalist"
  --output-dir "$oracle_selection_root"
  --proposal-num 64
  --workers "$oracle_workers"
)
official_command=(
  python "$project_root/navsim_v1.1/navsim/navsim/planning/script/run_pdm_score.py"
  train_test_split=navtest
  "metric_cache_path=$pdms_metric_cache"
  agent=human_agent
  experiment_name=register64-oracle64-navtest-pdms
  "output_dir=$official_results"
  "pred_dir=$oracle_prediction_root"
  split=test
  worker=ray_distributed_no_torch
  "worker.threads_per_node=$oracle_workers"
  worker.use_distributed=false
  gpu=false
)
summary_command=(
  python "$project_root/tools/collect_register64_oracle_pdms_results.py"
  --oracle-report "$oracle_selection_root/oracle_report.json"
  --official-results-dir "$official_results"
  --source-summary "$source_summary"
  --datalist "$navtest_datalist"
  --output-dir "$summary_root"
  --expected-scenarios 12146
)

if (( dry_run )); then
  echo "[oracle64] dry_run=1 writes=0 imports=0"
  echo "[oracle64] project_root=$project_root"
  echo "[oracle64] source_run=$source_run"
  echo "[oracle64] source_checkpoints=generator:$generator_checkpoint drivor:$drivor_checkpoint"
  echo "[oracle64] checkpoint_steps=$checkpoint_steps checkpoint_min_age=$CHECKPOINT_MIN_AGE_SECONDS"
  echo "[oracle64] protocol=navsim-v1.1-pdms candidates=64 scenes=12146"
  echo "[oracle64] topology=machines:$NUM_MACHINES machine_rank:$MACHINE_RANK local:$LOCAL_NUM_PROCESSES total:$NUM_PROCESSES"
  echo "[oracle64] cpu_workers=$oracle_workers inference_batch_per_rank=$REGISTER64_ORACLE_INFER_BATCH_SIZE"
  echo "[oracle64] datalist=$navtest_datalist data_root=$data_root metric_cache=$pdms_metric_cache"
  echo "[oracle64] output=$run_root"
  echo "[oracle64] overwrite=$overwrite resume=$resume cache_generation=$AUTO_GENERATE_CACHES"
  print_command "${candidate_command[@]}"
  print_command "${oracle_command[@]}"
  print_command env CUDA_VISIBLE_DEVICES= PYTHONPATH="$project_root/navsim_v1.1/navsim:$project_root:${PYTHONPATH:-}" NAVSIM_DEVKIT_ROOT="$project_root/navsim_v1.1/navsim" NAVSIM_EXP_ROOT="$official_results" OPENSCENE_DATA_ROOT="$OPENSCENE_DATA_ROOT" "${official_command[@]}"
  print_command "${summary_command[@]}"
  echo "[oracle64] accelerator/model/evaluation=NOT_RUN"
  exit 0
fi

oracle_phase=bootstrap
on_oracle_error() {
  local status="$?"
  if (( BASH_SUBSHELL > 0 )); then return "$status"; fi
  echo "[oracle64] failed phase=$oracle_phase line=${BASH_LINENO[0]} status=$status" >&2
  exit "$status"
}
trap on_oracle_error ERR

require_uint() {
  local name="$1" value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "[oracle64] $name must be a non-negative integer: $value" >&2
    exit 2
  fi
}
required_path() {
  if [[ ! -e "$1" ]]; then echo "[oracle64] missing required path: $1" >&2; exit 2; fi
}

oracle_phase=contract
for pair in \
  "LOCAL_NUM_PROCESSES:$LOCAL_NUM_PROCESSES" \
  "NUM_MACHINES:$NUM_MACHINES" \
  "MACHINE_RANK:$MACHINE_RANK" \
  "NUM_PROCESSES:$NUM_PROCESSES" \
  "REGISTER64_ORACLE_MAIN_PROCESS_PORT:$REGISTER64_ORACLE_MAIN_PROCESS_PORT" \
  "REGISTER64_ORACLE_INFER_BATCH_SIZE:$REGISTER64_ORACLE_INFER_BATCH_SIZE" \
  "REGISTER64_ORACLE_INFER_WORKERS:$REGISTER64_ORACLE_INFER_WORKERS" \
  "CHECKPOINT_MIN_AGE_SECONDS:$CHECKPOINT_MIN_AGE_SECONDS" \
  "REGISTER64_ORACLE_WORKERS:$oracle_workers"; do
  require_uint "${pair%%:*}" "${pair#*:}"
done
if (( NUM_MACHINES != 1 || MACHINE_RANK != 0 )); then
  echo "[oracle64] this formal evaluator requires one DLC node" >&2
  exit 2
fi
if (( NUM_PROCESSES != LOCAL_NUM_PROCESSES || NUM_PROCESSES < 1 )); then
  echo "[oracle64] NUM_PROCESSES must equal LOCAL_NUM_PROCESSES and be positive" >&2
  exit 2
fi
if (( oracle_workers < 1 || REGISTER64_ORACLE_INFER_BATCH_SIZE < 1 )); then
  echo "[oracle64] worker count and inference batch must be positive" >&2
  exit 2
fi
if [[ "$checkpoint_steps" != best ]]; then
  echo "[oracle64] --checkpoint-steps is fixed to 'best' for this completed-run diagnostic" >&2
  exit 2
fi
if (( resume && overwrite )); then
  echo "[oracle64] --resume and --overwrite are mutually exclusive" >&2
  exit 2
fi
if ! [[ "$run_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[oracle64] run ID contains unsupported characters: $run_id" >&2
  exit 2
fi
case "$REGISTER64_ORACLE_ALLOW_DIRTY" in 0|1) ;; *) echo "[oracle64] REGISTER64_ORACLE_ALLOW_DIRTY must be 0 or 1" >&2; exit 2 ;; esac
source_prefix="$source_run/"
run_prefix="$(realpath -m "$run_root")/"
if [[ "$run_prefix" == "$source_prefix"* || "$source_prefix" == "$run_prefix"* ]]; then
  echo "[oracle64] evaluation output must be isolated from the source training run" >&2
  exit 2
fi
if [[ -e "$run_root" && "$resume" != 1 && "$overwrite" != 1 ]]; then
  echo "[oracle64] Refusing to overwrite existing prediction/artifact output: $run_root; use --resume, --overwrite, or a new --run-id" >&2
  exit 2
fi

actual_branch="$(env -u GIT_DIR -u GIT_WORK_TREE git -C "$project_root" branch --show-current)"
source_commit="$(env -u GIT_DIR -u GIT_WORK_TREE git -C "$project_root" rev-parse HEAD)"
if [[ "$actual_branch" != "$REGISTER64_ORACLE_EXPECTED_BRANCH" ]]; then
  echo "[oracle64] wrong branch: expected=$REGISTER64_ORACLE_EXPECTED_BRANCH actual=${actual_branch:-DETACHED}" >&2
  exit 2
fi
dirty_state="$(env -u GIT_DIR -u GIT_WORK_TREE git -C "$project_root" status --short)"
if [[ "$REGISTER64_ORACLE_ALLOW_DIRTY" != 1 && -n "$dirty_state" ]]; then
  echo "[oracle64] formal source is dirty; commit it or set REGISTER64_ORACLE_ALLOW_DIRTY=1 explicitly" >&2
  exit 2
fi

for path in "$source_run" "$generator_config" "$generator_checkpoint" \
  "$generator_complete" "$drivor_checkpoint" "$drivor_complete" \
  "$source_summary" "$navtest_datalist" "$data_root/meta/test" \
  "$pdms_metric_cache" "$QWEN_VLM_PATH/config.json" \
  "$OPENSCENE_DATA_ROOT" "$NUPLAN_MAPS_ROOT" \
  "$REGISTER64_NAVTEST_LOG_ROOT" "$project_root/navsim_v1.1/navsim"; do
  required_path "$path"
done
current_epoch="$(date +%s)"
for checkpoint in "$generator_checkpoint" "$drivor_checkpoint"; do
  checkpoint_age=$((current_epoch - $(stat -c %Y "$checkpoint")))
  if (( checkpoint_age < CHECKPOINT_MIN_AGE_SECONDS )); then
    echo "[oracle64] unstable checkpoint age: $checkpoint age=${checkpoint_age}s minimum=${CHECKPOINT_MIN_AGE_SECONDS}s" >&2
    exit 2
  fi
done
if ! compgen -G "$QWEN_VLM_PATH/*.safetensors" >/dev/null; then
  echo "[oracle64] Qwen safetensors are missing under $QWEN_VLM_PATH" >&2
  exit 2
fi

oracle_phase=input-identity
python - "$generator_complete" "$generator_checkpoint" register_generator \
  "$drivor_complete" "$drivor_checkpoint" drivor_scorer <<'PY'
import hashlib
import json
import sys
from pathlib import Path

for offset in (1, 4):
    marker = Path(sys.argv[offset])
    checkpoint = Path(sys.argv[offset + 1])
    expected_stage = sys.argv[offset + 2]
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("stage") != expected_stage:
        raise RuntimeError(f"invalid stage completion marker: {marker}")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    if payload.get("selected_checkpoint") != checkpoint.name:
        raise RuntimeError(f"selected checkpoint name mismatch: {marker}")
    if payload.get("selected_checkpoint_sha256") != digest.hexdigest():
        raise RuntimeError(f"checkpoint hash mismatch: {checkpoint}")
print("[oracle64] source checkpoint contracts passed")
PY

expected_navtest="$(python -c 'import json,sys; value=json.load(open(sys.argv[1])); assert len(value)==len(set(value)); print(len(value))' "$navtest_datalist")"
if (( expected_navtest != 12146 )); then
  echo "[oracle64] formal navtest must contain 12146 unique scenes, found $expected_navtest" >&2
  exit 2
fi
python "$project_root/tools/validate_navsim_metric_cache.py" \
  --cache-root "$pdms_metric_cache" \
  --expected-datalist "$navtest_datalist" \
  --check-cache-files
python - "$generator_config" "$source_summary" "$expected_navtest" <<'PY'
import csv
import sys
from starVLA.training.config_loader import load_training_config

config = load_training_config(sys.argv[1])
assert config.framework.name == "QwenRegisterPlanner"
assert config.framework.inference.selector_type == "drivor"
assert int(config.framework.register_generator.proposal_num) == 64
assert bool(config.framework.inference.return_all_proposals)
with open(sys.argv[2], newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
pdms = [row for row in rows if row.get("protocol") == "pdms_v1_1"]
assert len(pdms) == 1 and int(pdms[0]["num_scenarios"]) == int(sys.argv[3])
print(f"[oracle64] source official PDMS={float(pdms[0]['score_percent']):.3f}")
PY

export PYTHONPATH="$project_root:$project_root/navsim:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export NAVSIM_USE_FEATURE_CACHE=0
unset NAVSIM_FEATURE_CACHE_ROOT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE=offline

# Accelerate owns the formal rank assignment. Inherited launcher variables can
# otherwise silently create an invalid WORLD_SIZE/RANK shard contract.
unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK ROLE_RANK

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((LOCAL_NUM_PROCESSES - 1)))"
fi
if (( NUM_PROCESSES == 1 )); then
  python "$project_root/tools/check_ppu_runtime.py" --expected-world-size 1
else
  accelerate launch --multi_gpu --mixed_precision bf16 \
    --num_machines 1 --num_processes "$NUM_PROCESSES" \
    --machine_rank 0 --main_process_port $((REGISTER64_ORACLE_MAIN_PROCESS_PORT + 1)) \
    --num_cpu_threads_per_process 1 \
    "$project_root/tools/check_ppu_runtime.py" \
    --expected-world-size "$NUM_PROCESSES"
fi

echo "[oracle64] source=$actual_branch@$source_commit dirty_allowed=$REGISTER64_ORACLE_ALLOW_DIRTY"
echo "[oracle64] source_run=$source_run"
echo "[oracle64] protocol=navsim-v1.1-pdms candidates=64 scenes=$expected_navtest"
echo "[oracle64] topology=machines:$NUM_MACHINES local_processes:$LOCAL_NUM_PROCESSES total_processes:$NUM_PROCESSES cpu_workers:$oracle_workers"
echo "[oracle64] inputs=datalist:$navtest_datalist cache:$pdms_metric_cache"
echo "[oracle64] cache_generation=$AUTO_GENERATE_CACHES (immutable source cache only)"
echo "[oracle64] outputs=$run_root"
if (( preflight_only )); then
  if [[ -e "$run_root" && "$overwrite" == 1 ]]; then
    echo "[oracle64] overwrite plan: move exact run to a recoverable timestamped sibling"
  fi
  echo "[oracle64] full preflight passed; model_inference=NOT_RUN official_scoring=NOT_RUN"
  exit 0
fi

if [[ -e "$run_root" && "$overwrite" == 1 ]]; then
  backup_root="${run_root}.replaced.$(date +'%Y%m%d_%H%M%S')"
  if [[ -e "$backup_root" ]]; then
    echo "[oracle64] overwrite backup already exists: $backup_root" >&2
    exit 2
  fi
  mv -- "$run_root" "$backup_root"
  echo "[oracle64] previous exact run moved to recoverable backup: $backup_root"
fi
mkdir -p "$output_root/launcher_logs"
exec > >(tee -a "$launcher_log") 2>&1
mkdir -p "$run_root"
export TRITON_CACHE_DIR="${TRITON_LOCAL_CACHE_ROOT:-/tmp/register64-oracle-triton}/$run_id"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_LOCAL_ROOT:-/tmp/register64-oracle-extensions}/$run_id"
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

oracle_phase=candidate-export
if [[ -e "$prediction_dir" ]]; then candidate_command+=(--resume); fi
print_command "${candidate_command[@]}"
"${candidate_command[@]}"
candidate_count="$(find "$candidate_dir" -maxdepth 1 -type f -name '*.npz' | wc -l)"
prediction_count="$(find "$prediction_dir" -maxdepth 1 -type f -name '*.npy' | wc -l)"
rank_manifest_count="$(find "$prediction_dir" -maxdepth 1 -type f -name 'inference_manifest.rank*.json' | wc -l)"
if (( candidate_count != expected_navtest || prediction_count != expected_navtest || rank_manifest_count != NUM_PROCESSES )); then
  echo "[oracle64] candidate export count mismatch: candidates=$candidate_count predictions=$prediction_count rank_manifests=$rank_manifest_count expected_scenes=$expected_navtest expected_ranks=$NUM_PROCESSES" >&2
  exit 2
fi

oracle_phase=oracle-selection
if [[ -e "$oracle_selection_root/oracle_scores.sqlite3" ]]; then oracle_command+=(--resume); fi
print_command "${oracle_command[@]}"
env \
  CUDA_VISIBLE_DEVICES= \
  PYTHONPATH="$project_root/navsim_v1.1/navsim:$project_root:${PYTHONPATH:-}" \
  NAVSIM_DEVKIT_ROOT="$project_root/navsim_v1.1/navsim" \
  OPENSCENE_DATA_ROOT="$OPENSCENE_DATA_ROOT" \
  "${oracle_command[@]}"

score_valid() {
  python "$project_root/tools/validate_navsim_score_csv.py" \
    --results-dir "$official_results" --protocol pdms \
    --expected-scenarios "$expected_navtest" \
    --expected-datalist "$navtest_datalist" >/dev/null 2>&1
}

oracle_phase=official-oracle-pdms
if score_valid; then
  echo "[oracle64] official Oracle@64 PDMS already valid: $official_results"
else
  mkdir -p "$official_results"
  print_command env CUDA_VISIBLE_DEVICES= PYTHONPATH="$project_root/navsim_v1.1/navsim:$project_root:${PYTHONPATH:-}" NAVSIM_DEVKIT_ROOT="$project_root/navsim_v1.1/navsim" NAVSIM_EXP_ROOT="$official_results" OPENSCENE_DATA_ROOT="$OPENSCENE_DATA_ROOT" "${official_command[@]}"
  env \
    CUDA_VISIBLE_DEVICES= \
    PYTHONPATH="$project_root/navsim_v1.1/navsim:$project_root:${PYTHONPATH:-}" \
    NAVSIM_DEVKIT_ROOT="$project_root/navsim_v1.1/navsim" \
    NAVSIM_EXP_ROOT="$official_results" \
    OPENSCENE_DATA_ROOT="$OPENSCENE_DATA_ROOT" \
    "${official_command[@]}"
fi
python "$project_root/tools/validate_navsim_score_csv.py" \
  --results-dir "$official_results" --protocol pdms \
  --expected-scenarios "$expected_navtest" \
  --expected-datalist "$navtest_datalist"

oracle_phase=summary
print_command "${summary_command[@]}"
"${summary_command[@]}"
# The Python collector writes summary.csv through os.replace. Keep a separate
# atomic protocol snapshot at the shell boundary as well.
protocol_tmp="$(mktemp "$summary_root/.protocol.json.tmp.XXXXXX")"
python - "$source_commit" "$run_id" >"$protocol_tmp" <<'PY'
import json
import sys
json.dump(
    {
        "repository_commit": sys.argv[1],
        "run_id": sys.argv[2],
        "protocol": "navsim_v1.1_pdms_oracle_at_64",
        "checkpoint_steps": "best",
        "candidate_count": 64,
    },
    sys.stdout,
    indent=2,
    sort_keys=True,
)
print()
PY
mv -f "$protocol_tmp" "$summary_root/protocol.json"
echo "[oracle64] complete"
echo "[oracle64] report=$summary_root/summary.md"
echo "[oracle64] scores=$summary_root/summary.csv"
