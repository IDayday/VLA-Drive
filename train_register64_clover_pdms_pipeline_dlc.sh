#!/usr/bin/env bash
# Complete non-interactive PDMS-first route:
# v1 cache -> Stage 1 -> 30x(bank, critic, generator) -> closing critic -> navtest PDMS.

set -Eeuo pipefail

dry_run="${DRY_RUN:-0}"
preflight_only="${PREFLIGHT_ONLY:-0}"
resume="${CLOVER_RESUME:-0}"
profile_steps="${CLOVER_PROFILE_STEPS:-0}"
while (( $# )); do
  case "$1" in
    --dry-run) dry_run=1 ;;
    --preflight-only) preflight_only=1 ;;
    --resume) resume=1 ;;
    --profile-steps)
      shift
      if (( ! $# )); then echo "[clover-pdms] --profile-steps requires a value" >&2; exit 2; fi
      profile_steps="$1"
      ;;
    --help|-h)
      echo "Usage: $0 [--dry-run] [--preflight-only] [--resume] [--profile-steps N]"
      exit 0
      ;;
    *) echo "[clover-pdms] unsupported argument: $1" >&2; exit 2 ;;
  esac
  shift
done

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"
cd "$project_root"

export CLOVER_EXPECTED_BRANCH="${CLOVER_EXPECTED_BRANCH:-feature/ddp-drs-scene-2048}"
export NUM_MACHINES="${NUM_MACHINES:-1}"
export MACHINE_RANK="${MACHINE_RANK:-0}"
export LOCAL_NUM_PROCESSES="${LOCAL_NUM_PROCESSES:-16}"
export NUM_PROCESSES="${NUM_PROCESSES:-$LOCAL_NUM_PROCESSES}"
export CLOVER_MAIN_PROCESS_PORT="${CLOVER_MAIN_PROCESS_PORT:-29831}"
export CLOVER_NUM_CYCLES="${CLOVER_NUM_CYCLES:-30}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-32}"
# The recipe is epoch/cycle driven; these values make that budget explicit in
# the formal run identity instead of inventing one global optimizer-step cap.
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-epoch_driven_25_plus_${CLOVER_NUM_CYCLES}_cycles}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-stage1_5epochs_then_each_cycle}"
export CLOVER_CACHE_WORKERS="${CLOVER_CACHE_WORKERS:-96}"
export CLOVER_SPLIT_WORKERS="${CLOVER_SPLIT_WORKERS:-32}"
export NAVSIM_NUM_WORKERS="${NAVSIM_NUM_WORKERS:-3}"
export NAVSIM_METRIC_WORKERS="${NAVSIM_METRIC_WORKERS:-4}"
export REGISTER64_BANK_LOADER_WORKERS="${REGISTER64_BANK_LOADER_WORKERS:-7}"
export CLOVER_VALIDATION_SIZE="${CLOVER_VALIDATION_SIZE:-2048}"
export CLOVER_SELECTION_SIZE="${CLOVER_SELECTION_SIZE:-2048}"
export CLOVER_VALIDATION_SCENES="${CLOVER_VALIDATION_SCENES:-1024}"
export CLOVER_INFER_BATCH_SIZE="${CLOVER_INFER_BATCH_SIZE:-4}"
export CLOVER_INFER_WORKERS="${CLOVER_INFER_WORKERS:-3}"
export CLOVER_BUILD_CACHES="${CLOVER_BUILD_CACHES:-auto}"
export CLOVER_ALLOW_DIRTY="${CLOVER_ALLOW_DIRTY:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CLOVER_RUN_ID="${CLOVER_RUN_ID:-register64-clover-pdms-$(date +'%Y%m%d_%H%M%S')}"
export CLOVER_OUTPUT_ROOT="${CLOVER_OUTPUT_ROOT:-${NAVSIM_EXP_ROOT:-$project_root/navsim_exp}/register64_clover_pdms}"
export DRY_RUN="$dry_run"
export PREFLIGHT_ONLY="$preflight_only"
export CLOVER_PROFILE_STEPS="$profile_steps"

export QWEN_VLM_PATH="${QWEN_VLM_PATH:-${BASE_VLM:-}}"
export VLM_ATTN_IMPLEMENTATION="${VLM_ATTN_IMPLEMENTATION:-sdpa}"
export DATA_ROOT="${DATA_ROOT:-$project_root/navsim_dataset}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$project_root/navsim_dataset_raw}"
source_datalist_default="${NAVSIM_DATALIST_PATH:-$project_root/train_meta.json}"
export CLOVER_SOURCE_DATALIST="${CLOVER_SOURCE_DATALIST:-$source_datalist_default}"
datalist_parent="$(cd -- "$(dirname -- "$CLOVER_SOURCE_DATALIST")" 2>/dev/null && pwd || dirname -- "$CLOVER_SOURCE_DATALIST")"
export CLOVER_NAVTEST_DATALIST="${CLOVER_NAVTEST_DATALIST:-$datalist_parent/test_meta.json}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$OPENSCENE_DATA_ROOT/maps}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export CLOVER_NAVTRAIN_LOG_ROOT="${CLOVER_NAVTRAIN_LOG_ROOT:-${QDS_NAVSIM_LOG_PATH:-$OPENSCENE_DATA_ROOT/navsim_logs/trainval}}"
export CLOVER_NAVTEST_LOG_ROOT="${CLOVER_NAVTEST_LOG_ROOT:-$OPENSCENE_DATA_ROOT/navsim_logs/test}"
export NAVSIM_USE_FEATURE_CACHE=0
unset NAVSIM_FEATURE_CACHE_ROOT NAVSIM_AGENT_DINO_CACHE_ROOT NAVSIM_VGGT_CACHE_ROOT

run_root="$CLOVER_OUTPUT_ROOT/$CLOVER_RUN_ID"
stage1_dir="$run_root/stage1"
cycles_root="$run_root/cycles"
split_root="$run_root/splits"
prediction_root="$run_root/predictions"
prediction_dir="$prediction_root/test"
pdms_results="$run_root/evaluation/pdms_v1_1"
summary_root="$run_root/summary"
shared_cache_root="${CLOVER_SHARED_CACHE_ROOT:-${NAVSIM_EXP_ROOT:-$project_root/navsim_exp}/shared_metric_cache/navsim_v1_1}"
export CLOVER_PDMS_TRAIN_METRIC_CACHE="${CLOVER_PDMS_TRAIN_METRIC_CACHE:-$shared_cache_root/navtrain}"
export CLOVER_PDMS_NAVTEST_METRIC_CACHE="${CLOVER_PDMS_NAVTEST_METRIC_CACHE:-$shared_cache_root/navtest}"
train_datalist="$split_root/train.json"
val_datalist="$split_root/val.json"
selection_datalist="$split_root/selection.json"

stage1_config="$project_root/starVLA/config/training/qwen_register64_clover_pdms_stage1.yaml"
bank_config="$project_root/starVLA/config/training/register64_clover_pdms_bank.yaml"
scorer_config="$project_root/starVLA/config/training/register64_clover_pdms_scorer.yaml"
refinement_config="$project_root/starVLA/config/training/qwen_register64_clover_pdms_stage2.yaml"
inference_config="$project_root/starVLA/config/training/register64_clover_pdms_inference.yaml"
deepspeed_config="${CLOVER_DEEPSPEED_CONFIG:-$project_root/starVLA/config/deepseeds/deepspeed_zero1.yaml}"

print_command() {
  printf '[clover-pdms] command:'
  printf ' %q' "$@"
  printf '\n'
}

if (( dry_run )); then
  echo "[clover-pdms] dry_run=1 writes=0 imports=0"
  echo "[clover-pdms] project_root=$project_root"
  echo "[clover-pdms] run_root=$run_root"
  echo "[clover-pdms] topology=machines:$NUM_MACHINES rank:$MACHINE_RANK local:$LOCAL_NUM_PROCESSES total:$NUM_PROCESSES"
  echo "[clover-pdms] effective batch=$TARGET_EFFECTIVE_BATCH_SIZE per_device=$PER_DEVICE_BATCH_SIZE accumulation=$GRADIENT_ACCUMULATION_STEPS"
  echo "[clover-pdms] recipe=stage1:25epochs cycles:$CLOVER_NUM_CYCLES*(critic1+generator1) closing_critic:1"
  echo "[clover-pdms] labels=NAVSIM-v1.1-PDMS pseudo_experts=${CLOVER_PSEUDO_EXPERT_PKL:-REQUIRED}"
  print_command python tools/prepare_register64_train_val_split.py --source "$CLOVER_SOURCE_DATALIST" --output-dir "$split_root" --validation-size "$CLOVER_VALIDATION_SIZE" --selection-size "$CLOVER_SELECTION_SIZE" --metadata-root "$DATA_ROOT/meta/train" --metadata-workers "$CLOVER_SPLIT_WORKERS" --require-log-disjoint --seed 2
  print_command python navsim_v1.1/navsim/navsim/planning/script/run_metric_caching.py train_test_split=navtrain "cache.cache_path=$CLOVER_PDMS_TRAIN_METRIC_CACHE" "worker.threads_per_node=$CLOVER_CACHE_WORKERS"
  if (( profile_steps )); then
    print_command accelerate launch --config_file "$deepspeed_config" --num_processes "$NUM_PROCESSES" starVLA/training/train_register_clover_stage1.py --config "$stage1_config" --profile-steps "$profile_steps"
    echo "[clover-pdms] profile_only=1 formal_training=NOT_RUN"
    exit 0
  else
    print_command accelerate launch --config_file "$deepspeed_config" --num_processes "$NUM_PROCESSES" starVLA/training/train_register_clover_stage1.py --config "$stage1_config"
  fi
  echo "[clover-pdms] repeat cycles 01..$CLOVER_NUM_CYCLES in critic-first order:"
  print_command accelerate launch --multi_gpu --num_processes "$NUM_PROCESSES" starVLA/training/build_register_candidate_bank.py --config "$bank_config" --split train
  print_command accelerate launch --multi_gpu --num_processes "$NUM_PROCESSES" starVLA/training/build_register_candidate_bank.py --config "$bank_config" --split val
  print_command accelerate launch --multi_gpu --num_processes "$NUM_PROCESSES" starVLA/training/build_register_candidate_bank.py --config "$bank_config" --split selection
  print_command accelerate launch --multi_gpu --num_processes "$NUM_PROCESSES" starVLA/training/train_register_drivor.py --config "$scorer_config"
  print_command accelerate launch --multi_gpu --num_processes "$NUM_PROCESSES" starVLA/training/train_register_clover_refinement.py --config "$refinement_config"
  echo "[clover-pdms] after cycle $CLOVER_NUM_CYCLES: refresh bank and fit one closing critic"
  print_command python tools/select_register64_clover_checkpoint_pair.py --candidate cycle_01 GENERATOR SCORER TRAIN_BANK SELECTION_BANK --output MODEL_SELECTION_JSON
  print_command accelerate launch --multi_gpu --num_processes "$NUM_PROCESSES" starVLA/training/export_register_navtest_predictions.py --config "$inference_config" --output-dir "$prediction_dir"
  print_command python navsim_v1.1/navsim/navsim/planning/script/run_pdm_score.py train_test_split=navtest "metric_cache_path=$CLOVER_PDMS_NAVTEST_METRIC_CACHE" "pred_dir=$prediction_root" split=test
  echo "[clover-pdms] formal_training=NOT_RUN"
  exit 0
fi

case "$resume:$preflight_only" in
  [01]:[01]) ;;
  *) echo "[clover-pdms] resume/preflight flags must be 0 or 1" >&2; exit 2 ;;
esac
if (( ! preflight_only )) && [[ -e "$run_root" && "$resume" != 1 ]]; then
  echo "[clover-pdms] Refusing to overwrite existing run_root=$run_root; nothing is overwritten. Choose a new CLOVER_RUN_ID or pass --resume" >&2
  exit 2
fi

mkdir -p "$CLOVER_OUTPUT_ROOT/launcher_logs"
launcher_log="$CLOVER_OUTPUT_ROOT/launcher_logs/${CLOVER_RUN_ID}.log"
exec > >(tee -a "$launcher_log") 2>&1
clover_phase=bootstrap
on_error() {
  local status="$?"
  if (( BASH_SUBSHELL > 0 )); then return "$status"; fi
  echo "[clover-pdms] failed phase=$clover_phase line=${BASH_LINENO[0]} status=$status" >&2
  exit "$status"
}
trap on_error ERR

require_uint() {
  local name="$1"
  if ! [[ "${!name}" =~ ^[0-9]+$ ]]; then
    echo "[clover-pdms] $name must be an unsigned integer: ${!name}" >&2
    exit 2
  fi
}
require_value() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "[clover-pdms] required variable is empty: $name" >&2
    exit 2
  fi
}
require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "[clover-pdms] missing required path: $path" >&2
    exit 2
  fi
}

clover_phase=source-contract
for name in NUM_MACHINES MACHINE_RANK LOCAL_NUM_PROCESSES NUM_PROCESSES \
  CLOVER_MAIN_PROCESS_PORT CLOVER_NUM_CYCLES CLOVER_CACHE_WORKERS CLOVER_SPLIT_WORKERS \
  NAVSIM_NUM_WORKERS NAVSIM_METRIC_WORKERS REGISTER64_BANK_LOADER_WORKERS \
  CLOVER_VALIDATION_SIZE CLOVER_SELECTION_SIZE CLOVER_VALIDATION_SCENES CLOVER_INFER_BATCH_SIZE \
  CLOVER_INFER_WORKERS CLOVER_ALLOW_DIRTY PER_DEVICE_BATCH_SIZE \
  GRADIENT_ACCUMULATION_STEPS TARGET_EFFECTIVE_BATCH_SIZE DRY_RUN \
  PREFLIGHT_ONLY CLOVER_PROFILE_STEPS; do
  require_uint "$name"
done
if (( NUM_MACHINES != 1 || MACHINE_RANK != 0 || NUM_PROCESSES != LOCAL_NUM_PROCESSES )); then
  echo "[clover-pdms] formal pipeline requires one node with NUM_PROCESSES=LOCAL_NUM_PROCESSES" >&2
  exit 2
fi
if (( NUM_PROCESSES != 16 )); then
  echo "[clover-pdms] formal recipe requires exactly 16 PPU; found $NUM_PROCESSES" >&2
  exit 2
fi
if (( CLOVER_NUM_CYCLES < 1 || CLOVER_NUM_CYCLES > 30 )); then
  echo "[clover-pdms] CLOVER_NUM_CYCLES must lie in [1,30]" >&2
  exit 2
fi
if (( 32 % NUM_PROCESSES != 0 )); then
  echo "[clover-pdms] world size must divide CLOVER global batch 32" >&2
  exit 2
fi
effective_batch=$((PER_DEVICE_BATCH_SIZE * NUM_PROCESSES * GRADIENT_ACCUMULATION_STEPS))
if (( effective_batch != TARGET_EFFECTIVE_BATCH_SIZE || effective_batch != 32 )); then
  echo "[clover-pdms] effective batch mismatch actual=$effective_batch target=$TARGET_EFFECTIVE_BATCH_SIZE recipe=32" >&2
  exit 2
fi
if (( CLOVER_VALIDATION_SIZE < CLOVER_VALIDATION_SCENES )); then
  echo "[clover-pdms] holdout must cover the fixed validation subset" >&2
  exit 2
fi
if (( CLOVER_SELECTION_SIZE < 1024 )); then
  echo "[clover-pdms] selection holdout must contain at least 1024 target scenes" >&2
  exit 2
fi
case "$CLOVER_BUILD_CACHES" in auto|0|1) ;; *) echo "[clover-pdms] CLOVER_BUILD_CACHES must be auto, 0, or 1" >&2; exit 2 ;; esac
if (( profile_steps && (preflight_only || resume) )); then
  echo "[clover-pdms] profiling is mutually exclusive with preflight-only/resume" >&2
  exit 2
fi
if ! [[ "$CLOVER_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[clover-pdms] CLOVER_RUN_ID contains unsupported characters" >&2
  exit 2
fi

actual_branch="$(env -u GIT_DIR -u GIT_WORK_TREE git -C "$project_root" branch --show-current)"
source_commit="$(env -u GIT_DIR -u GIT_WORK_TREE git -C "$project_root" rev-parse HEAD)"
if [[ "$actual_branch" != "$CLOVER_EXPECTED_BRANCH" ]]; then
  echo "[clover-pdms] wrong branch expected=$CLOVER_EXPECTED_BRANCH actual=${actual_branch:-DETACHED}" >&2
  exit 2
fi
if (( ! CLOVER_ALLOW_DIRTY )) && [[ -n "$(env -u GIT_DIR -u GIT_WORK_TREE git -C "$project_root" status --porcelain)" ]]; then
  echo "[clover-pdms] worktree is dirty; formal artifact identity would be ambiguous" >&2
  exit 2
fi
require_value QWEN_VLM_PATH
require_value CLOVER_PSEUDO_EXPERT_PKL
for path in "$QWEN_VLM_PATH" "$CLOVER_PSEUDO_EXPERT_PKL" "$DATA_ROOT" \
  "$OPENSCENE_DATA_ROOT" "$NUPLAN_MAPS_ROOT" "$CLOVER_NAVTRAIN_LOG_ROOT" \
  "$CLOVER_NAVTEST_LOG_ROOT" "$CLOVER_SOURCE_DATALIST" \
  "$DATA_ROOT/meta/train" \
  "$CLOVER_NAVTEST_DATALIST" "$project_root/navsim_v1.1/navsim" \
  "$stage1_config" "$bank_config" "$scorer_config" "$refinement_config" \
  "$inference_config" "$deepspeed_config"; do
  require_path "$path"
done

export PYTHONPATH="$project_root/navsim_v1.1/navsim:$project_root:${PYTHONPATH:-}"
export NAVSIM_DEVKIT_ROOT="$project_root/navsim_v1.1/navsim"
python - "$stage1_config" "$bank_config" "$scorer_config" "$refinement_config" "$inference_config" <<'PY'
import sys
from starVLA.training.config_loader import load_training_config
from starVLA.training.register_stage_utils import validate_bank_only_training_profile

stage1, bank, scorer, refinement, inference = map(load_training_config, sys.argv[1:])
assert stage1.framework.name == "QwenRegisterClover"
assert stage1.metric_supervisor.protocol == "navsim_v1_1_pdms_two_way"
assert stage1.framework.generator_loss.stage_loss_mode == "final_only"
assert stage1.framework.register_generator.proposal_num == 64
assert stage1.trainer.max_epochs == 25 and stage1.trainer.global_batch_size == 32
assert stage1.framework.qwenvl.freeze_visual is False
assert bank.candidate_bank.label_protocol == "navsim_v1_1_pdms_two_way"
assert bank.candidate_bank.splits.selection.dataset_split == "train"
assert scorer.candidate_bank.label_protocol == "navsim_v1_1_pdms_two_way"
assert "selection_root" in scorer.candidate_bank
assert scorer.model.aggregate_head is True
assert scorer.model.selection_mode == "calibrated_hybrid"
assert scorer.trainer.epochs == 1 and scorer.trainer.global_batch_size == 32
validate_bank_only_training_profile(scorer, expected_name="clover_pdms_value_bank_v1")
assert refinement.loss.trajectory_weight == 0.1
assert refinement.loss.diversity_weight == 0.02
assert refinement.loss.topk_weight == 1.0
assert refinement.loss.pareto_weight == 1.0
assert refinement.loss.stability_weight == 0.05
assert inference.framework.inference.selector_type == "drivor"
assert inference.framework.drivor_scorer.selection_mode == "calibrated_hybrid"
assert inference.framework.drivor_scorer.label_protocol == "navsim_v1_1_pdms_two_way"
print("[clover-pdms] config contract passed")
PY

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((LOCAL_NUM_PROCESSES - 1)))"
fi
accelerate launch --multi_gpu --mixed_precision bf16 \
  --num_machines 1 --num_processes "$NUM_PROCESSES" --machine_rank 0 \
  --main_process_port $((CLOVER_MAIN_PROCESS_PORT + 90)) \
  --num_cpu_threads_per_process 1 \
  "$project_root/tools/check_ppu_runtime.py" --expected-world-size "$NUM_PROCESSES"
available_cpus="$(nproc)"
required_cpu_slots=$((LOCAL_NUM_PROCESSES * (1 + NAVSIM_NUM_WORKERS + NAVSIM_METRIC_WORKERS)))
if (( available_cpus < required_cpu_slots )); then
  echo "[clover-pdms] CPU contract failed available=$available_cpus required=$required_cpu_slots" >&2
  exit 2
fi
echo "[clover-pdms] source=$actual_branch@$source_commit"
echo "[clover-pdms] topology=16PPU CPU:$available_cpus stage1_batch=32(per_device:2)"
echo "[clover-pdms] CPU slots=rank16 loader:$((16 * NAVSIM_NUM_WORKERS)) metric:$((16 * NAVSIM_METRIC_WORKERS))"
echo "[clover-pdms] pseudo_experts=$CLOVER_PSEUDO_EXPERT_PKL"
echo "[clover-pdms] outputs=$run_root log=$launcher_log"
if (( preflight_only )); then
  echo "[clover-pdms] full preflight passed; formal_training=NOT_RUN"
  exit 0
fi

mkdir -p "$run_root" "$cycles_root" "$split_root" "$summary_root"
export TRITON_CACHE_DIR="${TRITON_LOCAL_CACHE_ROOT:-/tmp/register64-clover-triton}/$CLOVER_RUN_ID"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_LOCAL_ROOT:-/tmp/register64-clover-extensions}/$CLOVER_RUN_ID"
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

run_distributed() {
  local mode="$1" port="$2" script="$3"
  shift 3
  local command
  if [[ "$mode" == deepspeed ]]; then
    command=(
      accelerate launch --config_file "$deepspeed_config"
      --num_machines 1 --num_processes "$NUM_PROCESSES" --machine_rank 0
      --main_process_port "$port" --num_cpu_threads_per_process 1 "$script"
    )
  else
    command=(
      accelerate launch --multi_gpu --mixed_precision bf16
      --num_machines 1 --num_processes "$NUM_PROCESSES" --machine_rank 0
      --main_process_port "$port" --num_cpu_threads_per_process 1 "$script"
    )
  fi
  print_command "${command[@]}" "$@"
  "${command[@]}" "$@"
}

component_complete() {
  local marker="$1" checkpoint="$2" expected_stage="$3"
  python - "$marker" "$checkpoint" "$expected_stage" <<'PY' >/dev/null 2>&1
import hashlib, json, sys
from pathlib import Path
marker, checkpoint, stage = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
if not marker.is_file() or not checkpoint.is_file(): raise SystemExit(1)
payload = json.loads(marker.read_text())
if payload.get("status") != "complete" or payload.get("stage") != stage: raise SystemExit(1)
h = hashlib.sha256()
with checkpoint.open("rb") as f:
    for chunk in iter(lambda: f.read(8 << 20), b""): h.update(chunk)
if payload.get("selected_checkpoint") != checkpoint.name: raise SystemExit(1)
if payload.get("selected_checkpoint_sha256") != h.hexdigest(): raise SystemExit(1)
PY
}

stage1_complete() {
  local marker="$1" generator="$2" scorer="$3"
  python - "$marker" "$generator" "$scorer" <<'PY' >/dev/null 2>&1
import hashlib, json, sys
from pathlib import Path
marker, generator, scorer = map(Path, sys.argv[1:])
if not all(path.is_file() for path in (marker, generator, scorer)): raise SystemExit(1)
payload = json.loads(marker.read_text())
if payload.get("status") != "complete" or payload.get("stage") != "clover_stage1": raise SystemExit(1)
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""): h.update(chunk)
    return h.hexdigest()
if payload.get("generator_checkpoint") != generator.name or payload.get("generator_sha256") != digest(generator): raise SystemExit(1)
if payload.get("scorer_checkpoint") != scorer.name or payload.get("scorer_sha256") != digest(scorer): raise SystemExit(1)
PY
}

bank_complete() {
  local root="$1" generator="$2"
  python - "$root" "$generator" <<'PY' >/dev/null 2>&1
import hashlib, sys
from starVLA.candidate_bank import CandidateBankReader
root, checkpoint = sys.argv[1:]
h = hashlib.sha256()
with open(checkpoint, "rb") as f:
    for chunk in iter(lambda: f.read(8 << 20), b""): h.update(chunk)
reader = CandidateBankReader(root, expected_generator_checkpoint_sha256=h.hexdigest(), strict=True)
try:
    assert reader.manifest.complete
    assert reader.manifest.proposal_num == 64
    assert reader.manifest.label_protocol == "navsim_v1_1_pdms_two_way"
finally:
    reader.close()
PY
}

ensure_v1_cache() {
  local split="$1" root="$2" expected="$3" log_root="$4"
  local validator=(python "$project_root/tools/validate_navsim_metric_cache.py" --cache-root "$root" --expected-datalist "$expected" --check-cache-files)
  if "${validator[@]}" >/dev/null 2>&1; then
    echo "[clover-pdms] metric cache valid split=$split root=$root"
    return
  fi
  if [[ "$CLOVER_BUILD_CACHES" == 0 ]]; then
    echo "[clover-pdms] missing/incomplete metric cache and generation disabled: $root" >&2
    return 2
  fi
  mkdir -p "$root"
  echo "[clover-pdms] building NAVSIM-v1.1 cache split=$split workers=$CLOVER_CACHE_WORKERS root=$root"
  env CUDA_VISIBLE_DEVICES= \
    PYTHONPATH="$project_root/navsim_v1.1/navsim:$project_root:${PYTHONPATH:-}" \
    NAVSIM_DEVKIT_ROOT="$project_root/navsim_v1.1/navsim" \
    NAVSIM_EXP_ROOT="$root" OPENSCENE_DATA_ROOT="$OPENSCENE_DATA_ROOT" \
    python "$project_root/navsim_v1.1/navsim/navsim/planning/script/run_metric_caching.py" \
      "train_test_split=$split" "navsim_log_path=$log_root" \
      "cache.cache_path=$root" cache.force_feature_computation=false \
      worker=ray_distributed_no_torch "worker.threads_per_node=$CLOVER_CACHE_WORKERS" \
      worker.use_distributed=false gpu=false
  "${validator[@]}"
}

build_bank() {
  local root="$1" generator="$2" port="$3"
  export REGISTER64_BANK_ROOT="$root"
  export REGISTER64_GENERATOR_CHECKPOINT="$generator"
  for split in train val selection; do
    local split_root_path="$root/$split"
    if bank_complete "$split_root_path" "$generator"; then
      echo "[clover-pdms] bank already complete split=$split root=$split_root_path"
      continue
    fi
    local args=(--config "$bank_config" --split "$split" --workers-per-rank "$NAVSIM_METRIC_WORKERS" --backend process)
    if (( resume )) && [[ -f "$split_root_path/build_identity.json" ]]; then args+=(--resume); fi
    run_distributed plain "$port" "$project_root/starVLA/training/build_register_candidate_bank.py" "${args[@]}"
    bank_complete "$split_root_path" "$generator"
    port=$((port + 1))
  done
}

clover_phase=prepare-splits
python "$project_root/tools/prepare_register64_train_val_split.py" \
  --source "$CLOVER_SOURCE_DATALIST" --output-dir "$split_root" \
  --validation-size "$CLOVER_VALIDATION_SIZE" \
  --selection-size "$CLOVER_SELECTION_SIZE" \
  --metadata-root "$DATA_ROOT/meta/train" \
  --metadata-workers "$CLOVER_SPLIT_WORKERS" --require-log-disjoint --seed 2
export NAVSIM_DATALIST_PATH="$train_datalist"
export NAVSIM_VAL_DATALIST_PATH="$val_datalist"
export NAVSIM_SELECTION_DATALIST_PATH="$selection_datalist"

clover_phase=cache-navtrain-v1-1
ensure_v1_cache navtrain "$CLOVER_PDMS_TRAIN_METRIC_CACHE" \
  "$CLOVER_SOURCE_DATALIST" "$CLOVER_NAVTRAIN_LOG_ROOT"

clover_phase=stage1
export VLA_OUTPUT_ROOT="$run_root"
export CLOVER_STAGE1_RUN_ID=stage1
stage1_generator="$stage1_dir/best_pdms_generator.pt"
stage1_scorer="$stage1_dir/best_pdms_scorer.pt"
stage1_marker="$stage1_dir/training_complete.json"
if (( profile_steps )); then
  stage1_args=(
    "$project_root/starVLA/training/train_register_clover_stage1.py"
    --config "$stage1_config" --profile-steps "$profile_steps"
  )
  run_distributed deepspeed "$CLOVER_MAIN_PROCESS_PORT" \
    "${stage1_args[@]}"
  require_path "$stage1_dir/profile_complete.json"
  echo "[clover-pdms] production profile complete; formal_training=NOT_RUN"
  exit 0
elif stage1_complete "$stage1_marker" "$stage1_generator" "$stage1_scorer"; then
  echo "[clover-pdms] Stage 1 already complete"
else
  unset CLOVER_STAGE1_RESUME
  if (( resume )) && [[ -d "$stage1_dir/checkpoints" ]]; then
    latest="$(find "$stage1_dir/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'epoch_*' -print | sort | tail -n 1)"
    if [[ -n "$latest" ]]; then export CLOVER_STAGE1_RESUME="$latest"; fi
  fi
  run_distributed deepspeed "$CLOVER_MAIN_PROCESS_PORT" \
    "$project_root/starVLA/training/train_register_clover_stage1.py" \
    --config "$stage1_config"
fi
stage1_complete "$stage1_marker" "$stage1_generator" "$stage1_scorer"

current_generator="$stage1_generator"
current_scorer="$stage1_scorer"
current_scorer_stage=clover_stage1_scorer
selection_args=()

for cycle in $(seq 1 "$CLOVER_NUM_CYCLES"); do
  stem="cycle_$(printf '%02d' "$cycle")"
  cycle_root="$cycles_root/$stem"
  cycle_bank="$cycle_root/candidate_bank"
  cycle_scorer_dir="$cycle_root/scorer"
  cycle_generator_dir="$cycle_root/generator"
  port=$((CLOVER_MAIN_PROCESS_PORT + cycle * 3))

  clover_phase="$stem-bank"
  build_bank "$cycle_bank" "$current_generator" "$port"

  clover_phase="$stem-critic"
  export VLA_OUTPUT_ROOT="$cycle_root"
  export CLOVER_SCORER_RUN_ID=scorer
  export REGISTER64_BANK_ROOT="$cycle_bank"
  export CLOVER_SCORER_INITIALIZE_FROM="$current_scorer"
  export CLOVER_SCORER_INITIALIZE_STAGE="$current_scorer_stage"
  cycle_scorer="$cycle_scorer_dir/best_regret.pt"
  cycle_scorer_marker="$cycle_scorer_dir/training_complete.json"
  if component_complete "$cycle_scorer_marker" "$cycle_scorer" drivor_scorer; then
    echo "[clover-pdms] $stem critic already complete"
  else
    run_distributed plain $((port + 2)) \
      "$project_root/starVLA/training/train_register_drivor.py" --config "$scorer_config"
  fi
  component_complete "$cycle_scorer_marker" "$cycle_scorer" drivor_scorer
  selection_args+=(
    --candidate "$stem" "$current_generator" "$cycle_scorer" \
    "$cycle_bank/train" "$cycle_bank/selection"
  )

  clover_phase="$stem-generator"
  export CLOVER_GENERATOR_RUN_ID=generator
  export REGISTER64_GENERATOR_CHECKPOINT="$current_generator"
  export REGISTER64_DRIVOR_CHECKPOINT="$cycle_scorer"
  cycle_generator="$cycle_generator_dir/refined_generator.pt"
  cycle_generator_marker="$cycle_generator_dir/training_complete.json"
  if component_complete "$cycle_generator_marker" "$cycle_generator" clover_stage2_generator; then
    echo "[clover-pdms] $stem generator already complete"
  else
    run_distributed plain $((port + 3)) \
      "$project_root/starVLA/training/train_register_clover_refinement.py" --config "$refinement_config"
  fi
  component_complete "$cycle_generator_marker" "$cycle_generator" clover_stage2_generator
  current_generator="$cycle_generator"
  current_scorer="$cycle_scorer"
  current_scorer_stage=drivor_scorer
done

# The last generator update changes the proposal distribution. Fit a final
# critic on that exact distribution so inference never uses a stale scorer.
clover_phase=closing-critic-bank
closing_root="$cycles_root/closing_critic"
closing_bank="$closing_root/candidate_bank"
closing_scorer_dir="$closing_root/scorer"
closing_port=$((CLOVER_MAIN_PROCESS_PORT + CLOVER_NUM_CYCLES * 3 + 4))
build_bank "$closing_bank" "$current_generator" "$closing_port"

clover_phase=closing-critic
export VLA_OUTPUT_ROOT="$closing_root"
export CLOVER_SCORER_RUN_ID=scorer
export REGISTER64_BANK_ROOT="$closing_bank"
export CLOVER_SCORER_INITIALIZE_FROM="$current_scorer"
export CLOVER_SCORER_INITIALIZE_STAGE=drivor_scorer
final_scorer="$closing_scorer_dir/best_regret.pt"
final_scorer_marker="$closing_scorer_dir/training_complete.json"
if component_complete "$final_scorer_marker" "$final_scorer" drivor_scorer; then
  echo "[clover-pdms] closing critic already complete"
else
  run_distributed plain $((closing_port + 2)) \
    "$project_root/starVLA/training/train_register_drivor.py" --config "$scorer_config"
fi
component_complete "$final_scorer_marker" "$final_scorer" drivor_scorer
selection_args+=(
  --candidate closing_critic "$current_generator" "$final_scorer" \
  "$closing_bank/train" "$closing_bank/selection"
)

# A conservative update can still regress despite the pre-update enrichment
# gate. Select the best generator with the critic fitted on its exact proposal
# distribution instead of assuming the last chronological cycle is best.
clover_phase=model-selection
model_selection="$run_root/model_selection.json"
python "$project_root/tools/select_register64_clover_checkpoint_pair.py" \
  "${selection_args[@]}" --output "$model_selection"
final_generator="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["generator_checkpoint"])' "$model_selection")"
final_scorer="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["scorer_checkpoint"])' "$model_selection")"
final_bank="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["train_bank_root"])' "$model_selection")"
final_bank="$(dirname -- "$final_bank")"

clover_phase=navtest-prediction
export REGISTER64_GENERATOR_CHECKPOINT="$final_generator"
export REGISTER64_DRIVOR_CHECKPOINT="$final_scorer"
prediction_args=(
  --config "$inference_config" --datalist "$CLOVER_NAVTEST_DATALIST"
  --data-root "$DATA_ROOT" --output-dir "$prediction_dir" --split test
  --batch-size "$CLOVER_INFER_BATCH_SIZE" --num-workers "$CLOVER_INFER_WORKERS"
  --generator-checkpoint "$final_generator" --drivor-checkpoint "$final_scorer"
)
if [[ -f "$prediction_dir/prediction_identity.json" ]]; then prediction_args+=(--resume); fi
run_distributed plain $((closing_port + 3)) \
  "$project_root/starVLA/training/export_register_navtest_predictions.py" "${prediction_args[@]}"
expected_navtest="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$CLOVER_NAVTEST_DATALIST")"

clover_phase=cache-navtest-v1-1
ensure_v1_cache navtest "$CLOVER_PDMS_NAVTEST_METRIC_CACHE" \
  "$CLOVER_NAVTEST_DATALIST" "$CLOVER_NAVTEST_LOG_ROOT"

clover_phase=official-navtest-pdms
if python "$project_root/tools/validate_navsim_score_csv.py" \
  --results-dir "$pdms_results" --protocol pdms \
  --expected-scenarios "$expected_navtest" \
  --expected-datalist "$CLOVER_NAVTEST_DATALIST" >/dev/null 2>&1; then
  echo "[clover-pdms] official PDMS already valid"
else
  mkdir -p "$pdms_results"
  env CUDA_VISIBLE_DEVICES= \
    PYTHONPATH="$project_root/navsim_v1.1/navsim:$project_root:${PYTHONPATH:-}" \
    NAVSIM_DEVKIT_ROOT="$project_root/navsim_v1.1/navsim" \
    NAVSIM_EXP_ROOT="$pdms_results" OPENSCENE_DATA_ROOT="$OPENSCENE_DATA_ROOT" \
    python "$project_root/navsim_v1.1/navsim/navsim/planning/script/run_pdm_score.py" \
      train_test_split=navtest "metric_cache_path=$CLOVER_PDMS_NAVTEST_METRIC_CACHE" \
      agent=human_agent experiment_name=register64-clover-navtest-pdms \
      "output_dir=$pdms_results" "pred_dir=$prediction_root" split=test \
      worker=ray_distributed_no_torch "worker.threads_per_node=$CLOVER_CACHE_WORKERS" \
      worker.use_distributed=false gpu=false
fi
python "$project_root/tools/validate_navsim_score_csv.py" \
  --results-dir "$pdms_results" --protocol pdms \
  --expected-scenarios "$expected_navtest" \
  --expected-datalist "$CLOVER_NAVTEST_DATALIST"

clover_phase=summary
python "$project_root/tools/collect_register64_clover_results.py" \
  --run-root "$run_root" --stage1-dir "$stage1_dir" \
  --cycles-root "$cycles_root" --num-cycles "$CLOVER_NUM_CYCLES" \
  --generator-checkpoint "$final_generator" --drivor-checkpoint "$final_scorer" \
  --bank-root "$final_bank" --model-selection "$model_selection" \
  --prediction-dir "$prediction_dir" \
  --pdms-results-dir "$pdms_results" --navtest-datalist "$CLOVER_NAVTEST_DATALIST" \
  --expected-scenarios "$expected_navtest" --output-dir "$summary_root"

echo "[clover-pdms] complete cycles=$CLOVER_NUM_CYCLES"
echo "[clover-pdms] final_generator=$final_generator"
echo "[clover-pdms] final_scorer=$final_scorer"
echo "[clover-pdms] model_selection=$model_selection"
echo "[clover-pdms] official_result=$summary_root/summary.md"
