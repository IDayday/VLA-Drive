#!/usr/bin/env bash
# Complete non-interactive DLC pipeline:
# Stage G -> v2 metric cache -> Stage B train/val banks -> Stage S -> optional
# Stage SD -> navtest export -> official v1.1 PDMS and v2 EPDMS -> summary.

set -Eeuo pipefail

dry_run=0
preflight_only=0
resume="${REGISTER64_RESUME:-0}"
while (( $# )); do
  case "$1" in
    --dry-run) dry_run=1 ;;
    --preflight-only) preflight_only=1 ;;
    --resume) resume=1 ;;
    --help|-h)
      echo "Usage: $0 [--dry-run] [--preflight-only] [--resume]"
      exit 0
      ;;
    *) echo "[register64] unsupported argument: $1" >&2; exit 2 ;;
  esac
  shift
done
export DRY_RUN="$dry_run"
export PREFLIGHT_ONLY="$preflight_only"

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fixed_arm="${REGISTER64_ARM:?Use a fixed Register64 arm wrapper}"
fixed_suprim="${REGISTER64_ENABLE_SUPRIM:?Use a fixed Register64 arm wrapper}"
fixed_generator_variant="${REGISTER64_GENERATOR_VARIANT:?Use a fixed Register64 arm wrapper}"
source "$project_root/load_env.sh"
export DRIVEDREAMER_ROOT="$project_root"
export REGISTER64_ARM="$fixed_arm"
export REGISTER64_ENABLE_SUPRIM="$fixed_suprim"
export REGISTER64_GENERATOR_VARIANT="$fixed_generator_variant"
cd "$project_root"

case "$REGISTER64_ARM:$REGISTER64_ENABLE_SUPRIM:$REGISTER64_GENERATOR_VARIANT" in
  off:0:frozen|on:1:frozen|off:0:visual_unfrozen) ;;
  *) echo "[register64] invalid fixed arm contract: $REGISTER64_ARM/$REGISTER64_ENABLE_SUPRIM/$REGISTER64_GENERATOR_VARIANT" >&2; exit 2 ;;
esac
case "$resume:$dry_run:$preflight_only" in
  [01]:[01]:[01]) ;;
  *) echo "[register64] resume/dry-run/preflight flags must be 0 or 1" >&2; exit 2 ;;
esac

export REGISTER64_EXPECTED_BRANCH="${REGISTER64_EXPECTED_BRANCH:-feature/ddp-drs-scene-2048}"
export NUM_MACHINES="${NUM_MACHINES:-1}"
export MACHINE_RANK="${MACHINE_RANK:-0}"
export LOCAL_NUM_PROCESSES="${LOCAL_NUM_PROCESSES:-16}"
export NUM_PROCESSES="${NUM_PROCESSES:-$((NUM_MACHINES * LOCAL_NUM_PROCESSES))}"
export REGISTER64_MAIN_PROCESS_PORT="${REGISTER64_MAIN_PROCESS_PORT:-29741}"
export REGISTER64_CACHE_WORKERS="${REGISTER64_CACHE_WORKERS:-96}"
export REGISTER64_BANK_LOADER_WORKERS="${REGISTER64_BANK_LOADER_WORKERS:-7}"
export NAVSIM_NUM_WORKERS="${NAVSIM_NUM_WORKERS:-3}"
export NAVSIM_METRIC_WORKERS="${NAVSIM_METRIC_WORKERS:-4}"
export REGISTER64_INFER_BATCH_SIZE="${REGISTER64_INFER_BATCH_SIZE:-4}"
export REGISTER64_INFER_WORKERS="${REGISTER64_INFER_WORKERS:-3}"
export REGISTER64_VALIDATION_SIZE="${REGISTER64_VALIDATION_SIZE:-4096}"
export REGISTER64_BUILD_CACHES="${REGISTER64_BUILD_CACHES:-${AUTO_GENERATE_CACHES:-auto}}"
export AUTO_GENERATE_CACHES="$REGISTER64_BUILD_CACHES"
export REGISTER64_ALLOW_DIRTY="${REGISTER64_ALLOW_DIRTY:-0}"
export REGISTER64_RUN_ID="${REGISTER64_RUN_ID:-register64-${REGISTER64_ARM}-$(date +'%Y%m%d_%H%M%S')}"
export REGISTER64_OUTPUT_ROOT="${REGISTER64_OUTPUT_ROOT:-${NAVSIM_EXP_ROOT:-$project_root/navsim_exp}/register64_complete_pipeline}"
export REGISTER64_DRIVOR_EPOCHS="${REGISTER64_DRIVOR_EPOCHS:-10}"

# Stage-G identity. Bank-only stages retain their independent global batch 256.
if [[ "$REGISTER64_GENERATOR_VARIANT" == visual_unfrozen ]]; then
  export TARGET_EFFECTIVE_BATCH_SIZE=32
  generator_stage_id=qwen_register64_generator_visual_unfrozen
  unset REGISTER64_TRAIN_FEATURE_CACHE_ROOT
  unset NAVSIM_FEATURE_CACHE_ROOT
  unset NAVSIM_AGENT_DINO_CACHE_ROOT
  unset NAVSIM_VGGT_CACHE_ROOT
  export NAVSIM_USE_FEATURE_CACHE=0
else
  export TARGET_EFFECTIVE_BATCH_SIZE=64
  generator_stage_id=qwen_register64_generator
fi
export GRADIENT_ACCUMULATION_STEPS=1
export MAX_TRAIN_STEPS=0  # epoch-bounded: at most 25 Stage-G epochs
export SAVE_INTERVAL=5    # component checkpoint interval in epochs
export RUN_ID="$REGISTER64_RUN_ID"

export QWEN_VLM_PATH="${QWEN_VLM_PATH:-${BASE_VLM:-}}"
export VLM_ATTN_IMPLEMENTATION="${VLM_ATTN_IMPLEMENTATION:-sdpa}"
export DATA_ROOT="${DATA_ROOT:-$project_root/navsim_dataset}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$project_root/navsim_dataset_raw}"
source_datalist_default="${NAVSIM_DATALIST_PATH:-$project_root/train_meta.json}"
export REGISTER64_SOURCE_DATALIST="${REGISTER64_SOURCE_DATALIST:-$source_datalist_default}"
datalist_parent="$(cd -- "$(dirname -- "$REGISTER64_SOURCE_DATALIST")" 2>/dev/null && pwd || dirname -- "$REGISTER64_SOURCE_DATALIST")"
export REGISTER64_NAVTEST_DATALIST="${REGISTER64_NAVTEST_DATALIST:-$datalist_parent/test_meta.json}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$OPENSCENE_DATA_ROOT/maps}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export REGISTER64_NAVTRAIN_LOG_ROOT="${REGISTER64_NAVTRAIN_LOG_ROOT:-${QDS_NAVSIM_LOG_PATH:-$OPENSCENE_DATA_ROOT/navsim_logs/trainval}}"
export REGISTER64_NAVTRAIN_SENSOR_ROOT="${REGISTER64_NAVTRAIN_SENSOR_ROOT:-${QDS_NAVSIM_SENSOR_PATH:-${NAVSIM_TRAINVAL_SENSOR_ROOT:-$OPENSCENE_DATA_ROOT/sensor_blobs/trainval}}}"
export REGISTER64_NAVTEST_LOG_ROOT="${REGISTER64_NAVTEST_LOG_ROOT:-$OPENSCENE_DATA_ROOT/navsim_logs/test}"
export REGISTER64_NAVTEST_SENSOR_ROOT="${REGISTER64_NAVTEST_SENSOR_ROOT:-$OPENSCENE_DATA_ROOT/sensor_blobs/test}"
export NAVSIM_TRAINVAL_SENSOR_ROOT="$REGISTER64_NAVTRAIN_SENSOR_ROOT"

run_root="$REGISTER64_OUTPUT_ROOT/$REGISTER64_RUN_ID"
stages_root="$run_root/stages"
split_root="$run_root/splits"
bank_root="$run_root/candidate_bank"
prediction_root="$run_root/predictions"
prediction_dir="$prediction_root/test"
cache_root="$run_root/caches"
train_metric_cache="${REGISTER64_TRAIN_METRIC_CACHE_ROOT:-$cache_root/navtrain_v2}"
pdms_metric_cache="${REGISTER64_PDMS_METRIC_CACHE_ROOT:-$cache_root/navtest_v1_1}"
epdms_metric_cache="${REGISTER64_EPDMS_METRIC_CACHE_ROOT:-$cache_root/navtest_v2}"
export PDMS_METRIC_CACHE_PATH="$pdms_metric_cache"
export EPDMS_METRIC_CACHE_PATH="$epdms_metric_cache"
pdms_results="$run_root/evaluation/pdms_v1_1"
epdms_results="$run_root/evaluation/epdms_v2"
summary_root="$run_root/summary"

generator_config="$project_root/starVLA/config/training/qwen_register64_generator.yaml"
bank_config="$project_root/starVLA/config/training/register64_candidate_bank.yaml"
drivor_config="$project_root/starVLA/config/training/register64_drivor_scorer.yaml"
suprim_config="$project_root/starVLA/config/training/register64_drivor_suprim_dynamic.yaml"
off_inference_config="$project_root/starVLA/config/training/register64_inference.yaml"
on_inference_config="$project_root/starVLA/config/training/register64_suprim_dynamic_inference.yaml"
deepspeed_config="${REGISTER64_DEEPSPEED_CONFIG:-$project_root/starVLA/config/deepseeds/deepspeed_zero1.yaml}"
if [[ "$REGISTER64_GENERATOR_VARIANT" == visual_unfrozen ]]; then
  generator_config="$project_root/starVLA/config/training/qwen_register64_generator_visual_unfrozen.yaml"
  bank_config="$project_root/starVLA/config/training/register64_candidate_bank_visual_unfrozen.yaml"
  off_inference_config="$project_root/starVLA/config/training/register64_inference_visual_unfrozen.yaml"
fi
selected_inference_config="$off_inference_config"
if [[ "$REGISTER64_ENABLE_SUPRIM" == 1 ]]; then
  selected_inference_config="$on_inference_config"
fi
generator_checkpoint="$stages_root/$generator_stage_id/best_minade_generator.pt"
drivor_checkpoint="$stages_root/register64_drivor_scorer/best_regret.pt"
suprim_checkpoint="$stages_root/register64_drivor_suprim_dynamic/best_regret.pt"
train_datalist="$split_root/train.json"
val_datalist="$split_root/val.json"

print_command() {
  printf '[register64] command:'
  printf ' %q' "$@"
  printf '\n'
}

if (( dry_run )); then
  echo "[register64] dry_run=1 writes=0 imports=0"
  echo "[register64] project_root=$project_root"
  echo "[register64] arm=$REGISTER64_ARM drivesuprim=$REGISTER64_ENABLE_SUPRIM generator_variant=$REGISTER64_GENERATOR_VARIANT run_root=$run_root"
  echo "[register64] topology=num_machines:$NUM_MACHINES machine_rank:$MACHINE_RANK local:$LOCAL_NUM_PROCESSES total:$NUM_PROCESSES"
  echo "[register64] schedule=generator:max25epochs drivor:${REGISTER64_DRIVOR_EPOCHS}epochs suprim:3epochs"
  echo "[register64] batch=stage_g:$TARGET_EFFECTIVE_BATCH_SIZE(per_device:$((TARGET_EFFECTIVE_BATCH_SIZE / NUM_PROCESSES))) scorer:256"
  echo "[register64] generator_config=$generator_config"
  if [[ "$REGISTER64_GENERATOR_VARIANT" == visual_unfrozen ]]; then
    echo "[register64] qwen_visual=trainable lr=2e-6 gradient_checkpointing=1 feature_cache=disabled"
  else
    echo "[register64] qwen_visual=frozen feature_cache=optional"
  fi
  echo "[register64] stages=split,G,cache-navtrain-v2,B-train,B-val,S$([[ "$REGISTER64_ENABLE_SUPRIM" == 1 ]] && printf ',SD'),predict,cache-navtest-v1.1,cache-navtest-v2,PDMS,EPDMS,summary"
  print_command python "$project_root/tools/prepare_register64_train_val_split.py" --source "$REGISTER64_SOURCE_DATALIST" --output-dir "$split_root" --validation-size "$REGISTER64_VALIDATION_SIZE" --seed 42
  print_command accelerate launch --config_file "$deepspeed_config" --num_machines 1 --num_processes "$NUM_PROCESSES" --main_process_port "$REGISTER64_MAIN_PROCESS_PORT" "$project_root/starVLA/training/train_register_generator.py" --config "$generator_config"
  print_command python "$project_root/navsim/navsim/planning/script/run_metric_caching.py" train_test_split=navtrain "metric_cache_path=$train_metric_cache" "worker.threads_per_node=$REGISTER64_CACHE_WORKERS"
  print_command accelerate launch --multi_gpu --num_machines 1 --num_processes "$NUM_PROCESSES" "$project_root/starVLA/training/build_register_candidate_bank.py" --config "$bank_config" --split train
  print_command accelerate launch --multi_gpu --num_machines 1 --num_processes "$NUM_PROCESSES" "$project_root/starVLA/training/build_register_candidate_bank.py" --config "$bank_config" --split val
  print_command accelerate launch --multi_gpu --num_machines 1 --num_processes "$NUM_PROCESSES" "$project_root/starVLA/training/train_register_drivor.py" --config "$drivor_config"
  if [[ "$REGISTER64_ENABLE_SUPRIM" == 1 ]]; then
    print_command accelerate launch --multi_gpu --num_machines 1 --num_processes "$NUM_PROCESSES" "$project_root/starVLA/training/train_register_suprim.py" --config "$suprim_config"
  fi
  print_command accelerate launch --multi_gpu --num_machines 1 --num_processes "$NUM_PROCESSES" "$project_root/starVLA/training/export_register_navtest_predictions.py" --config "$selected_inference_config" --datalist "$REGISTER64_NAVTEST_DATALIST" --data-root "$DATA_ROOT" --output-dir "$prediction_dir" --split test
  print_command python "$project_root/navsim_v1.1/navsim/navsim/planning/script/run_metric_caching.py" train_test_split=navtest "cache.cache_path=$PDMS_METRIC_CACHE_PATH" "worker.threads_per_node=$REGISTER64_CACHE_WORKERS"
  print_command python "$project_root/navsim/navsim/planning/script/run_metric_caching.py" train_test_split=navtest "metric_cache_path=$EPDMS_METRIC_CACHE_PATH" "worker.threads_per_node=$REGISTER64_CACHE_WORKERS"
  print_command python "$project_root/navsim_v1.1/navsim/navsim/planning/script/run_pdm_score.py" train_test_split=navtest "metric_cache_path=$PDMS_METRIC_CACHE_PATH" agent=human_agent "pred_dir=$prediction_root" split=test
  print_command python "$project_root/navsim/navsim/planning/script/run_pdm_score_one_stage.py" train_test_split=navtest "metric_cache_path=$EPDMS_METRIC_CACHE_PATH" agent=human_agent "pred_dir=$prediction_root" split=test
  echo "[register64] official_eval=v1.1:average v2:average_all_frames"
  echo "[register64] formal_training=NOT_RUN"
  exit 0
fi

mkdir -p "$REGISTER64_OUTPUT_ROOT/launcher_logs"
launcher_log="$REGISTER64_OUTPUT_ROOT/launcher_logs/${REGISTER64_RUN_ID}.log"
exec > >(tee -a "$launcher_log") 2>&1
register64_phase=bootstrap
on_register64_error() {
  local status="$?"
  if (( BASH_SUBSHELL > 0 )); then return "$status"; fi
  echo "[register64] failed phase=$register64_phase line=${BASH_LINENO[0]} status=$status" >&2
  exit "$status"
}
trap on_register64_error ERR

if (( ! preflight_only )) && [[ -e "$run_root" && "$resume" != 1 ]]; then
  echo "[register64] Refusing to overwrite existing run: $run_root; use a new REGISTER64_RUN_ID or --resume" >&2
  exit 2
fi

required_value() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then echo "[register64] required variable is empty: $name" >&2; exit 2; fi
}
required_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then echo "[register64] missing required path: $path" >&2; exit 2; fi
}
stage_complete() {
  local marker="$1" checkpoint="$2" expected_stage="$3"
  python - "$marker" "$checkpoint" "$expected_stage" <<'PY' >/dev/null 2>&1
import hashlib
import json
import sys
from pathlib import Path

marker = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
expected_stage = sys.argv[3]
if not marker.is_file() or not checkpoint.is_file():
    raise SystemExit(1)
payload = json.loads(marker.read_text(encoding="utf-8"))
if payload.get("status") != "complete" or payload.get("stage") != expected_stage:
    raise SystemExit(1)
digest = hashlib.sha256()
with checkpoint.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 << 20), b""):
        digest.update(chunk)
if payload.get("selected_checkpoint") != checkpoint.name:
    raise SystemExit(1)
if payload.get("selected_checkpoint_sha256") != digest.hexdigest():
    raise SystemExit(1)
PY
}
require_uint() {
  local name="$1"
  if ! [[ "${!name}" =~ ^[0-9]+$ ]]; then echo "[register64] $name must be an integer: ${!name}" >&2; exit 2; fi
}

register64_phase=source-contract
for name in NUM_MACHINES MACHINE_RANK LOCAL_NUM_PROCESSES NUM_PROCESSES \
  REGISTER64_MAIN_PROCESS_PORT REGISTER64_CACHE_WORKERS \
  REGISTER64_BANK_LOADER_WORKERS NAVSIM_NUM_WORKERS NAVSIM_METRIC_WORKERS \
  REGISTER64_INFER_BATCH_SIZE REGISTER64_INFER_WORKERS REGISTER64_VALIDATION_SIZE \
  REGISTER64_DRIVOR_EPOCHS TARGET_EFFECTIVE_BATCH_SIZE \
  GRADIENT_ACCUMULATION_STEPS MAX_TRAIN_STEPS SAVE_INTERVAL; do
  require_uint "$name"
done
if (( NUM_MACHINES != 1 || MACHINE_RANK != 0 )); then
  echo "[register64] complete cache/train/eval pipeline currently requires one 16-device DLC node" >&2
  exit 2
fi
if (( NUM_PROCESSES != LOCAL_NUM_PROCESSES || NUM_PROCESSES < 1 )); then
  echo "[register64] NUM_PROCESSES must equal LOCAL_NUM_PROCESSES on the single DLC node" >&2
  exit 2
fi
if (( TARGET_EFFECTIVE_BATCH_SIZE % NUM_PROCESSES != 0 || 256 % NUM_PROCESSES != 0 )); then
  echo "[register64] world size must divide Stage-G batch $TARGET_EFFECTIVE_BATCH_SIZE and bank-stage batch 256" >&2
  exit 2
fi
export PER_DEVICE_BATCH_SIZE=$((TARGET_EFFECTIVE_BATCH_SIZE / NUM_PROCESSES))
effective_batch=$((NUM_PROCESSES * PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
if (( effective_batch != TARGET_EFFECTIVE_BATCH_SIZE )); then
  echo "[register64] Refusing effective batch $effective_batch; expected $TARGET_EFFECTIVE_BATCH_SIZE" >&2
  exit 2
fi
if (( REGISTER64_DRIVOR_EPOCHS < 1 || REGISTER64_DRIVOR_EPOCHS > 10 )); then
  echo "[register64] REGISTER64_DRIVOR_EPOCHS must be in [1,10]" >&2
  exit 2
fi
if (( REGISTER64_VALIDATION_SIZE < 1024 )); then
  echo "[register64] validation holdout must contain at least the fixed 1024-scene Stage-G subset" >&2
  exit 2
fi
case "$REGISTER64_BUILD_CACHES" in auto|0|1) ;; *) echo "[register64] REGISTER64_BUILD_CACHES must be auto, 0, or 1" >&2; exit 2 ;; esac
if ! [[ "$REGISTER64_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[register64] REGISTER64_RUN_ID contains unsupported characters" >&2
  exit 2
fi

actual_branch="$(env -u GIT_DIR -u GIT_WORK_TREE git -C "$project_root" branch --show-current)"
source_commit="$(env -u GIT_DIR -u GIT_WORK_TREE git -C "$project_root" rev-parse HEAD)"
if [[ "$actual_branch" != "$REGISTER64_EXPECTED_BRANCH" ]]; then
  echo "[register64] wrong branch: expected=$REGISTER64_EXPECTED_BRANCH actual=${actual_branch:-DETACHED}" >&2
  exit 2
fi
if [[ "$REGISTER64_ALLOW_DIRTY" != 1 ]] && [[ -n "$(env -u GIT_DIR -u GIT_WORK_TREE git -C "$project_root" status --short)" ]]; then
  echo "[register64] formal DLC source is dirty; commit/push the intended source or set REGISTER64_ALLOW_DIRTY=1" >&2
  exit 2
fi
for name in QWEN_VLM_PATH DATA_ROOT OPENSCENE_DATA_ROOT REGISTER64_SOURCE_DATALIST \
  REGISTER64_NAVTEST_DATALIST NUPLAN_MAPS_ROOT REGISTER64_NAVTRAIN_LOG_ROOT \
  REGISTER64_NAVTRAIN_SENSOR_ROOT REGISTER64_NAVTEST_LOG_ROOT REGISTER64_NAVTEST_SENSOR_ROOT; do
  required_value "$name"
done
for path in "$generator_config" "$bank_config" "$drivor_config" "$suprim_config" \
  "$selected_inference_config" "$deepspeed_config" \
  "$QWEN_VLM_PATH/config.json" "$DATA_ROOT/meta/train" "$DATA_ROOT/meta/test" \
  "$REGISTER64_SOURCE_DATALIST" "$REGISTER64_NAVTEST_DATALIST" "$NUPLAN_MAPS_ROOT" \
  "$REGISTER64_NAVTRAIN_LOG_ROOT" "$REGISTER64_NAVTRAIN_SENSOR_ROOT" \
  "$REGISTER64_NAVTEST_LOG_ROOT" "$REGISTER64_NAVTEST_SENSOR_ROOT" \
  "$project_root/navsim" "$project_root/navsim_v1.1/navsim"; do
  required_path "$path"
done
if ! compgen -G "$QWEN_VLM_PATH/*.safetensors" >/dev/null; then
  echo "[register64] Qwen safetensors are missing under $QWEN_VLM_PATH" >&2
  exit 2
fi

export PYTHONPATH="$project_root:$project_root/navsim:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export NAVSIM_PREFETCH_FACTOR="${NAVSIM_PREFETCH_FACTOR:-2}"
export NAVSIM_PIN_MEMORY="${NAVSIM_PIN_MEMORY:-1}"
export NAVSIM_WORKER_THREADS="${NAVSIM_WORKER_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export VLA_OUTPUT_ROOT="$stages_root"
export REGISTER64_BANK_ROOT="$bank_root"

register64_phase=input-preflight
python - "$REGISTER64_SOURCE_DATALIST" "$REGISTER64_NAVTEST_DATALIST" "$DATA_ROOT" <<'PY'
import json
import pickle
import sys
from pathlib import Path

from starVLA.dataloader.navsim_dataset import resolve_navsim_data_path

for datalist_name, physical_split in ((sys.argv[1], "train"), (sys.argv[2], "test")):
    datalist = Path(datalist_name)
    tokens = json.loads(datalist.read_text(encoding="utf-8"))
    if not tokens or len(tokens) != len(set(tokens)) or not all(isinstance(t, str) for t in tokens):
        raise ValueError(f"invalid NAVSIM datalist: {datalist}")
    record_path = Path(sys.argv[3]) / "meta" / physical_split / f"{tokens[0]}.pkl"
    with record_path.open("rb") as stream:
        record = pickle.load(stream)
    for view in ("cam_l0", "cam_f0", "cam_r0"):
        embedded = record["glo_images"][view]["image_paths"][3]
        resolved = Path(resolve_navsim_data_path(embedded))
        if not resolved.is_file():
            raise FileNotFoundError(f"image relocation failed: {embedded} -> {resolved}")
    print(f"[register64] {physical_split} input preflight token={tokens[0]} count={len(tokens)}")
PY

python - "$generator_config" "$bank_config" "$drivor_config" "$suprim_config" \
  "$selected_inference_config" "$REGISTER64_GENERATOR_VARIANT" <<'PY'
import sys
from starVLA.training.config_loader import load_training_config

generator, bank, drivor, suprim, inference = map(load_training_config, sys.argv[1:6])
variant = sys.argv[6]
assert generator.framework.name == "QwenRegisterGenerator"
assert generator.framework.register_generator.proposal_num == 64
assert generator.framework.register_generator.num_layers == 4
assert generator.framework.register_generator.num_heads == 1
assert generator.validation.dataset_split == "train"
assert "pdm_oracle" not in generator.validation
assert bank.candidate_bank.splits.val.dataset_split == "train"
assert drivor.training_profile.name == "drivor_offline_bank_v1"
assert int(drivor.trainer.epochs) == int(__import__("os").environ["REGISTER64_DRIVOR_EPOCHS"])
assert drivor.trainer.max_epochs == 10
assert suprim.training_profile.name == "drivesuprim_dynamic_bank_v1"
assert suprim.trainer.epochs == 3 and suprim.trainer.max_epochs == 5
freeze_sets = []
for config in (generator, bank, inference):
    freeze_sets.append({
        value.strip()
        for value in str(config.trainer.freeze_modules).split(",")
        if value.strip()
    })
if variant == "visual_unfrozen":
    for config, frozen in zip((generator, bank, inference), freeze_sets):
        assert config.framework.qwenvl.freeze_visual is False
        assert config.framework.qwenvl.visual_gradient_checkpointing is True
        assert "qwen_vl_interface.model.visual" not in frozen
        assert "qwen_vl_interface.model.lm_head" in frozen
        assert float(config.optimizer.learning_rates.qwen_visual) == 2.0e-6
    assert int(generator.trainer.global_batch_size) == 32
else:
    assert all("qwen_vl_interface.model.visual" in frozen for frozen in freeze_sets)
    assert int(generator.trainer.global_batch_size) == 64
print("[register64] config contract passed")
PY

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((LOCAL_NUM_PROCESSES - 1)))"
fi
if (( NUM_PROCESSES == 1 )); then
  python "$project_root/tools/check_ppu_runtime.py" --expected-world-size 1
else
  accelerate launch --multi_gpu --mixed_precision bf16 \
    --num_machines 1 --num_processes "$NUM_PROCESSES" \
    --machine_rank 0 --main_process_port $((REGISTER64_MAIN_PROCESS_PORT + 20)) \
    --num_cpu_threads_per_process 1 \
    "$project_root/tools/check_ppu_runtime.py" \
    --expected-world-size "$NUM_PROCESSES"
fi
available_cpus="$(nproc)"
nominal_cpu_slots=$((LOCAL_NUM_PROCESSES * (1 + REGISTER64_BANK_LOADER_WORKERS)))
if (( available_cpus < nominal_cpu_slots )); then
  echo "[register64] warning: available_cpus=$available_cpus nominal_bank_slots=$nominal_cpu_slots" >&2
fi

echo "[register64] source=$actual_branch@$source_commit"
echo "[register64] arm=$REGISTER64_ARM drivesuprim=$REGISTER64_ENABLE_SUPRIM generator_variant=$REGISTER64_GENERATOR_VARIANT"
echo "[register64] topology=machines:$NUM_MACHINES machine_rank:$MACHINE_RANK local_processes:$LOCAL_NUM_PROCESSES total_processes:$NUM_PROCESSES"
echo "[register64] batch=stage_g:$effective_batch(per_device:$PER_DEVICE_BATCH_SIZE) scorer:256(per_device:$((256 / NUM_PROCESSES))) suprim:256(per_device:$((256 / NUM_PROCESSES)))"
echo "[register64] cpu=available:$available_cpus stage_g_workers:$((NUM_PROCESSES * NAVSIM_NUM_WORKERS)) bank_loader_workers:$((NUM_PROCESSES * REGISTER64_BANK_LOADER_WORKERS)) bank_metric_workers:$((NUM_PROCESSES * NAVSIM_METRIC_WORKERS))"
echo "[register64] outputs=$run_root launcher_log=$launcher_log"
echo "[register64] scorer_target=NAVSIM-v2 shared_checkpoint_for_PDMS_and_EPDMS"
if (( preflight_only )); then
  echo "[register64] full preflight passed; formal_training=NOT_RUN"
  exit 0
fi

mkdir -p "$run_root" "$stages_root" "$split_root" "$cache_root" "$summary_root"
export TRITON_CACHE_DIR="${TRITON_LOCAL_CACHE_ROOT:-/tmp/register64-triton}/$REGISTER64_RUN_ID"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_LOCAL_ROOT:-/tmp/register64-extensions}/$REGISTER64_RUN_ID"
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

run_distributed() {
  local mode="$1" port="$2" script="$3"
  shift 3
  local command
  if (( NUM_PROCESSES == 1 )); then
    command=(python "$script")
  elif [[ "$mode" == deepspeed ]]; then
    command=(
      accelerate launch --config_file "$deepspeed_config"
      --num_machines 1 --num_processes "$NUM_PROCESSES"
      --machine_rank 0 --main_process_port "$port"
      --num_cpu_threads_per_process 1 "$script"
    )
  else
    command=(
      accelerate launch --multi_gpu --mixed_precision bf16
      --num_machines 1 --num_processes "$NUM_PROCESSES"
      --machine_rank 0 --main_process_port "$port"
      --num_cpu_threads_per_process 1 "$script"
    )
  fi
  print_command "${command[@]}" "$@"
  "${command[@]}" "$@"
}

ensure_metric_cache() {
  local version="$1" split="$2" root="$3" expected_datalist="$4"
  local validator=(python "$project_root/tools/validate_navsim_metric_cache.py" --cache-root "$root" --expected-datalist "$expected_datalist" --check-cache-files)
  if "${validator[@]}" >/dev/null 2>&1; then
    echo "[register64] metric cache valid: version=$version split=$split root=$root"
    return
  fi
  if [[ "$REGISTER64_BUILD_CACHES" == 0 ]]; then
    echo "[register64] metric cache missing/incomplete and generation disabled: $root" >&2
    return 2
  fi
  mkdir -p "$root"
  echo "[register64] building metric cache: version=$version split=$split workers=$REGISTER64_CACHE_WORKERS root=$root"
  if [[ "$version" == v1.1 ]]; then
    env \
      PYTHONPATH="$project_root/navsim_v1.1/navsim:$project_root:${PYTHONPATH:-}" \
      NAVSIM_DEVKIT_ROOT="$project_root/navsim_v1.1/navsim" \
      NAVSIM_EXP_ROOT="$root" \
      OPENSCENE_DATA_ROOT="$OPENSCENE_DATA_ROOT" \
      python "$project_root/navsim_v1.1/navsim/navsim/planning/script/run_metric_caching.py" \
        "train_test_split=$split" \
        "navsim_log_path=$REGISTER64_NAVTEST_LOG_ROOT" \
        "cache.cache_path=$root" \
        cache.force_feature_computation=false \
        worker=ray_distributed_no_torch \
        "worker.threads_per_node=$REGISTER64_CACHE_WORKERS" \
        worker.use_distributed=false gpu=false
  else
    local log_root sensor_root
    if [[ "$split" == navtrain ]]; then
      log_root="$REGISTER64_NAVTRAIN_LOG_ROOT"
      sensor_root="$REGISTER64_NAVTRAIN_SENSOR_ROOT"
    else
      log_root="$REGISTER64_NAVTEST_LOG_ROOT"
      sensor_root="$REGISTER64_NAVTEST_SENSOR_ROOT"
    fi
    env \
      PYTHONPATH="$project_root/navsim:$project_root:${PYTHONPATH:-}" \
      NAVSIM_DEVKIT_ROOT="$project_root/navsim" \
      NAVSIM_EXP_ROOT="$root" \
      OPENSCENE_DATA_ROOT="$OPENSCENE_DATA_ROOT" \
      python "$project_root/navsim/navsim/planning/script/run_metric_caching.py" \
        "train_test_split=$split" \
        "navsim_log_path=$log_root" \
        "original_sensor_path=$sensor_root" \
        "metric_cache_path=$root" \
        force_feature_computation=false \
        worker=ray_distributed_no_torch \
        "worker.threads_per_node=$REGISTER64_CACHE_WORKERS" \
        worker.use_distributed=false gpu=false
  fi
  "${validator[@]}"
}

register64_phase=prepare-splits
python "$project_root/tools/prepare_register64_train_val_split.py" \
  --source "$REGISTER64_SOURCE_DATALIST" --output-dir "$split_root" \
  --validation-size "$REGISTER64_VALIDATION_SIZE" --seed 42
export NAVSIM_DATALIST_PATH="$train_datalist"
export NAVSIM_VAL_DATALIST_PATH="$val_datalist"
unset NAVSIM_METRIC_CACHE_ROOT

if [[ -n "${REGISTER64_TRAIN_FEATURE_CACHE_ROOT:-}" ]]; then
  required_path "$REGISTER64_TRAIN_FEATURE_CACHE_ROOT"
  export NAVSIM_FEATURE_CACHE_ROOT="$REGISTER64_TRAIN_FEATURE_CACHE_ROOT"
  export NAVSIM_USE_FEATURE_CACHE=1
  echo "[register64] train_feature_cache=$NAVSIM_FEATURE_CACHE_ROOT"
else
  unset NAVSIM_FEATURE_CACHE_ROOT
  export NAVSIM_USE_FEATURE_CACHE=0
  echo "[register64] train_feature_cache=disabled"
fi

register64_phase=stage-g
generator_complete="$stages_root/$generator_stage_id/training_complete.json"
if stage_complete "$generator_complete" "$generator_checkpoint" register_generator; then
  echo "[register64] Stage G already complete: $generator_checkpoint"
else
  unset REGISTER64_GENERATOR_RESUME
  if (( resume )) && [[ -d "$stages_root/$generator_stage_id/checkpoints" ]]; then
    latest_resume="$(find "$stages_root/$generator_stage_id/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'epoch_*' -print | sort | tail -n 1)"
    if [[ -n "$latest_resume" ]]; then export REGISTER64_GENERATOR_RESUME="$latest_resume"; fi
  fi
  run_distributed deepspeed "$REGISTER64_MAIN_PROCESS_PORT" \
    "$project_root/starVLA/training/train_register_generator.py" --config "$generator_config"
fi
if ! stage_complete "$generator_complete" "$generator_checkpoint" register_generator; then
  echo "[register64] Stage G did not produce a valid completion contract" >&2
  exit 2
fi
export REGISTER64_GENERATOR_CHECKPOINT="$generator_checkpoint"

# Stage G is deliberately independent of NAVSIM metric caches. Only after the
# geometry-selected generator is frozen do we prepare labels for Stage B.
register64_phase=cache-navtrain-v2
ensure_metric_cache v2 navtrain "$train_metric_cache" "$REGISTER64_SOURCE_DATALIST"
export NAVSIM_METRIC_CACHE_ROOT="$train_metric_cache"

build_bank_split() {
  local logical_split="$1" port="$2" split_dir="$bank_root/$logical_split"
  register64_phase="stage-b-$logical_split"
  if python "$project_root/starVLA/training/build_register_candidate_bank.py" --config "$bank_config" --split "$logical_split" --validate-only >/dev/null 2>&1; then
    echo "[register64] Stage B $logical_split bank already complete: $split_dir"
    return
  fi
  local bank_args=(--config "$bank_config" --split "$logical_split" --workers-per-rank "$NAVSIM_METRIC_WORKERS" --backend process)
  if (( resume )) && [[ -e "$split_dir/build_identity.json" ]]; then bank_args+=(--resume); fi
  run_distributed plain "$port" "$project_root/starVLA/training/build_register_candidate_bank.py" "${bank_args[@]}"
  python "$project_root/starVLA/training/build_register_candidate_bank.py" --config "$bank_config" --split "$logical_split" --validate-only
}
build_bank_split train $((REGISTER64_MAIN_PROCESS_PORT + 1))
build_bank_split val $((REGISTER64_MAIN_PROCESS_PORT + 2))

register64_phase=stage-s
export REGISTER64_DRIVOR_CHECKPOINT="$drivor_checkpoint"
drivor_complete="$stages_root/register64_drivor_scorer/training_complete.json"
if stage_complete "$drivor_complete" "$drivor_checkpoint" drivor_scorer; then
  echo "[register64] Stage S already complete: $drivor_checkpoint"
else
  run_distributed plain $((REGISTER64_MAIN_PROCESS_PORT + 3)) \
    "$project_root/starVLA/training/train_register_drivor.py" --config "$drivor_config"
fi
if ! stage_complete "$drivor_complete" "$drivor_checkpoint" drivor_scorer; then
  echo "[register64] Stage S did not produce a valid completion contract" >&2
  exit 2
fi

if [[ "$REGISTER64_ENABLE_SUPRIM" == 1 ]]; then
  register64_phase=stage-sd
  export REGISTER64_SUPRIM_DYNAMIC_CHECKPOINT="$suprim_checkpoint"
  suprim_complete="$stages_root/register64_drivor_suprim_dynamic/training_complete.json"
  if stage_complete "$suprim_complete" "$suprim_checkpoint" suprim_dynamic; then
    echo "[register64] Stage SD already complete: $suprim_checkpoint"
  else
    run_distributed plain $((REGISTER64_MAIN_PROCESS_PORT + 4)) \
      "$project_root/starVLA/training/train_register_suprim.py" --config "$suprim_config"
  fi
  if ! stage_complete "$suprim_complete" "$suprim_checkpoint" suprim_dynamic; then
    echo "[register64] Stage SD did not produce a valid completion contract" >&2
    exit 2
  fi
fi

register64_phase=navtest-prediction
unset NAVSIM_FEATURE_CACHE_ROOT
export NAVSIM_USE_FEATURE_CACHE=0
inference_config="$off_inference_config"
prediction_args=(
  --config "$inference_config" --datalist "$REGISTER64_NAVTEST_DATALIST"
  --data-root "$DATA_ROOT" --output-dir "$prediction_dir" --split test
  --batch-size "$REGISTER64_INFER_BATCH_SIZE" --num-workers "$REGISTER64_INFER_WORKERS"
  --generator-checkpoint "$generator_checkpoint" --drivor-checkpoint "$drivor_checkpoint"
)
if [[ "$REGISTER64_ENABLE_SUPRIM" == 1 ]]; then
  inference_config="$on_inference_config"
  prediction_args=(
    --config "$inference_config" --datalist "$REGISTER64_NAVTEST_DATALIST"
    --data-root "$DATA_ROOT" --output-dir "$prediction_dir" --split test
    --batch-size "$REGISTER64_INFER_BATCH_SIZE" --num-workers "$REGISTER64_INFER_WORKERS"
    --generator-checkpoint "$generator_checkpoint" --drivor-checkpoint "$drivor_checkpoint"
    --suprim-checkpoint "$suprim_checkpoint"
  )
fi
if [[ -d "$prediction_dir" ]]; then prediction_args+=(--resume); fi
run_distributed plain $((REGISTER64_MAIN_PROCESS_PORT + 5)) \
  "$project_root/starVLA/training/export_register_navtest_predictions.py" "${prediction_args[@]}"
expected_navtest="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$REGISTER64_NAVTEST_DATALIST")"
prediction_count="$(find "$prediction_dir" -maxdepth 1 -type f -name '*.npy' | wc -l)"
if (( prediction_count != expected_navtest )); then
  echo "[register64] prediction count mismatch: $prediction_count != $expected_navtest" >&2
  exit 2
fi

register64_phase=cache-navtest-v1-1
ensure_metric_cache v1.1 navtest "$PDMS_METRIC_CACHE_PATH" "$REGISTER64_NAVTEST_DATALIST"
register64_phase=cache-navtest-v2
ensure_metric_cache v2 navtest "$EPDMS_METRIC_CACHE_PATH" "$REGISTER64_NAVTEST_DATALIST"

score_valid() {
  python "$project_root/tools/validate_navsim_score_csv.py" \
    --results-dir "$1" --protocol "$2" --expected-scenarios "$expected_navtest" \
    --expected-datalist "$REGISTER64_NAVTEST_DATALIST" >/dev/null 2>&1
}

register64_phase=official-pdms-v1-1
if score_valid "$pdms_results" pdms; then
  echo "[register64] official PDMS already valid: $pdms_results"
else
  mkdir -p "$pdms_results"
  env \
    CUDA_VISIBLE_DEVICES= \
    PYTHONPATH="$project_root/navsim_v1.1/navsim:$project_root:${PYTHONPATH:-}" \
    NAVSIM_DEVKIT_ROOT="$project_root/navsim_v1.1/navsim" \
    NAVSIM_EXP_ROOT="$pdms_results" \
    OPENSCENE_DATA_ROOT="$OPENSCENE_DATA_ROOT" \
    python "$project_root/navsim_v1.1/navsim/navsim/planning/script/run_pdm_score.py" \
      train_test_split=navtest "metric_cache_path=$PDMS_METRIC_CACHE_PATH" \
      agent=human_agent experiment_name=register64-navtest-pdms \
      "output_dir=$pdms_results" "pred_dir=$prediction_root" split=test \
      worker=ray_distributed_no_torch "worker.threads_per_node=$REGISTER64_CACHE_WORKERS" \
      worker.use_distributed=false gpu=false
fi
python "$project_root/tools/validate_navsim_score_csv.py" \
  --results-dir "$pdms_results" --protocol pdms --expected-scenarios "$expected_navtest" \
  --expected-datalist "$REGISTER64_NAVTEST_DATALIST"

register64_phase=official-epdms-v2
if score_valid "$epdms_results" epdms; then
  echo "[register64] official EPDMS already valid: $epdms_results"
else
  mkdir -p "$epdms_results"
  env \
    CUDA_VISIBLE_DEVICES= \
    PYTHONPATH="$project_root/navsim:$project_root:${PYTHONPATH:-}" \
    NAVSIM_DEVKIT_ROOT="$project_root/navsim" \
    NAVSIM_EXP_ROOT="$epdms_results" \
    OPENSCENE_DATA_ROOT="$OPENSCENE_DATA_ROOT" \
    python "$project_root/navsim/navsim/planning/script/run_pdm_score_one_stage.py" \
      train_test_split=navtest "metric_cache_path=$EPDMS_METRIC_CACHE_PATH" \
      agent=human_agent experiment_name=register64-navtest-epdms \
      "output_dir=$epdms_results" "pred_dir=$prediction_root" split=test \
      worker=ray_distributed_no_torch "worker.threads_per_node=$REGISTER64_CACHE_WORKERS" \
      worker.use_distributed=false gpu=false
fi
python "$project_root/tools/validate_navsim_score_csv.py" \
  --results-dir "$epdms_results" --protocol epdms --expected-scenarios "$expected_navtest" \
  --expected-datalist "$REGISTER64_NAVTEST_DATALIST"

register64_phase=summary
summary_args=(
  --run-root "$run_root" --arm "$REGISTER64_ARM"
  --generator-variant "$REGISTER64_GENERATOR_VARIANT"
  --generator-stage-id "$generator_stage_id"
  --generator-checkpoint "$generator_checkpoint" --drivor-checkpoint "$drivor_checkpoint"
  --bank-root "$bank_root" --prediction-dir "$prediction_dir"
  --pdms-results-dir "$pdms_results"
  --epdms-results-dir "$epdms_results" --expected-scenarios "$expected_navtest"
  --navtest-datalist "$REGISTER64_NAVTEST_DATALIST"
  --output-dir "$summary_root"
)
if [[ "$REGISTER64_ENABLE_SUPRIM" == 1 ]]; then summary_args+=(--suprim-checkpoint "$suprim_checkpoint"); fi
python "$project_root/tools/collect_register64_pipeline_results.py" "${summary_args[@]}"

echo "[register64] complete arm=$REGISTER64_ARM"
echo "[register64] weights=$stages_root"
echo "[register64] official_results=$summary_root/summary.md"
echo "[register64] stable_summary=$summary_root/summary.csv"
