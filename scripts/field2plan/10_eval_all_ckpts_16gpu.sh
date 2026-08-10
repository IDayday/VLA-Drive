#!/usr/bin/env bash
# Incrementally evaluate an explicit Field2Plan experiment suite on one 16-PPU
# DLC node.  Up to eight checkpoints run concurrently; each checkpoint keeps
# the historical two-shard inference protocol so seed/token assignment remains
# directly comparable.  With no override this retains the original Phase-2
# eight-arm behavior.

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/../.." && pwd)"
source "${project_root}/env.sh"

default_experiments=(
  p2_00_nosup_noaccess
  p2_01_nosup_access
  p2_10_sup_noaccess_da3
  p2_11_sup_access_da3
  p2_11_sup_access_vggt
  p2_random_access_da3
  p2_shuffled_access_da3
  p2_state_mlp_access
)
if [[ -n "${FIELD2PLAN_EVAL_EXPERIMENTS:-}" ]]; then
  IFS=',' read -r -a experiments <<<"${FIELD2PLAN_EVAL_EXPERIMENTS}"
else
  experiments=("${default_experiments[@]}")
fi
experiment_seed="${FIELD2PLAN_EXPERIMENT_SEED:-42}"
if (( ${#experiments[@]} == 0 )); then
  echo "[field2plan-eval] ERROR: experiment list cannot be empty" >&2
  exit 2
fi
declare -A experiment_seen=()
for experiment in "${experiments[@]}"; do
  if ! [[ "$experiment" =~ ^[a-zA-Z0-9_]+$ ]]; then
    echo "[field2plan-eval] ERROR: invalid experiment name: $experiment" >&2
    exit 2
  fi
  if [[ -n "${experiment_seen[$experiment]:-}" ]]; then
    echo "[field2plan-eval] ERROR: duplicate experiment: $experiment" >&2
    exit 2
  fi
  experiment_seen["$experiment"]=1
done
if ! [[ "$experiment_seed" =~ ^[0-9]+$ ]]; then
  echo "[field2plan-eval] ERROR: FIELD2PLAN_EXPERIMENT_SEED must be non-negative" >&2
  exit 2
fi

infer_world_size=2
num_accelerators="${NUM_ACCELERATORS:-16}"
max_parallel_infer="${MAX_PARALLEL_INFER:-8}"
batch_size="${BATCH_SIZE:-32}"
num_workers="${NUM_WORKERS:-2}"
infer_seed="${INFER_SEED:-20260808}"
eval_threads="${EVAL_THREADS:-16}"
max_parallel_pdms="${MAX_PARALLEL_PDMS:-4}"
pdms_worker_backend="${PDMS_WORKER_BACKEND:-thread_pool}"
min_step="${MIN_STEP:-10000}"
max_step="${MAX_STEP:-100000}"
watch_for_checkpoints="${WATCH_FOR_CHECKPOINTS:-1}"
poll_seconds="${POLL_SECONDS:-120}"
checkpoint_min_age="${CHECKPOINT_MIN_AGE_SECONDS:-120}"
checkpoint_min_bytes="${CHECKPOINT_MIN_BYTES:-1000000000}"
max_task_attempts="${MAX_TASK_ATTEMPTS:-2}"
save_diagnostics="${SAVE_DIAGNOSTICS:-0}"
overwrite_predictions="${OVERWRITE_PREDICTIONS:-0}"
topology_only="${EVAL_TOPOLOGY_ONLY:-0}"
preflight_only="${EVAL_PREFLIGHT_ONLY:-0}"
discovery_only="${EVAL_DISCOVERY_ONLY:-0}"

protocol_id="navsim_v1_1_pdms_ws2_seed${infer_seed}"
orchestration_revision="${EVAL_ORCHESTRATION_REVISION:-distenvfix-v1}"
shared_root="${DRIVEDREAMER_SHARED_ROOT:-/mnt/zhangt_workspace/project/DriveDreamer-Policy}"
artifact_root="${EVAL_ARTIFACT_ROOT:-${shared_root}/navsim_exp/field2plan_eval_16gpu_live/${protocol_id}-${orchestration_revision}}"
prediction_root="${PRED_ROOT:-${shared_root}/navsim_planning_results/field2plan_all_ckpts_${protocol_id}-${orchestration_revision}}"
navsim_eval_root="${EVAL_ROOT:-${artifact_root}/navsim_outputs}"
log_root="${LOG_ROOT:-${artifact_root}/logs}"
summary_root="${SUMMARY_ROOT:-${artifact_root}/summary}"
state_root="${STATE_ROOT:-${artifact_root}/state}"
metric_cache_path="${METRIC_CACHE_PATH:-${project_root}/navsim_exp/metric_cache_navtest_v1_1}"
test_meta="${TEST_META:-${project_root}/test_meta.json}"
data_root="${DATA_ROOT}"
navsim_v1_root="${project_root}/navsim_v1.1/navsim"
run_id="${EVAL_RUN_ID:-field2plan-eval-$(date -u +%Y%m%dT%H%M%SZ)}"
local_cache_root="${LOCAL_CACHE_ROOT:-/tmp/drivedreamer-field2plan-eval/${run_id}}"

die() {
  echo "[field2plan-eval] ERROR: $*" >&2
  exit 2
}

require_uint() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer, got ${value}"
}

for pair in \
  "NUM_ACCELERATORS:${num_accelerators}" \
  "MAX_PARALLEL_INFER:${max_parallel_infer}" \
  "BATCH_SIZE:${batch_size}" \
  "NUM_WORKERS:${num_workers}" \
  "INFER_SEED:${infer_seed}" \
  "EVAL_THREADS:${eval_threads}" \
  "MAX_PARALLEL_PDMS:${max_parallel_pdms}" \
  "MIN_STEP:${min_step}" \
  "MAX_STEP:${max_step}" \
  "POLL_SECONDS:${poll_seconds}" \
  "CHECKPOINT_MIN_AGE_SECONDS:${checkpoint_min_age}" \
  "CHECKPOINT_MIN_BYTES:${checkpoint_min_bytes}" \
  "MAX_TASK_ATTEMPTS:${max_task_attempts}"; do
  require_uint "${pair%%:*}" "${pair#*:}"
done
(( num_accelerators > 0 && num_accelerators % infer_world_size == 0 )) || \
  die "NUM_ACCELERATORS must be positive and divisible by ${infer_world_size}"
(( max_parallel_infer > 0 && max_parallel_infer <= num_accelerators / infer_world_size )) || \
  die "MAX_PARALLEL_INFER must be in [1,$((num_accelerators / infer_world_size))]"
(( batch_size > 0 && eval_threads > 0 && max_parallel_pdms > 0 )) || \
  die "batch size, evaluator threads, and PDMS concurrency must be positive"
(( min_step > 0 && max_step >= min_step )) || die "invalid MIN_STEP/MAX_STEP"
[[ "$watch_for_checkpoints" =~ ^[01]$ ]] || die "WATCH_FOR_CHECKPOINTS must be 0 or 1"
[[ "$save_diagnostics" =~ ^[01]$ ]] || die "SAVE_DIAGNOSTICS must be 0 or 1"
[[ "$overwrite_predictions" =~ ^[01]$ ]] || die "OVERWRITE_PREDICTIONS must be 0 or 1"
[[ "$discovery_only" =~ ^[01]$ ]] || die "EVAL_DISCOVERY_ONLY must be 0 or 1"
[[ "$pdms_worker_backend" == "thread_pool" || "$pdms_worker_backend" == "ray" ]] || \
  die "PDMS_WORKER_BACKEND must be thread_pool or ray"

dlc_world_size="${WORLD_SIZE:-1}"
dlc_rank="${RANK:-0}"
if [[ "$dlc_world_size" != "1" || "$dlc_rank" != "0" ]]; then
  die "this launcher is single-node only; DLC WORLD_SIZE=${dlc_world_size} RANK=${dlc_rank}"
fi

declare -a all_devices=()
if [[ -n "${EVAL_DEVICE_IDS:-}" ]]; then
  IFS=',' read -r -a all_devices <<<"${EVAL_DEVICE_IDS}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a all_devices <<<"${CUDA_VISIBLE_DEVICES}"
else
  available_count="${AVAILABLE_ACCELERATOR_COUNT:-$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)}"
  require_uint "AVAILABLE_ACCELERATOR_COUNT" "$available_count"
  for ((device_index = 0; device_index < available_count; device_index++)); do
    all_devices+=("$device_index")
  done
fi
(( ${#all_devices[@]} >= num_accelerators )) || \
  die "requested ${num_accelerators} accelerators but only ${#all_devices[@]} are visible"
devices=("${all_devices[@]:0:num_accelerators}")

echo "project_root=${project_root}"
echo "run_id=${run_id}"
echo "script_sha256=$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"
echo "orchestration_revision=${orchestration_revision}"
echo "devices=$(IFS=,; echo "${devices[*]}")"
echo "topology=nodes:1 node_rank:0 physical_accelerators:${num_accelerators} concurrent_checkpoints:${max_parallel_infer} inference_shards_per_checkpoint:${infer_world_size}"
echo "inference=batch_per_shard:${batch_size} workers_per_shard:${num_workers} seed:${infer_seed} protocol:${protocol_id}"
echo "pdms=metric:NAVSIM-v1.1-PDMS backend:${pdms_worker_backend} threads_per_job:${eval_threads} max_parallel_jobs:${max_parallel_pdms}"
echo "steps=min:${min_step} max:${max_step} watch:${watch_for_checkpoints} poll_seconds:${poll_seconds}"
echo "experiment_seed=${experiment_seed} experiments=$(IFS=,; echo "${experiments[*]}")"
echo "data_root=${data_root}"
echo "prediction_root=${prediction_root}"
echo "summary_csv=${summary_root}/summary.csv"
echo "summary_md=${summary_root}/summary.md"
echo "events=${summary_root}/events.tsv"

if [[ "$topology_only" == "1" ]]; then
  echo "[field2plan-eval] topology-only validation passed"
  exit 0
fi

for required_path in \
  "${project_root}/infer.py" \
  "${project_root}/env.sh" \
  "${test_meta}" \
  "${data_root}" \
  "${metric_cache_path}/metadata" \
  "${navsim_v1_root}/navsim/planning/script/run_pdm_score.py" \
  "${project_root}/tools/field2plan/record_pdms_result.py"; do
  [[ -e "$required_path" ]] || die "required path is missing: ${required_path}"
done
for experiment in "${experiments[@]}"; do
  [[ -d "${project_root}/navsim_exp/field2plan-${experiment}-steps100000-seed${experiment_seed}/checkpoints" ]] || \
    die "experiment checkpoint directory is missing: ${experiment}"
done

mkdir -p \
  "${prediction_root}" \
  "${navsim_eval_root}" \
  "${log_root}/inference" \
  "${log_root}/pdms" \
  "${summary_root}" \
  "${state_root}/attempts" \
  "${state_root}/completed" \
  "${state_root}/inference_done" \
  "${state_root}/terminal_failures" \
  "${state_root}/stamps" \
  "${local_cache_root}"

events_tsv="${summary_root}/events.tsv"
if [[ ! -s "$events_tsv" ]]; then
  {
    flock -x 200
    if [[ ! -s "$events_tsv" ]]; then
      printf 'timestamp_utc\tstate\texperiment\tstep\tdetail\n' >&200
    fi
  } 200>>"${events_tsv}"
fi

record_event() {
  local state="$1"
  local experiment="$2"
  local step="$3"
  local detail="${4:-}"
  detail="${detail//$'\n'/ }"
  detail="${detail//$'\t'/ }"
  {
    flock -x 200
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$state" "$experiment" "$step" "$detail" >&200
  } 200>>"${events_tsv}"
}

atomic_marker() {
  local path="$1"
  local temporary="${path}.tmp-$$-$RANDOM"
  printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$temporary"
  mv -f "$temporary" "$path"
}

if [[ ! -f "${summary_root}/summary.csv" ]]; then
  summary_tmp="${summary_root}/.summary.csv.tmp-$$"
  printf '%s\n' 'updated_at_utc,status,protocol,experiment,checkpoint_step,pdms,scenario_count,valid_scenarios,failed_scenarios,inference_world_size,inference_seed,checkpoint,checkpoint_bytes,prediction_dir,result_csv,evaluator_log,error' >"$summary_tmp"
  mv -f "$summary_tmp" "${summary_root}/summary.csv"
fi
if [[ ! -f "${summary_root}/summary.md" ]]; then
  summary_md_tmp="${summary_root}/.summary.md.tmp-$$"
  printf '# Field2Plan live NAVSIM v1.1 PDMS summary\n\nNo evaluation records yet.\n' >"$summary_md_tmp"
  mv -f "$summary_md_tmp" "${summary_root}/summary.md"
fi

python - "${summary_root}/protocol.json" "$project_root" "$protocol_id" "$orchestration_revision" \
  "$infer_seed" "$infer_world_size" "$num_accelerators" "$batch_size" \
  "$num_workers" "$min_step" "$max_step" "$experiment_seed" "$pdms_worker_backend" \
  "$eval_threads" "${experiments[@]}" <<'PY'
import json
import os
from pathlib import Path
import subprocess
import sys

(
    output,
    project_root,
    protocol,
    orchestration_revision,
    seed,
    infer_world_size,
    accelerators,
    batch_size,
    workers,
    min_step,
    max_step,
    experiment_seed,
    pdms_worker_backend,
    evaluator_threads,
    *experiments,
) = sys.argv[1:]
payload = {
    "protocol": protocol,
    "orchestration_revision": orchestration_revision,
    "metric": "NAVSIM v1.1 PDMS",
    "git_commit": subprocess.check_output(
        ["git", "-C", project_root, "rev-parse", "HEAD"], text=True
    ).strip(),
    "inference_seed": int(seed),
    "inference_world_size_per_checkpoint": int(infer_world_size),
    "physical_accelerators": int(accelerators),
    "batch_size_per_shard": int(batch_size),
    "num_workers_per_shard": int(workers),
    "min_step": int(min_step),
    "max_step": int(max_step),
    "experiment_seed": int(experiment_seed),
    "pdms_worker_backend": pdms_worker_backend,
    "evaluator_threads_per_job": int(evaluator_threads),
    "experiments": experiments,
}
path = Path(output)
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
with temporary.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
PY

# The available feature cache contains navtrain only.  NAVSIM navtest
# inference must use raw test images and must not silently probe train entries.
unset NAVSIM_FEATURE_CACHE_ROOT

echo "[field2plan-eval] dependency/data preflight"
python - "$test_meta" "$data_root" "$num_accelerators" <<'PY'
import json
from pathlib import Path
import sys
import torch
import accelerate
import deepspeed
import transformers

tokens = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not isinstance(tokens, list) or len(tokens) != 12146 or len(set(tokens)) != len(tokens):
    raise SystemExit(f"expected 12146 unique navtest tokens, got {len(tokens)}")
data_root = Path(sys.argv[2])
if not data_root.is_dir():
    raise SystemExit(f"processed data root is missing: {data_root}")
expected_devices = int(sys.argv[3])
visible_devices = torch.cuda.device_count()
if visible_devices < expected_devices:
    raise SystemExit(
        f"expected at least {expected_devices} visible accelerators, "
        f"PyTorch sees {visible_devices}"
    )
print(
    f"torch={torch.__version__} transformers={transformers.__version__} "
    f"accelerate={accelerate.__version__} deepspeed={deepspeed.__version__} "
    f"visible_accelerators={visible_devices} navtest_tokens={len(tokens)}"
)
PY

if [[ "$preflight_only" == "1" ]]; then
  echo "[field2plan-eval] preflight-only validation passed"
  exit 0
fi

checkpoint_is_stable() {
  local checkpoint="$1"
  local size modified now age
  size="$(stat -c '%s' "$checkpoint")"
  modified="$(stat -c '%Y' "$checkpoint")"
  now="$(date +%s)"
  age=$((now - modified))
  (( size >= checkpoint_min_bytes && age >= checkpoint_min_age ))
}

task_key() {
  printf '%s-step%s' "$1" "$2"
}

checkpoint_path() {
  printf '%s/navsim_exp/field2plan-%s-steps100000-seed%s/checkpoints/steps_%s_pytorch_model.pt' \
    "$project_root" "$1" "$experiment_seed" "$2"
}

prediction_run_root() {
  printf '%s/field2plan-%s-steps100000-seed%s-step%s' \
    "$prediction_root" "$1" "$experiment_seed" "$2"
}

attempt_count() {
  local key="$1"
  local path="${state_root}/attempts/${key}.txt"
  if [[ -f "$path" ]]; then
    cat "$path"
  else
    echo 0
  fi
}

increment_attempt() {
  local key="$1"
  local current next temporary
  current="$(attempt_count "$key")"
  next=$((current + 1))
  temporary="${state_root}/attempts/.${key}.tmp-$$-$RANDOM"
  printf '%s\n' "$next" >"$temporary"
  mv -f "$temporary" "${state_root}/attempts/${key}.txt"
  echo "$next"
}

discover_pending_tasks() {
  local experiment checkpoint filename step key attempts
  shopt -s nullglob
  for experiment in "${experiments[@]}"; do
    for checkpoint in "${project_root}/navsim_exp/field2plan-${experiment}-steps100000-seed${experiment_seed}/checkpoints"/steps_*_pytorch_model.pt; do
      filename="${checkpoint##*/}"
      [[ "$filename" =~ ^steps_([0-9]+)_pytorch_model\.pt$ ]] || continue
      step="${BASH_REMATCH[1]}"
      (( step >= min_step && step <= max_step )) || continue
      key="$(task_key "$experiment" "$step")"
      [[ ! -f "${state_root}/completed/${key}.done" ]] || continue
      [[ ! -f "${state_root}/terminal_failures/${key}.failed" ]] || continue
      attempts="$(attempt_count "$key")"
      (( attempts < max_task_attempts )) || continue
      checkpoint_is_stable "$checkpoint" || continue
      printf '%s|%s|%s\n' "$step" "$experiment" "$checkpoint"
    done
  done | sort -t'|' -k1,1n -k2,2
}

validate_predictions() {
  local prediction_dir="$1"
  local checkpoint="$2"
  local step="$3"
  python - "$test_meta" "$prediction_dir" "$checkpoint" "$step" \
    "$infer_seed" "$infer_world_size" "$save_diagnostics" <<'PY'
import json
from pathlib import Path
import sys
import numpy as np

meta_path, prediction_value, checkpoint_value, step, seed, world_size, save_diagnostics = sys.argv[1:]
prediction_dir = Path(prediction_value)
checkpoint = Path(checkpoint_value).resolve()
step = int(step)
seed = int(seed)
world_size = int(world_size)
save_diagnostics = bool(int(save_diagnostics))
tokens = json.loads(Path(meta_path).read_text(encoding="utf-8"))
expected = set(tokens)
actual = {path.stem for path in prediction_dir.glob("*.npy")}
if actual != expected:
    raise SystemExit(
        f"prediction token mismatch: missing={len(expected - actual)} "
        f"extra={len(actual - expected)}"
    )
if save_diagnostics:
    diagnostics = {path.stem for path in (prediction_dir / "diagnostics").glob("*.npz")}
    if diagnostics != expected:
        raise SystemExit(
            f"diagnostic token mismatch: missing={len(expected - diagnostics)} "
            f"extra={len(diagnostics - expected)}"
        )
for rank in range(world_size):
    manifest_path = prediction_dir / f"inference_manifest.rank{rank}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["world_size"] != world_size or manifest["rank"] != rank:
        raise SystemExit(f"invalid shard manifest topology: {manifest_path}")
    if manifest["seed"] != seed or manifest["model_iter"] != step:
        raise SystemExit(f"invalid shard manifest seed/step: {manifest_path}")
    if Path(manifest["checkpoint_file"]).resolve() != checkpoint:
        raise SystemExit(f"shard manifest checkpoint mismatch: {manifest_path}")
for token in tokens:
    action = np.load(prediction_dir / f"{token}.npy", allow_pickle=False)
    if action.shape != (8, 3) or action.dtype != np.float32 or not np.isfinite(action).all():
        raise SystemExit(
            f"invalid prediction {token}: shape={action.shape} "
            f"dtype={action.dtype} finite={np.isfinite(action).all()}"
        )
print(f"validated {len(tokens)} predictions in {prediction_dir}")
PY
}

record_failed_result() {
  local experiment="$1"
  local step="$2"
  local checkpoint="$3"
  local prediction_dir="$4"
  local evaluator_log="$5"
  local error="$6"
  python "${project_root}/tools/field2plan/record_pdms_result.py" \
    --summary-root "$summary_root" \
    --status failed \
    --protocol "$protocol_id" \
    --experiment "$experiment" \
    --checkpoint-step "$step" \
    --inference-world-size "$infer_world_size" \
    --inference-seed "$infer_seed" \
    --checkpoint "$checkpoint" \
    --prediction-dir "$prediction_dir" \
    --evaluator-log "$evaluator_log" \
    --error "$error"
}

run_inference_task() {
  local experiment="$1"
  local step="$2"
  local checkpoint="$3"
  local device0="$4"
  local device1="$5"
  local key checkpoint_dir run_root prediction_dir rank device log_path
  key="$(task_key "$experiment" "$step")"
  checkpoint_dir="$(dirname -- "$(dirname -- "$checkpoint")")"
  run_root="$(prediction_run_root "$experiment" "$step")"
  prediction_dir="${run_root}/test"
  mkdir -p "$prediction_dir"
  record_event INFER_RUNNING "$experiment" "$step" "devices=${device0},${device1}"

  local -a infer_pids=()
  local -a common_args=(
    --ckpt_dir "$checkpoint_dir"
    --model_iter "$step"
    --datalist_path "$test_meta"
    --data_root "$data_root"
    --out_dir "$prediction_root"
    --split test
    --batch_size "$batch_size"
    --num_workers "$num_workers"
    --world_size "$infer_world_size"
    --seed "$infer_seed"
    --qwen_forward_mode auto
    --smooth 0
  )
  if [[ "$save_diagnostics" == "1" ]]; then
    common_args+=(--save_diagnostics)
  fi
  if [[ "$overwrite_predictions" == "1" ]]; then
    common_args+=(--overwrite)
  fi

  for rank in 0 1; do
    if [[ "$rank" == "0" ]]; then
      device="$device0"
    else
      device="$device1"
    fi
    log_path="${log_root}/inference/${key}.rank${rank}.log"
    mkdir -p "${local_cache_root}/${key}/rank${rank}"
    (
      export CUDA_VISIBLE_DEVICES="$device"
      # These are independent single-device inference shards, not torchrun
      # workers.  PAI-DLC injects distributed variables into the parent job;
      # inheriting them makes Accelerate initialize a process group during
      # module import and causes all concurrent checkpoint jobs to contend for
      # the same MASTER_PORT.  Sharding is controlled exclusively by the
      # explicit --rank/--world_size arguments below.
      unset WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT
      unset GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE TORCHELASTIC_RUN_ID
      export TRITON_CACHE_DIR="${local_cache_root}/${key}/rank${rank}/triton"
      export TORCH_EXTENSIONS_DIR="${local_cache_root}/${key}/rank${rank}/torch_extensions"
      mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
      exec python "${project_root}/infer.py" "${common_args[@]}" --rank "$rank"
    ) >"$log_path" 2>&1 &
    infer_pids+=("$!")
  done

  local failed=0
  local pid
  for pid in "${infer_pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" != "0" ]]; then
    record_event INFER_FAILED "$experiment" "$step" "logs=${log_root}/inference/${key}.rank*.log"
    return 1
  fi
  if ! validate_predictions "$prediction_dir" "$checkpoint" "$step" \
    >"${log_root}/inference/${key}.validate.log" 2>&1; then
    record_event VALIDATION_FAILED "$experiment" "$step" "log=${log_root}/inference/${key}.validate.log"
    return 1
  fi
  atomic_marker "${state_root}/inference_done/${key}.done"
  record_event INFER_COMPLETE "$experiment" "$step" "prediction_dir=${prediction_dir}"
}

run_pdms_task() {
  local experiment="$1"
  local step="$2"
  local checkpoint="$3"
  local key short_name run_root prediction_dir eval_name evaluator_log stamp result_csv error_text
  key="$(task_key "$experiment" "$step")"
  short_name="$experiment"
  run_root="$(prediction_run_root "$experiment" "$step")"
  prediction_dir="${run_root}/test"
  eval_name="${short_name}_step${step}_${protocol_id}"
  evaluator_log="${log_root}/pdms/${key}.log"
  stamp="${state_root}/stamps/${key}.stamp"
  touch "$stamp"
  record_event PDMS_RUNNING "$experiment" "$step" "threads=${eval_threads}"

  local -a worker_args=()
  if [[ "$pdms_worker_backend" == "thread_pool" ]]; then
    worker_args=(worker=single_machine_thread_pool worker.max_workers="$eval_threads")
  else
    worker_args=(worker=ray_distributed_no_torch worker.threads_per_node="$eval_threads" worker.log_to_driver=false)
  fi

  if (
    export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
    export NAVSIM_EXP_ROOT="$navsim_eval_root"
    export NAVSIM_DEVKIT_ROOT="$navsim_v1_root"
    export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"
    export RAY_TMPDIR="${local_cache_root}/${key}/ray"
    mkdir -p "$RAY_TMPDIR"
    exec python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score.py" \
      train_test_split=navtest \
      agent=human_agent \
      experiment_name="$eval_name" \
      metric_cache_path="$metric_cache_path" \
      pred_dir="$run_root" \
      split=test \
      "${worker_args[@]}"
  ) >"$evaluator_log" 2>&1; then
    result_csv="$(find "${navsim_eval_root}/${eval_name}" -type f -name '*.csv' -newer "$stamp" -print 2>/dev/null | sort | tail -n 1)"
    if [[ -z "$result_csv" || ! -s "$result_csv" ]]; then
      error_text="evaluator exited successfully but produced no new result CSV"
      record_failed_result "$experiment" "$step" "$checkpoint" "$prediction_dir" "$evaluator_log" "$error_text"
      record_event PDMS_FAILED "$experiment" "$step" "$error_text"
      return 1
    fi
    if python "${project_root}/tools/field2plan/record_pdms_result.py" \
      --summary-root "$summary_root" \
      --status complete \
      --protocol "$protocol_id" \
      --experiment "$experiment" \
      --checkpoint-step "$step" \
      --inference-world-size "$infer_world_size" \
      --inference-seed "$infer_seed" \
      --checkpoint "$checkpoint" \
      --prediction-dir "$prediction_dir" \
      --evaluator-log "$evaluator_log" \
      --result-csv "$result_csv" \
      >>"$evaluator_log" 2>&1; then
      atomic_marker "${state_root}/completed/${key}.done"
      record_event COMPLETE "$experiment" "$step" "result_csv=${result_csv}"
      return 0
    fi
    error_text="result recorder rejected evaluator CSV"
  else
    error_text="PDMS evaluator exited non-zero"
  fi
  record_failed_result "$experiment" "$step" "$checkpoint" "$prediction_dir" "$evaluator_log" "$error_text"
  record_event PDMS_FAILED "$experiment" "$step" "$error_text"
  return 1
}

mark_terminal_if_exhausted() {
  local experiment="$1"
  local step="$2"
  local key attempts
  key="$(task_key "$experiment" "$step")"
  attempts="$(attempt_count "$key")"
  if (( attempts >= max_task_attempts )); then
    atomic_marker "${state_root}/terminal_failures/${key}.failed"
    record_event TERMINAL_FAILURE "$experiment" "$step" "attempts=${attempts}"
  fi
}

run_pdms_batch() {
  local -a tasks=("$@")
  local -a pids=()
  local -a metadata=()
  local task step experiment checkpoint pid index
  for task in "${tasks[@]}"; do
    IFS='|' read -r step experiment checkpoint <<<"$task"
    run_pdms_task "$experiment" "$step" "$checkpoint" &
    pids+=("$!")
    metadata+=("$task")
    if (( ${#pids[@]} >= max_parallel_pdms )); then
      for index in "${!pids[@]}"; do
        pid="${pids[$index]}"
        IFS='|' read -r step experiment checkpoint <<<"${metadata[$index]}"
        if ! wait "$pid"; then
          mark_terminal_if_exhausted "$experiment" "$step"
        fi
      done
      pids=()
      metadata=()
    fi
  done
  for index in "${!pids[@]}"; do
    pid="${pids[$index]}"
    IFS='|' read -r step experiment checkpoint <<<"${metadata[$index]}"
    if ! wait "$pid"; then
      mark_terminal_if_exhausted "$experiment" "$step"
    fi
  done
}

target_complete() {
  local experiment key
  for experiment in "${experiments[@]}"; do
    key="$(task_key "$experiment" "$max_step")"
    [[ -f "${state_root}/completed/${key}.done" ]] || return 1
  done
}

if [[ "$discovery_only" == "1" ]]; then
  mapfile -t discovered_tasks < <(discover_pending_tasks)
  printf 'discovered_pending_checkpoints=%s\n' "${#discovered_tasks[@]}"
  printf '%s\n' "${discovered_tasks[@]}"
  exit 0
fi

while true; do
  mapfile -t pending_tasks < <(discover_pending_tasks)
  if (( ${#pending_tasks[@]} == 0 )); then
    terminal_count="$(find "${state_root}/terminal_failures" -maxdepth 1 -type f -name '*.failed' | wc -l)"
    if (( terminal_count > 0 )); then
      die "${terminal_count} checkpoint task(s) exhausted retries; inspect ${summary_root}/events.tsv"
    fi
    if [[ "$watch_for_checkpoints" == "0" ]]; then
      echo "[field2plan-eval] no more stable checkpoints in the current snapshot"
      break
    fi
    if target_complete; then
      echo "[field2plan-eval] all eight experiment arms completed step ${max_step}"
      break
    fi
    echo "[field2plan-eval] waiting ${poll_seconds}s for new stable checkpoints"
    sleep "$poll_seconds"
    continue
  fi

  batch_count="${#pending_tasks[@]}"
  if (( batch_count > max_parallel_infer )); then
    batch_count="$max_parallel_infer"
  fi
  batch_tasks=("${pending_tasks[@]:0:batch_count}")
  declare -a infer_pids=()
  declare -a infer_metadata=()
  for index in "${!batch_tasks[@]}"; do
    task="${batch_tasks[$index]}"
    IFS='|' read -r step experiment checkpoint <<<"$task"
    key="$(task_key "$experiment" "$step")"
    attempt="$(increment_attempt "$key")"
    device0="${devices[$((index * infer_world_size))]}"
    device1="${devices[$((index * infer_world_size + 1))]}"
    echo "[infer-launch] experiment=${experiment} step=${step} attempt=${attempt} devices=${device0},${device1}"
    run_inference_task "$experiment" "$step" "$checkpoint" "$device0" "$device1" &
    infer_pids+=("$!")
    infer_metadata+=("$task")
  done

  successful_tasks=()
  for index in "${!infer_pids[@]}"; do
    pid="${infer_pids[$index]}"
    task="${infer_metadata[$index]}"
    IFS='|' read -r step experiment checkpoint <<<"$task"
    if wait "$pid"; then
      successful_tasks+=("$task")
    else
      key="$(task_key "$experiment" "$step")"
      prediction_dir="$(prediction_run_root "$experiment" "$step")/test"
      evaluator_log="${log_root}/inference/${key}.rank0.log"
      record_failed_result "$experiment" "$step" "$checkpoint" "$prediction_dir" "$evaluator_log" "inference or prediction validation failed"
      mark_terminal_if_exhausted "$experiment" "$step"
    fi
  done

  if (( ${#successful_tasks[@]} > 0 )); then
    run_pdms_batch "${successful_tasks[@]}"
  fi
  echo "[field2plan-eval] live summary: ${summary_root}/summary.md"
done

echo "[field2plan-eval] completed"
echo "summary_csv=${summary_root}/summary.csv"
echo "summary_md=${summary_root}/summary.md"
echo "events=${summary_root}/events.tsv"
