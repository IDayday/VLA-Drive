#!/usr/bin/env bash
# Non-interactive NAVSIM full-training entrypoint for local and PAI-DLC jobs.
# PAI-DLC injects WORLD_SIZE/RANK/NPROC_PER_NODE/MASTER_ADDR/MASTER_PORT.
# The formal job is pinned to 1 node x 16 accelerators. The released effective
# batch is always preserved at 32:
#   formal: 1 node x 16 accelerators x batch 2 x accumulation 1 = 32
#   smoke:  1 node x  2 accelerators x batch 2 x accumulation 8 = 32

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"
source "$project_root/env.sh"

export TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-32}"

# DLC defines WORLD_SIZE as the number of nodes and RANK as the node rank.
# Explicit NUM_MACHINES/MACHINE_RANK/LOCAL_NUM_PROCESSES values take precedence
# so the same script can also be used outside DLC.
export NUM_MACHINES="${NUM_MACHINES:-${WORLD_SIZE:-1}}"
export MACHINE_RANK="${MACHINE_RANK:-${RANK:-0}}"

actual_local_devices="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
available_local_devices="${NPROC_PER_NODE:-$actual_local_devices}"

# PAI-DLC sets NPROC_PER_NODE and/or PAI_JOB_ID. Formal jobs are intentionally
# strict: the requested topology is one node with 16 global accelerators. Local
# smoke tests remain dynamic and use however many accelerators are visible.
formal_job=0
if [ -n "${NPROC_PER_NODE:-}" ] || [ -n "${PAI_JOB_ID:-}" ]; then
  formal_job=1
fi

if (( formal_job == 1 )); then
  expected_formal_machines="${FORMAL_NUM_MACHINES:-1}"
  if ! [[ "$expected_formal_machines" =~ ^[0-9]+$ ]] || (( expected_formal_machines < 1 )); then
    echo "[training] FORMAL_NUM_MACHINES must be a positive integer, got: ${expected_formal_machines}" >&2
    exit 2
  fi
  if (( NUM_MACHINES != expected_formal_machines )); then
    echo "[training] Formal NAVSIM v1 training requires ${expected_formal_machines} node, but DLC reports NUM_MACHINES=${NUM_MACHINES}" >&2
    echo "[training] Recreate the DLC job as 1 node x 16 accelerators." >&2
    exit 2
  fi
fi

if [ -z "${NUM_PROCESSES:-}" ] && [ -z "${LOCAL_NUM_PROCESSES:-}" ]; then
  if (( formal_job == 1 )); then
    export NUM_PROCESSES="${FORMAL_NUM_PROCESSES:-16}"
    if (( NUM_PROCESSES % NUM_MACHINES != 0 )); then
      echo "[training] FORMAL_NUM_PROCESSES=${NUM_PROCESSES} must be divisible by NUM_MACHINES=${NUM_MACHINES}" >&2
      exit 2
    fi
    export LOCAL_NUM_PROCESSES=$((NUM_PROCESSES / NUM_MACHINES))
  else
    export LOCAL_NUM_PROCESSES="$available_local_devices"
    export NUM_PROCESSES="$LOCAL_NUM_PROCESSES"
  fi
elif [ -z "${LOCAL_NUM_PROCESSES:-}" ]; then
  if (( NUM_PROCESSES % NUM_MACHINES != 0 )); then
    echo "[training] NUM_PROCESSES=${NUM_PROCESSES} must be divisible by NUM_MACHINES=${NUM_MACHINES}" >&2
    exit 2
  fi
  export LOCAL_NUM_PROCESSES=$((NUM_PROCESSES / NUM_MACHINES))
elif [ -z "${NUM_PROCESSES:-}" ]; then
  export NUM_PROCESSES=$((NUM_MACHINES * LOCAL_NUM_PROCESSES))
fi
export MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-${MASTER_ADDR:-127.0.0.1}}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-${MASTER_PORT:-29687}}"

integer_vars=(
  NUM_MACHINES
  MACHINE_RANK
  LOCAL_NUM_PROCESSES
  NUM_PROCESSES
  available_local_devices
  TARGET_EFFECTIVE_BATCH_SIZE
  MAIN_PROCESS_PORT
)
for integer_var in "${integer_vars[@]}"; do
  integer_value="${!integer_var}"
  if ! [[ "$integer_value" =~ ^[0-9]+$ ]]; then
    echo "[training] ${integer_var} must be a non-negative integer, got: ${integer_value}" >&2
    exit 2
  fi
done

if (( NUM_MACHINES < 1 || LOCAL_NUM_PROCESSES < 1 || NUM_PROCESSES < 1 )); then
  echo "[training] NUM_MACHINES, LOCAL_NUM_PROCESSES and NUM_PROCESSES must be positive" >&2
  exit 2
fi
if (( MACHINE_RANK >= NUM_MACHINES )); then
  echo "[training] MACHINE_RANK=${MACHINE_RANK} is outside NUM_MACHINES=${NUM_MACHINES}" >&2
  exit 2
fi
expected_processes=$((NUM_MACHINES * LOCAL_NUM_PROCESSES))
if (( NUM_PROCESSES != expected_processes )); then
  echo "[training] NUM_PROCESSES=${NUM_PROCESSES}, but NUM_MACHINES=${NUM_MACHINES} x LOCAL_NUM_PROCESSES=${LOCAL_NUM_PROCESSES} gives ${expected_processes}" >&2
  exit 2
fi
if (( LOCAL_NUM_PROCESSES > available_local_devices )); then
  echo "[training] Need ${LOCAL_NUM_PROCESSES} local accelerators, but DLC exposes ${available_local_devices}" >&2
  exit 2
fi
if (( formal_job == 1 )); then
  expected_formal_processes="${FORMAL_NUM_PROCESSES:-16}"
  if ! [[ "$expected_formal_processes" =~ ^[0-9]+$ ]] || (( expected_formal_processes < 1 )); then
    echo "[training] FORMAL_NUM_PROCESSES must be a positive integer, got: ${expected_formal_processes}" >&2
    exit 2
  fi
  if (( NUM_PROCESSES != expected_formal_processes )); then
    echo "[training] Formal NAVSIM v1 training requires ${expected_formal_processes} global accelerators, but NUM_PROCESSES=${NUM_PROCESSES}" >&2
    exit 2
  fi
fi

if [ -z "${PER_DEVICE_BATCH_SIZE:-}" ]; then
  if (( NUM_PROCESSES >= 8 )); then
    if (( TARGET_EFFECTIVE_BATCH_SIZE % NUM_PROCESSES != 0 )); then
      echo "[training] TARGET_EFFECTIVE_BATCH_SIZE must be divisible by NUM_PROCESSES" >&2
      exit 2
    fi
    export PER_DEVICE_BATCH_SIZE=$((TARGET_EFFECTIVE_BATCH_SIZE / NUM_PROCESSES))
  else
    export PER_DEVICE_BATCH_SIZE=2
  fi
fi

if [ -z "${GRADIENT_ACCUMULATION_STEPS:-}" ]; then
  batch_per_microstep=$((NUM_PROCESSES * PER_DEVICE_BATCH_SIZE))
  if (( batch_per_microstep > TARGET_EFFECTIVE_BATCH_SIZE || TARGET_EFFECTIVE_BATCH_SIZE % batch_per_microstep != 0 )); then
    echo "[training] Cannot preserve effective batch ${TARGET_EFFECTIVE_BATCH_SIZE} with NUM_PROCESSES=${NUM_PROCESSES} and PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}" >&2
    exit 2
  fi
  export GRADIENT_ACCUMULATION_STEPS=$((TARGET_EFFECTIVE_BATCH_SIZE / batch_per_microstep))
fi
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100000}"
export NAVSIM_VIDEO_SOURCE="${NAVSIM_VIDEO_SOURCE:-images}"
export STARVLA_PREFETCH_QWEN="${STARVLA_PREFETCH_QWEN:-1}"
host_cpu_count="${TRAINING_HOST_CPU_COUNT_OVERRIDE:-$(getconf _NPROCESSORS_ONLN)}"
if ! [[ "$host_cpu_count" =~ ^[1-9][0-9]*$ ]]; then
  echo "[training] host CPU count must be a positive integer, got: $host_cpu_count" >&2
  exit 2
fi
if (( formal_job == 1 )); then
  # Qwen CPU preprocessing is compute-heavy while cached NAVSIM I/O is not.
  # On the formal 128-CPU/16-rank host, two workers x three threads per rank
  # overlap preprocessing with the accelerator and leave one spare CPU/rank.
  cpu_per_rank=$((host_cpu_count / LOCAL_NUM_PROCESSES))
  if [ "$STARVLA_PREFETCH_QWEN" = "1" ]; then
    default_workers=2
    default_worker_threads=$(((cpu_per_rank - 2) / default_workers))
    if (( default_worker_threads < 1 )); then default_worker_threads=1; fi
  else
    default_workers=$((cpu_per_rank - 2))
    if (( default_workers < 1 )); then default_workers=1; fi
    if (( default_workers > 8 )); then default_workers=8; fi
    default_worker_threads=1
  fi
  export NAVSIM_NUM_WORKERS="${NAVSIM_NUM_WORKERS:-$default_workers}"
  export NAVSIM_PREFETCH_FACTOR="${NAVSIM_PREFETCH_FACTOR:-2}"
  export NAVSIM_WORKER_THREADS="${NAVSIM_WORKER_THREADS:-$default_worker_threads}"
  export NAVSIM_STAGE_CACHE_TO_RAM="${NAVSIM_STAGE_CACHE_TO_RAM:-auto}"
  export NAVSIM_STAGE_METADATA_TO_RAM="${NAVSIM_STAGE_METADATA_TO_RAM:-auto}"
else
  export NAVSIM_NUM_WORKERS="${NAVSIM_NUM_WORKERS:-1}"
  export NAVSIM_PREFETCH_FACTOR="${NAVSIM_PREFETCH_FACTOR:-2}"
  export NAVSIM_WORKER_THREADS="${NAVSIM_WORKER_THREADS:-1}"
  export NAVSIM_STAGE_CACHE_TO_RAM="${NAVSIM_STAGE_CACHE_TO_RAM:-0}"
  export NAVSIM_STAGE_METADATA_TO_RAM="${NAVSIM_STAGE_METADATA_TO_RAM:-0}"
fi
export NAVSIM_PIN_MEMORY="${NAVSIM_PIN_MEMORY:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
for loader_integer in NAVSIM_NUM_WORKERS NAVSIM_PREFETCH_FACTOR NAVSIM_WORKER_THREADS; do
  loader_value="${!loader_integer}"
  if ! [[ "$loader_value" =~ ^[0-9]+$ ]]; then
    echo "[training] $loader_integer must be a non-negative integer, got: $loader_value" >&2
    exit 2
  fi
done
if (( NAVSIM_PREFETCH_FACTOR < 1 || NAVSIM_WORKER_THREADS < 1 )); then
  echo "[training] NAVSIM_PREFETCH_FACTOR and NAVSIM_WORKER_THREADS must be positive" >&2
  exit 2
fi
if [ "$NAVSIM_PIN_MEMORY" != "0" ] && [ "$NAVSIM_PIN_MEMORY" != "1" ]; then
  echo "[training] NAVSIM_PIN_MEMORY must be 0 or 1" >&2
  exit 2
fi
if ! [[ "${NAVSIM_RAM_COPY_WORKERS:-}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[training] NAVSIM_RAM_COPY_WORKERS must be a positive integer" >&2
  exit 2
fi
if ! [[ "${NAVSIM_METADATA_COPY_WORKERS:-}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[training] NAVSIM_METADATA_COPY_WORKERS must be a positive integer" >&2
  exit 2
fi
total_loader_workers=$((NAVSIM_NUM_WORKERS * LOCAL_NUM_PROCESSES))
total_loader_worker_threads=$((total_loader_workers * NAVSIM_WORKER_THREADS))
if (( formal_job == 1 && total_loader_worker_threads + LOCAL_NUM_PROCESSES > host_cpu_count )); then
  echo "[training] DataLoader oversubscription: worker_threads=$total_loader_worker_threads ranks=$LOCAL_NUM_PROCESSES host_cpus=$host_cpu_count" >&2
  exit 2
fi
export VLM_ATTN_IMPLEMENTATION="${VLM_ATTN_IMPLEMENTATION:-flash_attention_2}"
export STARVLA_DENSE_FLASH_ATTN="${STARVLA_DENSE_FLASH_ATTN:-1}"
export STARVLA_CACHE_WAN_ROPE="${STARVLA_CACHE_WAN_ROPE:-1}"
export STARVLA_FUSED_ADAMW="${STARVLA_FUSED_ADAMW:-1}"
export STARVLA_ACTION_PREPROJECT="${STARVLA_ACTION_PREPROJECT:-1}"
export TRAIN_ACCELERATE_CONFIG="${TRAIN_ACCELERATE_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}"
export WANDB_MODE="${WANDB_MODE:-offline}"

for optimization_flag in \
  STARVLA_DENSE_FLASH_ATTN \
  STARVLA_CACHE_WAN_ROPE \
  STARVLA_FUSED_ADAMW \
  STARVLA_ACTION_PREPROJECT \
  STARVLA_PREFETCH_QWEN; do
  optimization_value="${!optimization_flag}"
  if [ "$optimization_value" != "0" ] && [ "$optimization_value" != "1" ]; then
    echo "[training] $optimization_flag must be 0 or 1, got: $optimization_value" >&2
    exit 2
  fi
done

if [ "${NAVSIM_USE_FEATURE_CACHE:-1}" = "1" ]; then
  : "${NAVSIM_FEATURE_CACHE_ROOT:?NAVSIM_FEATURE_CACHE_ROOT is required when NAVSIM_USE_FEATURE_CACHE=1}"
else
  unset NAVSIM_FEATURE_CACHE_ROOT
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  visible_devices=""
  for ((device_idx = 0; device_idx < LOCAL_NUM_PROCESSES; device_idx++)); do
    if [ -n "$visible_devices" ]; then
      visible_devices+=","
    fi
    visible_devices+="$device_idx"
  done
  export CUDA_VISIBLE_DEVICES="$visible_devices"
fi

timestamp="$(date +'%Y%m%d_%H%M%S')"
job_tag="${JOB_NAME:-${PAI_JOB_ID:-}}"
if [ -z "$job_tag" ] && (( NUM_MACHINES > 1 )); then
  job_tag="${MASTER_ADDR:-}"
fi
if [ -z "$job_tag" ]; then
  job_tag="$timestamp"
fi
job_tag="${job_tag//[^[:alnum:]._-]/_}"
export RUN_ID="${RUN_ID:-navsim-v1-formal-${NUM_PROCESSES}gpu-bz${PER_DEVICE_BATCH_SIZE}-ga${GRADIENT_ACCUMULATION_STEPS}-flashattn-${job_tag}}"
launcher_log_dir="${NAVSIM_EXP_ROOT}/launcher_logs"
mkdir -p "$launcher_log_dir"
launcher_log="${launcher_log_dir}/${RUN_ID}.node${MACHINE_RANK}.log"

# Triton cache must be node-local. A shared CPFS cache produced stale file
# handles when many DeepSpeed processes updated its autotune table at exit.
export TRITON_CACHE_DIR="${TRITON_LOCAL_CACHE_ROOT:-/tmp/drivedreamer-triton}/${RUN_ID}/node${MACHINE_RANK}"
mkdir -p "$TRITON_CACHE_DIR"

exec > >(tee -a "$launcher_log") 2>&1

effective_batch=$((NUM_PROCESSES * PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
if (( effective_batch != TARGET_EFFECTIVE_BATCH_SIZE )); then
  echo "[training] Refusing effective batch ${effective_batch}; released configuration requires ${TARGET_EFFECTIVE_BATCH_SIZE}" >&2
  exit 2
fi

echo "[training] project_root=$project_root"
echo "[training] run_id=$RUN_ID"
echo "[training] devices=$CUDA_VISIBLE_DEVICES"
echo "[training] topology=nodes:${NUM_MACHINES} node_rank:${MACHINE_RANK} local_processes:${LOCAL_NUM_PROCESSES} global_processes:${NUM_PROCESSES} available_local_devices:${available_local_devices} master:${MAIN_PROCESS_IP}:${MAIN_PROCESS_PORT}"
echo "[training] per_device_batch=$PER_DEVICE_BATCH_SIZE gradient_accumulation=$GRADIENT_ACCUMULATION_STEPS effective_batch=$effective_batch (target=$TARGET_EFFECTIVE_BATCH_SIZE)"
echo "[training] attention=$VLM_ATTN_IMPLEMENTATION deepspeed_config=$TRAIN_ACCELERATE_CONFIG"
echo "[training] kernel_optimizations=dense_flash:${STARVLA_DENSE_FLASH_ATTN} wan_rope_cache:${STARVLA_CACHE_WAN_ROPE} qwen_worker_preprocess:${STARVLA_PREFETCH_QWEN} fused_adamw:${STARVLA_FUSED_ADAMW} action_preproject:${STARVLA_ACTION_PREPROJECT}"
echo "[training] max_steps=$MAX_TRAIN_STEPS log=$launcher_log"
echo "[training] feature_cache=${NAVSIM_FEATURE_CACHE_ROOT:-disabled} components=${NAVSIM_CACHE_COMPONENTS:-none} strict=${NAVSIM_CACHE_STRICT:-0}"
echo "[training] dataloader=host_cpus:${host_cpu_count} workers_per_rank:${NAVSIM_NUM_WORKERS} worker_threads:${NAVSIM_WORKER_THREADS} total_workers:${total_loader_workers} prefetch_factor:${NAVSIM_PREFETCH_FACTOR} pin_memory:${NAVSIM_PIN_MEMORY}"
echo "[training] ram_cache=mode:${NAVSIM_STAGE_CACHE_TO_RAM} components:${NAVSIM_RAM_CACHE_COMPONENTS:-none} root:${NAVSIM_RAM_CACHE_ROOT:-unset} reserve_gb:${NAVSIM_RAM_RESERVE_GB:-unset} copy_workers:${NAVSIM_RAM_COPY_WORKERS:-unset}"
echo "[training] ram_metadata=mode:${NAVSIM_STAGE_METADATA_TO_RAM} root:${NAVSIM_RAM_DATA_ROOT:-unset} copy_workers:${NAVSIM_METADATA_COPY_WORKERS:-unset}"

if [ "${TRAINING_TOPOLOGY_ONLY:-0}" = "1" ]; then
  echo "[training] TRAINING_TOPOLOGY_ONLY=1; data preflight and launcher were not started"
  exit 0
fi

required_paths=(
  "$BASE_VLM/config.json"
  "$VIDEO_MODEL"
  "$DATA_ROOT/meta/train"
  "$NAVSIM_TRAINVAL_SENSOR_ROOT"
  "$project_root/train_meta.json"
)
if [ -n "${NAVSIM_FEATURE_CACHE_ROOT:-}" ]; then
  IFS=',' read -r -a cache_components <<< "${NAVSIM_CACHE_COMPONENTS:-wan,ppd}"
  for cache_component in "${cache_components[@]}"; do
    cache_component="${cache_component//[[:space:]]/}"
    required_paths+=("$NAVSIM_FEATURE_CACHE_ROOT/$cache_component/manifest.json")
  done
fi
for required_path in "${required_paths[@]}"; do
  if [ ! -e "$required_path" ]; then
    echo "[training] missing required path: $required_path" >&2
    exit 2
  fi
done

python - <<'PY'
import json
import hashlib
import os
import pickle
from pathlib import Path

import accelerate
import deepspeed
import flash_attn
import torch
from nuplan.common.actor_state.state_representation import StateSE2
from starVLA.dataloader.navsim_dataset import NavSimDataset, resolve_navsim_data_path

with open("train_meta.json", "r", encoding="utf-8") as stream:
    train_samples = json.load(stream)
sample_count = len(train_samples)
with open("train_meta.json", "rb") as stream:
    datalist_sha256 = hashlib.sha256(stream.read()).hexdigest()

def model_tree_signature(path_value):
    path = Path(path_value).resolve()
    digest = hashlib.sha256()
    digest.update(str(path).encode("utf-8"))
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        stat = item.stat()
        digest.update(str(item.relative_to(path.parent if path.is_file() else path)).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()

signature_cache = {}

cache_root = os.environ.get("NAVSIM_FEATURE_CACHE_ROOT", "")
if cache_root:
    components = [
        value.strip()
        for value in os.environ.get("NAVSIM_CACHE_COMPONENTS", "wan,ppd").split(",")
        if value.strip()
    ]
    expected_model_paths = {
        "qwen": [os.environ["BASE_VLM"]],
        "wan": [os.environ["VIDEO_MODEL"]],
        "ppd": [os.environ["PPD_MODEL"], os.environ["DEPTH_ANYTHING_MODEL"]],
    }
    expected_contract = {
        "ver_1225": 1,
        "act_tok": 8,
        "w_depth": 1,
        "enable_image_aug": 0,
        "video_source": os.environ.get("NAVSIM_VIDEO_SOURCE", "images"),
        "video_text_input": 0,
        "load_2d_data": 1,
        "load_3d_data": 0,
        "load_reward_data": 0,
    }
    for component in components:
        manifest_path = Path(cache_root) / component / "manifest.json"
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if not manifest.get("complete", False):
            raise RuntimeError(f"Incomplete NAVSIM feature cache: {manifest_path}")
        if int(manifest.get("sample_count", -1)) != sample_count:
            raise RuntimeError(
                f"Cache sample count mismatch for {component}: "
                f"cache={manifest.get('sample_count')} dataset={sample_count}"
            )
        # Cache ownership is fixed by the cache manifest, not by the current
        # training world size.  Every training rank can read every shared
        # shard, so a 2-card smoke test can safely consume a complete cache
        # produced by the formal 16-card precompute job.
        cache_world_size = int(manifest.get("world_size", -1))
        if cache_world_size < 1:
            raise RuntimeError(
                f"Invalid cache shard count for {component}: {cache_world_size}"
            )
        if cache_world_size != int(os.environ["NUM_PROCESSES"]):
            print(
                f"[training] cache shards differ from training ranks for {component}: "
                f"cache={cache_world_size} training={os.environ['NUM_PROCESSES']} "
                "(supported: cache files are shared and manifest-addressed)"
            )
        if manifest.get("datalist_sha256") != datalist_sha256:
            raise RuntimeError(f"Cache datalist fingerprint mismatch: {manifest_path}")
        actual_paths = [str(Path(value).resolve()) for value in manifest.get("model_paths", [])]
        wanted_paths = [str(Path(value).resolve()) for value in expected_model_paths[component]]
        if actual_paths != wanted_paths:
            raise RuntimeError(
                f"Cache model path mismatch for {component}: "
                f"cache={actual_paths} expected={wanted_paths}"
            )
        cached_signatures = manifest.get("model_signatures", {})
        for model_path in wanted_paths:
            if model_path not in signature_cache:
                signature_cache[model_path] = model_tree_signature(model_path)
            if cached_signatures.get(model_path) != signature_cache[model_path]:
                raise RuntimeError(
                    f"Cache model fingerprint mismatch for {component}: {model_path}"
                )
        if manifest.get("cache_contract") != expected_contract:
            raise RuntimeError(
                f"Cache preprocessing contract mismatch for {component}: "
                f"cache={manifest.get('cache_contract')} expected={expected_contract}"
            )
        completions = manifest.get("rank_completions", [])
        if len(completions) != int(manifest["world_size"]):
            raise RuntimeError(f"Incomplete cache rank markers: {manifest_path}")
        for completion in completions:
            owned = int(completion.get("owned_samples", -1))
            accounted = int(completion.get("written", 0)) + int(completion.get("skipped", 0))
            if accounted != owned:
                raise RuntimeError(
                    f"Cache rank did not account for all samples: {component} {completion}"
                )

expected_devices = int(os.environ["LOCAL_NUM_PROCESSES"])
actual_devices = torch.cuda.device_count()
if actual_devices < expected_devices:
    raise RuntimeError(
        f"Only {actual_devices} visible accelerators on this node, "
        f"but LOCAL_NUM_PROCESSES={expected_devices}"
    )
if sample_count != 103288:
    raise RuntimeError(f"Expected 103288 NAVSIM training samples, found {sample_count}")

# Processed metadata contains the absolute project path used during data
# generation. Validate that runtime path remapping reaches real DLC files.
sample_indices = sorted({0, sample_count // 3, 2 * sample_count // 3, sample_count - 1})
checked_images = 0
for sample_index in sample_indices:
    token = train_samples[sample_index]
    metadata_path = Path(os.environ["DATA_ROOT"]) / "meta" / "train" / f"{token}.pkl"
    with metadata_path.open("rb") as stream:
        metadata = pickle.load(stream)
    for view in ("cam_l0", "cam_f0", "cam_r0"):
        for embedded_path in metadata["glo_images"][view]["image_paths"][3:12]:
            resolved_path = Path(resolve_navsim_data_path(embedded_path))
            if not resolved_path.is_file():
                raise FileNotFoundError(
                    f"NAVSIM image path remapping failed: {embedded_path} -> {resolved_path}"
                )
            checked_images += 1

print(
    "[training] preflight OK:",
    f"torch={torch.__version__}",
    f"accelerate={accelerate.__version__}",
    f"deepspeed={deepspeed.__version__}",
    f"flash_attn={flash_attn.__version__}",
    f"visible_devices={actual_devices}",
    f"train_samples={sample_count}",
    f"sampled_images={checked_images}",
)
PY

if [ "${TRAINING_PREFLIGHT_ONLY:-0}" = "1" ]; then
  echo "[training] TRAINING_PREFLIGHT_ONLY=1; launcher was not started"
  exit 0
fi

if [ -n "${NAVSIM_FEATURE_CACHE_ROOT:-}" ] && [ "$NAVSIM_STAGE_CACHE_TO_RAM" != "0" ]; then
  shared_feature_cache_root="$NAVSIM_FEATURE_CACHE_ROOT"
  runtime_feature_cache_root="$(
    bash "$project_root/tools/stage_navsim_cache.sh" \
      "$shared_feature_cache_root" \
      "$NAVSIM_RAM_CACHE_ROOT/$RUN_ID" \
      "${NAVSIM_CACHE_COMPONENTS:-wan,ppd}" \
      "${NAVSIM_RAM_CACHE_COMPONENTS:-wan,ppd}"
  )"
  export NAVSIM_SHARED_FEATURE_CACHE_ROOT="$shared_feature_cache_root"
  export NAVSIM_FEATURE_CACHE_ROOT="$runtime_feature_cache_root"
  echo "[training] runtime_feature_cache=$NAVSIM_FEATURE_CACHE_ROOT shared_source=$NAVSIM_SHARED_FEATURE_CACHE_ROOT"
fi

# The cached PPD payload replaces the ~41 GiB depth-pickle tree. Only the
# ~4 GiB raw metadata files are needed at runtime, and random opens of 103,288
# small CPFS files are much more expensive than serving them from tmpfs.
if [ "$NAVSIM_STAGE_METADATA_TO_RAM" != "0" ]; then
  case ",${NAVSIM_CACHE_COMPONENTS:-}," in
    *,ppd,*)
      shared_data_root="$DATA_ROOT"
      runtime_data_root="$(
        bash "$project_root/tools/stage_navsim_metadata.sh" \
          "$shared_data_root" \
          "$NAVSIM_RAM_DATA_ROOT/$RUN_ID" \
          "$project_root/train_meta.json"
      )"
      export NAVSIM_SHARED_DATA_ROOT="$shared_data_root"
      export DATA_ROOT="$runtime_data_root"
      echo "[training] runtime_data_root=$DATA_ROOT shared_data_root=$NAVSIM_SHARED_DATA_ROOT"
      ;;
    *)
      echo "[training] metadata RAM staging skipped because ppd cache is not enabled"
      ;;
  esac
fi

exec bash "$project_root/8-train.sh"
