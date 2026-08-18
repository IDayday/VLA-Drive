#!/usr/bin/env bash
# Offline-only final-layer dense VGGT cache generation/validation/estimation.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

# Resolve path-bearing CLI options before capability checks. The Python tool
# still receives environment-derived options first and the original CLI last.
# This keeps the required precedence: CLI > one-shot env > env.local > default.
effective_vggt_repo="${VGGT_REPO:-}"
effective_vggt_checkpoint="${VGGT_CHECKPOINT:-}"
effective_cache_root="${NAVSIM_VGGT_DENSE_CACHE_ROOT:-}"
operation="build"
cli_arguments=("$@")
for ((argument_index = 0; argument_index < ${#cli_arguments[@]}; argument_index++)); do
  argument="${cli_arguments[$argument_index]}"
  case "$argument" in
    --vggt-repo)
      ((argument_index += 1))
      effective_vggt_repo="${cli_arguments[$argument_index]:-}"
      ;;
    --vggt-repo=*) effective_vggt_repo="${argument#*=}" ;;
    --vggt-checkpoint)
      ((argument_index += 1))
      effective_vggt_checkpoint="${cli_arguments[$argument_index]:-}"
      ;;
    --vggt-checkpoint=*) effective_vggt_checkpoint="${argument#*=}" ;;
    --cache-root)
      ((argument_index += 1))
      effective_cache_root="${cli_arguments[$argument_index]:-}"
      ;;
    --cache-root=*) effective_cache_root="${argument#*=}" ;;
    --estimate-only) operation="estimate" ;;
    --validate-only) operation="validate" ;;
  esac
done

if [[ "$operation" != "validate" && ! -d "$effective_vggt_repo" ]]; then
  echo "Missing local VGGT repository: $effective_vggt_repo (no download is attempted)" >&2
  exit 2
fi
if [[ "$operation" == "build" && ! -f "$effective_vggt_checkpoint" ]]; then
  echo "Missing local VGGT checkpoint: $effective_vggt_checkpoint (no download is attempted)" >&2
  exit 2
fi
if [[ "$operation" != "estimate" && -z "$effective_cache_root" ]]; then
  echo "Set NAVSIM_VGGT_DENSE_CACHE_ROOT or explicit --cache-root" >&2
  exit 2
fi

split="${SPLIT:-train}"
datalist="${NAVSIM_DATALIST_PATH:-$DRIVEDREAMER_ROOT/${split}_meta.json}"
num_machines="${NUM_MACHINES:-1}"
machine_rank="${MACHINE_RANK:-0}"
local_processes="${VGGT_DENSE_CACHE_NUM_PROCESSES:-${LOCAL_NUM_PROCESSES:-${NPROC_PER_NODE:-16}}}"
if [[ "$operation" != "build" ]]; then
  # Estimate and validation do not run the teacher model and must not scan the
  # same records once per accelerator process.
  num_machines=1
  machine_rank=0
  local_processes=1
fi
if ! [[ "$num_machines" =~ ^[1-9][0-9]*$ && "$local_processes" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_MACHINES and local cache process count must be positive integers" >&2
  exit 2
fi
if ! [[ "$machine_rank" =~ ^[0-9]+$ ]] || (( machine_rank >= num_machines )); then
  echo "MACHINE_RANK must be in [0, NUM_MACHINES)" >&2
  exit 2
fi
global_processes=$((num_machines * local_processes))
# 1.2 TiB aggregate address-space budget gives headroom over the measured
# ~925 GiB payload estimate. LMDB map_size is a maximum, not eager allocation.
default_map_size_gb=$(((1200 + global_processes - 1) / global_processes))
if (( default_map_size_gb < 32 )); then
  default_map_size_gb=32
fi
map_size_gb="${VGGT_DENSE_CACHE_MAP_SIZE_GB:-$default_map_size_gb}"
args=(
  --datalist-path "$datalist"
  --data-root "$DATA_ROOT"
  --split "$split"
  --sensor-root "$NAVSIM_TRAINVAL_SENSOR_ROOT"
  --cache-root "$NAVSIM_VGGT_DENSE_CACHE_ROOT"
  --vggt-repo "$VGGT_REPO"
  --vggt-checkpoint "$VGGT_CHECKPOINT"
  --views "${VGGT_DENSE_VIEWS:-cam_f0,cam_l0,cam_r0}"
  --frame-index "${VGGT_DENSE_FRAME_INDEX:-3}"
  --batch-size "${VGGT_DENSE_CACHE_BATCH_SIZE:-1}"
  --map-size-gb "$map_size_gb"
  --commit-interval "${VGGT_DENSE_CACHE_COMMIT_INTERVAL:-8}"
)
if [[ "${VGGT_DENSE_CACHE_FULL:-0}" != "1" && -n "${VGGT_DENSE_CACHE_MAX_SAMPLES:-}" ]]; then
  args+=(--max-samples "$VGGT_DENSE_CACHE_MAX_SAMPLES")
fi
if [[ "${VGGT_DENSE_CACHE_OVERWRITE:-0}" == "1" ]]; then
  args+=(--overwrite)
fi
# User CLI is last and therefore has highest precedence.
args+=("$@")

torchrun_args=(
  --nnodes "$num_machines"
  --nproc-per-node "$local_processes"
)
if (( num_machines == 1 )); then
  torchrun_args=(--standalone "${torchrun_args[@]}")
else
  : "${MASTER_ADDR:?Set MASTER_ADDR to the rank-0 host for multi-node cache generation}"
  torchrun_args+=(
    --node-rank "$machine_rank"
    --master-addr "$MASTER_ADDR"
    --master-port "${MASTER_PORT:-29671}"
  )
fi

echo "Dense VGGT cache topology: nodes=$num_machines local_ppus=$local_processes global_processes=$global_processes"
echo "Dense VGGT LMDB map limit: ${map_size_gb} GiB per rank"
set -x
torchrun "${torchrun_args[@]}" \
  tools/precompute_vggt_dense_cache.py "${args[@]}"
