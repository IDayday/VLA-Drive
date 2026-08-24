#!/usr/bin/env bash
# One-task, non-interactive DLC training for QwenPI-DrivoRSuprim.
#
# Baseline-frozen Qwen visual tower and tied LM-head/input embedding, with the
# Qwen language decoder trainable -> Q-Former + Flow-DiT -> DrivoR. DriveSuprim
# joint coarse/fine reranking is selected by the resolved training config.
# The paired production wrappers use 16 GPUs x micro-batch 4 x accumulation 1
# = 64. The common entrypoint remains topology-overridable for local preflight.

set -Eeuo pipefail

qds_phase=bootstrap
on_qds_error() {
  local status="$?"
  if (( BASH_SUBSHELL > 0 )); then
    return "$status"
  fi
  echo "[qds] failed phase=$qds_phase line=${BASH_LINENO[0]} status=$status" >&2
  exit "$status"
}
trap on_qds_error ERR

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Preserve explicit wrapper/one-shot values across env.local.sh. The local
# environment is required to use fallback assignments, but the launcher keeps
# this precedence mechanically true for the two fixed experiment selectors.
qds_config_was_set="${QDS_CONFIG_YAML+x}"
qds_config_one_shot="${QDS_CONFIG_YAML-}"
qds_expected_was_set="${QDS_EXPECT_DRIVESUPRIM+x}"
qds_expected_one_shot="${QDS_EXPECT_DRIVESUPRIM-}"
source "$project_root/load_env.sh"
if [[ "$qds_config_was_set" == "x" ]]; then
  export QDS_CONFIG_YAML="$qds_config_one_shot"
fi
if [[ "$qds_expected_was_set" == "x" ]]; then
  export QDS_EXPECT_DRIVESUPRIM="$qds_expected_one_shot"
fi
export DRIVEDREAMER_ROOT="$project_root"
cd "$project_root"

# Persist bootstrap/source-contract failures as well as distributed-training
# output. This matters for non-interactive DLC tasks, where stdout may otherwise
# disappear before the late training logger is installed.
export VLA_OUTPUT_ROOT="${VLA_OUTPUT_ROOT:-${NAVSIM_EXP_ROOT:-$project_root/navsim_exp}/qwenpi_drivor_suprim}"
export QDS_RUN_ID="${QDS_RUN_ID:-qwenpi-drivor-suprim-$(date +'%Y%m%d_%H%M%S')}"
if ! [[ "$QDS_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[qds] QDS_RUN_ID may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
fi
mkdir -p "$VLA_OUTPUT_ROOT/launcher_logs"
launcher_log="$VLA_OUTPUT_ROOT/launcher_logs/${QDS_RUN_ID}.log"
exec > >(tee -a "$launcher_log") 2>&1
qds_phase=source-contract
echo "[qds] launcher_log=$launcher_log phase=$qds_phase"

if [[ ! -f "$project_root/starVLA/model/framework/QwenPI_DrivoRSuprim.py" ]]; then
  echo "[qds] QwenPI-DrivoRSuprim is absent under project_root=$project_root" >&2
  exit 2
fi

export QDS_EXPECTED_BRANCH="${QDS_EXPECTED_BRANCH:-feature/ddp-drs-scene-2048}"
qds_git_metadata=direct
qds_git=(git -C "$project_root")
if ! git -C "$project_root" rev-parse --git-dir >/dev/null 2>&1; then
  if [[ ! -f "$project_root/.git" ]]; then
    echo "[qds] source checkout has no usable Git metadata: $project_root" >&2
    exit 2
  fi
  declared_git_dir="$(sed -n 's/^gitdir: //p' "$project_root/.git")"
  if [[ -z "$declared_git_dir" ]]; then
    echo "[qds] linked-worktree .git file has no gitdir entry" >&2
    exit 2
  fi
  linked_admin_name="$(basename -- "$declared_git_dir")"
  declared_common_checkout="${declared_git_dir%/.git/worktrees/*}"
  common_checkout_name="$(basename -- "$declared_common_checkout")"
  fallback_git_dir="$(dirname -- "$project_root")/$common_checkout_name/.git/worktrees/$linked_admin_name"
  if [[ ! -d "$fallback_git_dir" ]]; then
    echo "[qds] linked-worktree Git metadata is unavailable" >&2
    echo "[qds] declared=$declared_git_dir fallback=$fallback_git_dir" >&2
    exit 2
  fi
  qds_git=(
    env -u GIT_DIR -u GIT_WORK_TREE
    git --git-dir="$fallback_git_dir" --work-tree="$project_root"
  )
  if ! "${qds_git[@]}" rev-parse --git-dir >/dev/null 2>&1; then
    echo "[qds] linked-worktree fallback is not a valid Git directory: $fallback_git_dir" >&2
    exit 2
  fi
  qds_git_metadata=linked-worktree-fallback
fi
actual_branch="$("${qds_git[@]}" branch --show-current)"
source_commit="$("${qds_git[@]}" rev-parse HEAD)"
unset GIT_DIR GIT_WORK_TREE
if [[ "$actual_branch" != "$QDS_EXPECTED_BRANCH" ]]; then
  echo "[qds] wrong source worktree: $project_root" >&2
  echo "[qds] expected branch: $QDS_EXPECTED_BRANCH; actual branch: ${actual_branch:-DETACHED}" >&2
  exit 2
fi

required_value() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "[qds] required environment variable is empty: $name" >&2
    exit 2
  fi
}

required_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "[qds] missing required path: $path" >&2
    exit 2
  fi
}

export QDS_CONFIG_YAML="${QDS_CONFIG_YAML:-$project_root/starVLA/config/training/qwenpi_drivor_suprim.yaml}"
export QWEN_VLM_PATH="${QWEN_VLM_PATH:-${BASE_VLM:-}}"
export QDS_ASSET_ROOT="${QDS_ASSET_ROOT:-${DRIVEDREAMER_SHARED_ROOT:-$project_root}/ddp_drs_assets}"
export SUPRIM_VOCAB_PATH="${SUPRIM_VOCAB_PATH:-$QDS_ASSET_ROOT/drivesuprim/test_8192_kmeans.npy}"
export QDS_STATIC_SCORE_AGGREGATE="${QDS_STATIC_SCORE_AGGREGATE:-$QDS_ASSET_ROOT/drivesuprim/official_cache/traj_pdm_v2/ori/vocab_score_8192_navtrain_final/navtrain.pkl}"
export QDS_STATIC_SCORE_SHARDS="${QDS_STATIC_SCORE_SHARDS:-$QDS_ASSET_ROOT/drivesuprim/static_scores_navtrain_sharded}"
export QDS_SPLIT_STATIC_SCORE_CACHE="${QDS_SPLIT_STATIC_SCORE_CACHE:-1}"
export QDS_DOWNLOAD_STATIC_SCORE="${QDS_DOWNLOAD_STATIC_SCORE:-1}"
export QDS_DOWNLOAD_VOCAB="${QDS_DOWNLOAD_VOCAB:-1}"

export DATA_ROOT="${DATA_ROOT:-$project_root/navsim_dataset}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$project_root/navsim_dataset_raw}"
export NAVSIM_DATALIST_PATH="${NAVSIM_DATALIST_PATH:-$project_root/train_meta.json}"
export QDS_SPLIT="${QDS_SPLIT:-train}"
export QDS_NAVSIM_LOG_PATH="${QDS_NAVSIM_LOG_PATH:-$OPENSCENE_DATA_ROOT/navsim_logs/trainval}"
export QDS_NAVSIM_SENSOR_PATH="${QDS_NAVSIM_SENSOR_PATH:-${NAVSIM_TRAINVAL_SENSOR_ROOT:-$OPENSCENE_DATA_ROOT/sensor_blobs/trainval}}"
# The processed metadata contains preprocessing-machine image paths.  The
# dataset relocator reads NAVSIM_TRAINVAL_SENSOR_ROOT, so bind the
# capability-specific launcher path to that runtime contract before spawning
# any rank.  Without this assignment a worker can exit with FileNotFoundError
# and the remaining ranks only show a misleading PCCL/NCCL service warning.
export NAVSIM_TRAINVAL_SENSOR_ROOT="$QDS_NAVSIM_SENSOR_PATH"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$OPENSCENE_DATA_ROOT/maps}"
export NAVSIM_METRIC_CACHE_ROOT="${NAVSIM_METRIC_CACHE_ROOT:-${NAVSIM_EXP_ROOT:-$project_root/navsim_exp}/qds_metric_cache_navtrain}"
export QDS_BUILD_METRIC_CACHE="${QDS_BUILD_METRIC_CACHE:-auto}"

export VLA_MAX_TRAIN_STEPS="${VLA_MAX_TRAIN_STEPS:-100000}"
export VLA_WARMUP_STEPS="${VLA_WARMUP_STEPS:-5000}"
export VLA_SAVE_INTERVAL="${VLA_SAVE_INTERVAL:-5000}"
export VLA_EVAL_INTERVAL="${VLA_EVAL_INTERVAL:-200000}"
export VLA_LOG_INTERVAL="${VLA_LOG_INTERVAL:-20}"
export VLA_BATCH_SIZE="${VLA_BATCH_SIZE:-4}"
export QDS_TARGET_EFFECTIVE_BATCH="${QDS_TARGET_EFFECTIVE_BATCH:-32}"
export QDS_LOCAL_PROCESSES="${QDS_LOCAL_PROCESSES:-8}"
export VLA_RESUME_CKPT="${VLA_RESUME_CKPT:-none}"
export NAVSIM_METRIC_WORKERS="${NAVSIM_METRIC_WORKERS:-1}"
export QDS_METRIC_CACHE_WORKERS="${QDS_METRIC_CACHE_WORKERS:-16}"
export QDS_DEEPSPEED_CONFIG="${QDS_DEEPSPEED_CONFIG:-$project_root/starVLA/config/deepseeds/deepspeed_zero1.yaml}"
export QDS_MAIN_PROCESS_PORT="${QDS_MAIN_PROCESS_PORT:-29691}"
export QDS_CAPACITY_PROBE="${QDS_CAPACITY_PROBE:-0}"
export QDS_LAUNCHER_VALIDATE_ONLY="${QDS_LAUNCHER_VALIDATE_ONLY:-0}"
export QDS_LAUNCHER_PREFLIGHT_ONLY="${QDS_LAUNCHER_PREFLIGHT_ONLY:-0}"
export QDS_CURRICULUM_STATIC_ONLY_END="${QDS_CURRICULUM_STATIC_ONLY_END:-0.10}"
export QDS_CURRICULUM_DYNAMIC_RAMP_END="${QDS_CURRICULUM_DYNAMIC_RAMP_END:-0.20}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-qwenpi-drivor-suprim}"
export WANDB_ENTITY="${WANDB_ENTITY:-local}"
export NAVSIM_USE_FEATURE_CACHE=0
export NAVSIM_FEATURE_CACHE_ROOT=""
export NAVSIM_VIDEO_SOURCE="${NAVSIM_VIDEO_SOURCE:-images}"
export NAVSIM_NUM_WORKERS="${NAVSIM_NUM_WORKERS:-3}"
export NAVSIM_PREFETCH_FACTOR="${NAVSIM_PREFETCH_FACTOR:-2}"
export NAVSIM_PIN_MEMORY="${NAVSIM_PIN_MEMORY:-1}"
export NAVSIM_WORKER_THREADS="${NAVSIM_WORKER_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$project_root:$project_root/navsim:${PYTHONPATH:-}"

qds_phase=config-contract
required_path "$QDS_CONFIG_YAML"
QDS_CONFIG_YAML="$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$QDS_CONFIG_YAML")"
export QDS_CONFIG_YAML
mapfile -t qds_config_contract < <(python - "$QDS_CONFIG_YAML" <<'PY'
import sys

from omegaconf import OmegaConf

from starVLA.training.config_loader import load_training_config

config = load_training_config(sys.argv[1])
enabled = OmegaConf.select(
    config, "framework.hierarchical_scorer.joint.enabled", default=False
)
num_candidates = OmegaConf.select(
    config, "framework.hierarchical_scorer.dynamic.num_candidates"
)
candidate_chunk_size = OmegaConf.select(
    config, "framework.hierarchical_scorer.dynamic.candidate_chunk_size"
)
num_inference_timesteps = OmegaConf.select(
    config, "framework.action_model.num_inference_timesteps"
)
scene_gradient_checkpointing = OmegaConf.select(
    config, "framework.scene_encoder.use_gradient_checkpointing", default=False
)
metric_backend = str(
    OmegaConf.select(
        config, "framework.dynamic_metric_supervisor.backend", default="thread"
    )
)
if not isinstance(enabled, bool):
    raise TypeError(
        "framework.hierarchical_scorer.joint.enabled must be a YAML boolean"
    )
if not isinstance(scene_gradient_checkpointing, bool):
    raise TypeError(
        "framework.scene_encoder.use_gradient_checkpointing must be a YAML boolean"
    )
for name, value in (
    ("num_candidates", num_candidates),
    ("candidate_chunk_size", candidate_chunk_size),
    ("num_inference_timesteps", num_inference_timesteps),
):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
if candidate_chunk_size > num_candidates:
    raise ValueError("candidate_chunk_size must not exceed num_candidates")
if metric_backend not in {"local", "thread", "process"}:
    raise ValueError(f"unsupported NAVSIM metric backend: {metric_backend!r}")
print("1" if enabled else "0")
print(num_candidates)
print(candidate_chunk_size)
print(num_inference_timesteps)
print("1" if scene_gradient_checkpointing else "0")
print(metric_backend)
PY
)
if (( ${#qds_config_contract[@]} != 6 )); then
  echo "[qds] failed to resolve the training performance contract" >&2
  exit 2
fi
export QDS_DRIVESUPRIM_ENABLED="${qds_config_contract[0]}"
export QDS_DYNAMIC_CANDIDATES="${qds_config_contract[1]}"
export QDS_CANDIDATE_CHUNK_SIZE="${qds_config_contract[2]}"
export QDS_INFERENCE_TIMESTEPS="${qds_config_contract[3]}"
export QDS_SCENE_GRADIENT_CHECKPOINTING="${qds_config_contract[4]}"
export QDS_METRIC_BACKEND="${qds_config_contract[5]}"
qds_chunks_per_sample=$((
  (QDS_DYNAMIC_CANDIDATES + QDS_CANDIDATE_CHUNK_SIZE - 1)
  / QDS_CANDIDATE_CHUNK_SIZE
))
case "${QDS_EXPECT_DRIVESUPRIM:-}" in
  "") ;;
  0|1)
    if [[ "$QDS_EXPECT_DRIVESUPRIM" != "$QDS_DRIVESUPRIM_ENABLED" ]]; then
      echo "[qds] config/launcher DriveSuprim switch mismatch" >&2
      echo "[qds] expected=$QDS_EXPECT_DRIVESUPRIM resolved=$QDS_DRIVESUPRIM_ENABLED config=$QDS_CONFIG_YAML" >&2
      exit 2
    fi
    ;;
  *)
    echo "[qds] QDS_EXPECT_DRIVESUPRIM must be 0 or 1" >&2
    exit 2
    ;;
esac
for argument in "$@"; do
  case "$argument" in
    --config_yaml|--config_yaml=*|--framework.hierarchical_scorer.joint.enabled|--framework.hierarchical_scorer.joint.enabled=*)
      echo "[qds] the paired launcher fixes config_yaml and the DriveSuprim intervention" >&2
      exit 2
      ;;
  esac
done

for name in QDS_LOCAL_PROCESSES VLA_BATCH_SIZE QDS_TARGET_EFFECTIVE_BATCH \
  VLA_MAX_TRAIN_STEPS VLA_WARMUP_STEPS VLA_SAVE_INTERVAL VLA_EVAL_INTERVAL \
  VLA_LOG_INTERVAL NAVSIM_NUM_WORKERS NAVSIM_PREFETCH_FACTOR \
  NAVSIM_METRIC_WORKERS QDS_METRIC_CACHE_WORKERS; do
  if ! [[ "${!name}" =~ ^[0-9]+$ ]]; then
    echo "[qds] $name must be a non-negative integer, got ${!name}" >&2
    exit 2
  fi
done
if (( QDS_LOCAL_PROCESSES < 1 || VLA_BATCH_SIZE < 1 || QDS_TARGET_EFFECTIVE_BATCH < 1 )); then
  echo "[qds] processes, micro batch, and effective batch must be positive" >&2
  exit 2
fi
if (( NAVSIM_PREFETCH_FACTOR < 1 || NAVSIM_METRIC_WORKERS < 1 )); then
  echo "[qds] prefetch factor and metric workers must be positive" >&2
  exit 2
fi
micro_global=$((QDS_LOCAL_PROCESSES * VLA_BATCH_SIZE))
if (( QDS_TARGET_EFFECTIVE_BATCH % micro_global != 0 )); then
  echo "[qds] effective batch must be divisible by processes x micro batch" >&2
  exit 2
fi
export VLA_GRAD_ACCUM_STEPS=$((QDS_TARGET_EFFECTIVE_BATCH / micro_global))
qds_dataloader_workers=$((QDS_LOCAL_PROCESSES * NAVSIM_NUM_WORKERS))
qds_metric_workers=$((QDS_LOCAL_PROCESSES * NAVSIM_METRIC_WORKERS))
qds_nominal_cpu_slots=$((
  QDS_LOCAL_PROCESSES + qds_dataloader_workers + qds_metric_workers
))

if [[ "$QDS_LAUNCHER_VALIDATE_ONLY" == "1" ]]; then
  echo "[qds] project_root=$project_root"
  echo "[qds] source_ref=$actual_branch@$source_commit"
  echo "[qds] git_metadata=$qds_git_metadata"
  echo "[qds] config=$QDS_CONFIG_YAML"
  if [[ "$QDS_DRIVESUPRIM_ENABLED" == "1" ]]; then
    echo "[qds] drivesuprim_rerank=on drivesuprim_assets=required"
  else
    echo "[qds] drivesuprim_rerank=off drivesuprim_assets=skipped"
  fi
  echo "[qds] topology=processes:$QDS_LOCAL_PROCESSES"
  echo "[qds] batch=micro:$VLA_BATCH_SIZE accumulation:$VLA_GRAD_ACCUM_STEPS effective:$QDS_TARGET_EFFECTIVE_BATCH"
  echo "[qds] dynamic_sampler=candidates:$QDS_DYNAMIC_CANDIDATES chunk:$QDS_CANDIDATE_CHUNK_SIZE euler_steps:$QDS_INFERENCE_TIMESTEPS chunks_per_sample:$qds_chunks_per_sample"
  if [[ "$QDS_SCENE_GRADIENT_CHECKPOINTING" == "1" ]]; then
    echo "[qds] scene_gradient_checkpointing=on"
  else
    echo "[qds] scene_gradient_checkpointing=off"
  fi
  echo "[qds] cpu_parallelism=ranks:$QDS_LOCAL_PROCESSES dataloader_workers:$qds_dataloader_workers metric_workers:$qds_metric_workers nominal_slots:$qds_nominal_cpu_slots"
  echo "[qds] navsim_scoring=vectorized_pool async_overlap=flow+drivor+coarse backend:$QDS_METRIC_BACKEND workers_per_rank:$NAVSIM_METRIC_WORKERS"
  echo "[qds] formal_training=NOT_RUN"
  exit 0
elif [[ "$QDS_LAUNCHER_VALIDATE_ONLY" != "0" ]]; then
  echo "[qds] QDS_LAUNCHER_VALIDATE_ONLY must be 0 or 1" >&2
  exit 2
fi
if [[ "$QDS_LAUNCHER_PREFLIGHT_ONLY" != "0" && "$QDS_LAUNCHER_PREFLIGHT_ONLY" != "1" ]]; then
  echo "[qds] QDS_LAUNCHER_PREFLIGHT_ONLY must be 0 or 1" >&2
  exit 2
fi

required_names=(
  QWEN_VLM_PATH
  DATA_ROOT
  OPENSCENE_DATA_ROOT
  NAVSIM_DATALIST_PATH
  NAVSIM_METRIC_CACHE_ROOT
  VLA_OUTPUT_ROOT
)
if [[ "$QDS_DRIVESUPRIM_ENABLED" == "1" ]]; then
  required_names+=(SUPRIM_VOCAB_PATH)
fi
for name in "${required_names[@]}"; do
  required_value "$name"
done
run_dir="$VLA_OUTPUT_ROOT/$QDS_RUN_ID"
if [[ -e "$run_dir" && "$VLA_RESUME_CKPT" == "none" ]]; then
  echo "[qds] refusing to overwrite existing non-resume run: $run_dir" >&2
  echo "[qds] choose a new QDS_RUN_ID or set an explicit VLA_RESUME_CKPT" >&2
  exit 2
fi

qds_phase=asset-preflight
if [[ "$QDS_DRIVESUPRIM_ENABLED" == "1" ]]; then
  mkdir -p "$QDS_ASSET_ROOT/drivesuprim/official_cache"
fi
if [[ "$QDS_DRIVESUPRIM_ENABLED" == "1" && ! -f "$SUPRIM_VOCAB_PATH" ]]; then
  if [[ "$QDS_DOWNLOAD_VOCAB" != "1" ]]; then
    echo "[qds] DriveSuprim vocabulary is missing: $SUPRIM_VOCAB_PATH" >&2
    exit 2
  fi
  echo "[qds] downloading the official 8192-trajectory vocabulary"
  python - <<'PY'
import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

source = Path(hf_hub_download(
    repo_id="OpenDriveLab/WorldEngine",
    repo_type="dataset",
    filename="data/alg_engine/test_8192_kmeans.npy",
))
destination = Path(os.environ["SUPRIM_VOCAB_PATH"])
destination.parent.mkdir(parents=True, exist_ok=True)
if source.resolve() != destination.resolve():
    shutil.copy2(source, destination)
print(f"[qds] downloaded vocabulary: {destination}")
PY
fi
required_paths=(
  "$QDS_CONFIG_YAML"
  "$QWEN_VLM_PATH/config.json"
  "$QWEN_VLM_PATH/model.safetensors"
  "$DATA_ROOT/meta/$QDS_SPLIT"
  "$NAVSIM_DATALIST_PATH"
  "$QDS_DEEPSPEED_CONFIG"
  "$NAVSIM_TRAINVAL_SENSOR_ROOT"
)
if [[ "$QDS_DRIVESUPRIM_ENABLED" == "1" ]]; then
  required_paths+=("$SUPRIM_VOCAB_PATH")
fi
for path in "${required_paths[@]}"; do
  required_path "$path"
done

# Validate the exact image-path contract before allocating distributed copies
# of the model. This loads one lightweight record and uses the same resolver
# as NavSimDataset; it does not read image pixels or optional model assets.
python - "$NAVSIM_DATALIST_PATH" "$DATA_ROOT/meta/$QDS_SPLIT" <<'PY'
import json
import pickle
import sys
from pathlib import Path

from starVLA.dataloader.navsim_dataset import resolve_navsim_data_path

datalist_path = Path(sys.argv[1])
metadata_root = Path(sys.argv[2])
tokens = json.loads(datalist_path.read_text(encoding="utf-8"))
if not tokens:
    raise RuntimeError(f"NAVSIM datalist is empty: {datalist_path}")
token = tokens[0]
if not isinstance(token, str):
    raise TypeError(
        f"NAVSIM datalist entries must be token strings, got {type(token).__name__}"
    )
metadata_path = metadata_root / f"{token}.pkl"
if not metadata_path.is_file():
    raise FileNotFoundError(f"NAVSIM metadata is missing: {metadata_path}")
with metadata_path.open("rb") as handle:
    record = pickle.load(handle)
for view in ("cam_f0", "cam_l0", "cam_r0"):
    embedded = record["glo_images"][view]["image_paths"][3]
    resolved = Path(resolve_navsim_data_path(embedded))
    if not resolved.is_file():
        raise FileNotFoundError(
            "NAVSIM image relocation failed before distributed launch: "
            f"view={view} embedded={embedded!r} resolved={str(resolved)!r}. "
            "Set QDS_NAVSIM_SENSOR_PATH to the directory containing the "
            "per-log CAM_* folders."
        )
print(f"[qds] image-path preflight passed for token={token}")
PY

actual_devices="$(python -c 'import torch; print(torch.cuda.device_count())')"
if ! [[ "$actual_devices" =~ ^[1-9][0-9]*$ ]]; then
  echo "[qds] training requires at least one visible accelerator" >&2
  exit 2
fi
if (( QDS_LOCAL_PROCESSES > actual_devices )); then
  echo "[qds] requested $QDS_LOCAL_PROCESSES devices; $actual_devices are visible" >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((QDS_LOCAL_PROCESSES - 1)))"
fi

if [[ "$QDS_DRIVESUPRIM_ENABLED" == "1" ]]; then
  qds_phase=drivesuprim-assets
  if [[ -n "${SUPRIM_STATIC_SCORE_CACHE:-}" ]]; then
    required_path "$SUPRIM_STATIC_SCORE_CACHE"
  else
    if [[ ! -f "$QDS_STATIC_SCORE_AGGREGATE" ]]; then
      if [[ "$QDS_DOWNLOAD_STATIC_SCORE" != "1" ]]; then
        echo "[qds] official static score cache is missing: $QDS_STATIC_SCORE_AGGREGATE" >&2
        exit 2
      fi
      echo "[qds] downloading the official DriveSuprim navtrain static-score cache"
      export QDS_HF_LOCAL_DIR="$QDS_ASSET_ROOT/drivesuprim/official_cache"
      python - <<'PY'
import os
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="alkaid-2000/DriveSuprim",
    filename="traj_pdm_v2/ori/vocab_score_8192_navtrain_final/navtrain.pkl",
    local_dir=os.environ["QDS_HF_LOCAL_DIR"],
)
print(f"[qds] downloaded static score cache: {path}")
PY
    fi
    required_path "$QDS_STATIC_SCORE_AGGREGATE"

    if [[ "$QDS_SPLIT_STATIC_SCORE_CACHE" == "1" ]]; then
      if [[ ! -f "$QDS_STATIC_SCORE_SHARDS/_SUCCESS.json" ]]; then
        python "$project_root/tools/split_drivesuprim_static_scores.py" \
          --input "$QDS_STATIC_SCORE_AGGREGATE" \
          --output-root "$QDS_STATIC_SCORE_SHARDS" \
          --split "$QDS_SPLIT" \
          --vocab-size 8192
      fi
      export SUPRIM_STATIC_SCORE_CACHE="$QDS_STATIC_SCORE_SHARDS"
    else
      export SUPRIM_STATIC_SCORE_CACHE="$QDS_STATIC_SCORE_AGGREGATE"
    fi
  fi
else
  unset SUPRIM_STATIC_SCORE_CACHE
fi

qds_phase=navsim-metric-cache
metric_metadata_count=0
if [[ -d "$NAVSIM_METRIC_CACHE_ROOT/metadata" ]]; then
  metric_metadata_count="$(find "$NAVSIM_METRIC_CACHE_ROOT/metadata" -maxdepth 1 -type f -name '*.csv' | wc -l)"
fi
if (( metric_metadata_count == 0 )); then
  if [[ "$QDS_BUILD_METRIC_CACHE" == "0" ]]; then
    echo "[qds] NAVSIM metric cache is absent: $NAVSIM_METRIC_CACHE_ROOT" >&2
    exit 2
  fi
  for path in "$QDS_NAVSIM_LOG_PATH" "$QDS_NAVSIM_SENSOR_PATH" "$NUPLAN_MAPS_ROOT"; do
    required_path "$path"
  done
  echo "[qds] generating the official NAVSIM $QDS_SPLIT metric cache"
  python "$project_root/navsim/navsim/planning/script/run_metric_caching.py" \
    train_test_split=navtrain \
    navsim_log_path="$QDS_NAVSIM_LOG_PATH" \
    original_sensor_path="$QDS_NAVSIM_SENSOR_PATH" \
    metric_cache_path="$NAVSIM_METRIC_CACHE_ROOT" \
    worker=ray_distributed_no_torch \
    worker.threads_per_node="$QDS_METRIC_CACHE_WORKERS" \
    worker.use_distributed=false \
    gpu=false
fi

if [[ "$QDS_LAUNCHER_PREFLIGHT_ONLY" == "1" ]]; then
  echo "[qds] full preflight passed; formal_training=NOT_RUN"
  exit 0
fi

export TRITON_CACHE_DIR="${TRITON_LOCAL_CACHE_ROOT:-/tmp/qds-triton}/$QDS_RUN_ID"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_LOCAL_ROOT:-/tmp/qds-extensions}/$QDS_RUN_ID"
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

echo "[qds] project_root=$project_root"
echo "[qds] source_ref=$actual_branch@$source_commit config=$QDS_CONFIG_YAML"
echo "[qds] git_metadata=$qds_git_metadata"
echo "[qds] run_id=$QDS_RUN_ID devices=$CUDA_VISIBLE_DEVICES processes=$QDS_LOCAL_PROCESSES"
echo "[qds] batch=micro:$VLA_BATCH_SIZE accumulation:$VLA_GRAD_ACCUM_STEPS effective:$QDS_TARGET_EFFECTIVE_BATCH"
echo "[qds] cpu_parallelism=ranks:$QDS_LOCAL_PROCESSES dataloader_workers:$qds_dataloader_workers metric_workers:$qds_metric_workers nominal_slots:$qds_nominal_cpu_slots prefetch_factor:$NAVSIM_PREFETCH_FACTOR"
echo "[qds] navsim_scoring=vectorized_pool async_overlap=flow+drivor+coarse backend:$QDS_METRIC_BACKEND workers_per_rank:$NAVSIM_METRIC_WORKERS"
echo "[qds] action_horizon=8 flow_train_repeats=8 num_dynamic_candidates=$QDS_DYNAMIC_CANDIDATES candidate_chunk_size=$QDS_CANDIDATE_CHUNK_SIZE euler_steps=$QDS_INFERENCE_TIMESTEPS chunks_per_sample=$qds_chunks_per_sample"
if [[ "$QDS_SCENE_GRADIENT_CHECKPOINTING" == "1" ]]; then
  echo "[qds] scene_gradient_checkpointing=on"
else
  echo "[qds] scene_gradient_checkpointing=off"
fi
if [[ "$QDS_DRIVESUPRIM_ENABLED" == "1" ]]; then
  echo "[qds] drivesuprim_rerank=on vocab=$SUPRIM_VOCAB_PATH static_scores=$SUPRIM_STATIC_SCORE_CACHE"
else
  echo "[qds] drivesuprim_rerank=off drivesuprim_assets=skipped"
fi
echo "[qds] qwen=$QWEN_VLM_PATH data=$DATA_ROOT datalist=$NAVSIM_DATALIST_PATH"
echo "[qds] metric_cache=$NAVSIM_METRIC_CACHE_ROOT output=$run_dir"
echo "[qds] steps=max:$VLA_MAX_TRAIN_STEPS warmup:$VLA_WARMUP_STEPS save:$VLA_SAVE_INTERVAL eval:$VLA_EVAL_INTERVAL log:$VLA_LOG_INTERVAL"
echo "[qds] resume=$VLA_RESUME_CKPT log=$launcher_log"

launch=(
  accelerate launch
  --config_file "$QDS_DEEPSPEED_CONFIG"
  --num_machines 1
  --num_processes "$QDS_LOCAL_PROCESSES"
  --main_process_port "$QDS_MAIN_PROCESS_PORT"
  "$project_root/starVLA/training/train_starvla.py"
  --config_yaml "$QDS_CONFIG_YAML"
  --run_id "$QDS_RUN_ID"
  --run_root_dir "$VLA_OUTPUT_ROOT"
  --datasets.vla_data.datalist_path "$NAVSIM_DATALIST_PATH"
  --datasets.vla_data.data_root "$DATA_ROOT"
  --datasets.vla_data.split "$QDS_SPLIT"
  --datasets.vla_data.per_device_batch_size "$VLA_BATCH_SIZE"
  --trainer.gradient_accumulation_steps "$VLA_GRAD_ACCUM_STEPS"
  --trainer.max_train_steps "$VLA_MAX_TRAIN_STEPS"
  --trainer.num_warmup_steps "$VLA_WARMUP_STEPS"
  --trainer.save_interval "$VLA_SAVE_INTERVAL"
  --trainer.eval_interval "$VLA_EVAL_INTERVAL"
  --trainer.logging_frequency "$VLA_LOG_INTERVAL"
  --trainer.resume_ckpt "$VLA_RESUME_CKPT"
  --trainer.curriculum.static_only_end "$QDS_CURRICULUM_STATIC_ONLY_END"
  --trainer.curriculum.dynamic_ramp_end "$QDS_CURRICULUM_DYNAMIC_RAMP_END"
)
if [[ "$QDS_DRIVESUPRIM_ENABLED" == "1" ]]; then
  launch+=(--framework.static_score_store.split "$QDS_SPLIT")
fi
if [[ "$QDS_CAPACITY_PROBE" == "1" ]]; then
  launch+=(
    --trainer.capacity_probe true
    --framework.dynamic_metric_supervisor.backend stub
  )
fi
launch+=("$@")
qds_phase=distributed-training
"${launch[@]}"

echo "[qds] training complete: $VLA_OUTPUT_ROOT/$QDS_RUN_ID"
