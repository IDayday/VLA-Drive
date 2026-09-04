#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=formal_runtime.sh
source "${SCRIPT_DIR}/formal_runtime.sh"
planreg_formal_runtime_setup "${REPO_ROOT}"

origin_step=21789
steps_per_epoch=807
continuation_epochs=6
continuation_steps=$((steps_per_epoch * continuation_epochs))
target_step=$((origin_step + continuation_steps))
checkpoint="${PLANREG_CONTINUATION_CHECKPOINT:-/mnt/project/DriveVLA-M0-formal-runs/formal_dual_init_gb128_asyncpdm_20260903/formal_base_init_wm_seed0/checkpoints/epoch_27_final.ckpt}"
output_dir="${PLANREG_OUTPUT_DIR:-/mnt/project/DriveVLA-M0-formal-runs/formal_base_init_wm_continuation_epoch28_33_seed0_20260904}"
vlm_path="${PLANREG_BASE_VLM_PATH:-/mnt/project/DriveVLA-M0-models/planreg-formal/InternVL3-2B-base-aligned}"
shared_init="${PLANREG_SHARED_INIT:-/mnt/project/DriveVLA-M0-models/planreg-formal/shared_planreg_init_seed0.pt}"
input_cache="${PLANREG_INPUT_CACHE:-/mnt/project/DriveVLA-M0-stage2/cache/feature_cache_navtrain_full}"
metric_cache="${PLANREG_TRAIN_METRIC_CACHE:-/mnt/project/DriveVLA-M0-stage2/cache/metric_cache_navtrain_full}"
navsim_root="${OPENSCENE_DATA_ROOT:-/mnt/project/DriveDreamer-Policy/navsim_raw}"
maps_root="${NUPLAN_MAPS_ROOT:-/mnt/navsim/maps}"
peer="${PLANREG_PEER_HOST:-training-vla-zt2}"
master_addr="${PLANREG_MASTER_ADDR:-$(hostname -I | awk '{print $1}')}"
master_port="${PLANREG_MASTER_PORT:-29671}"

for required_file in "${checkpoint}" "${shared_init}" "${input_cache}/planreg_input_only_manifest.json"; do
  [[ -f "${required_file}" ]] || { echo "Missing continuation input: ${required_file}" >&2; exit 2; }
done
for required_dir in "${vlm_path}" "${metric_cache}" "${navsim_root}/navsim_logs/trainval" "${navsim_root}/sensor_blobs/trainval" "${maps_root}"; do
  [[ -d "${required_dir}" ]] || { echo "Missing continuation directory: ${required_dir}" >&2; exit 2; }
done

if [[ -e "${output_dir}" ]]; then
  if [[ -z "${RESUME_CHECKPOINT:-}" ]]; then
    echo "Refusing to reuse continuation output without explicit RESUME_CHECKPOINT: ${output_dir}" >&2
    exit 2
  fi
  resolved_resume="$(readlink -f "${RESUME_CHECKPOINT}")"
  resolved_output="$(readlink -f "${output_dir}")"
  case "${resolved_resume}" in
    "${resolved_output}"/checkpoints/*) checkpoint="${resolved_resume}" ;;
    *) echo "RESUME_CHECKPOINT must belong to this continuation run: ${resolved_output}/checkpoints" >&2; exit 2 ;;
  esac
fi

export PLANREG_BASE_VLM_PATH="${vlm_path}"
export PLANREG_FORMAL_VLM_PATH="${vlm_path}"
export PLANREG_INITIALIZATION_VARIANT=base
export PLANREG_VLM_CHECKPOINT_SHA256=0bd7cfa0ab23300304dd627abb09abbdc38748c8c8ff6c3209baf73a81fb421f
export PLANREG_VLM_CONFIG_SHA256=7617ced46ca592375c4d6394d9e7b0c8e3c2835f66b8d10c9977f0036caa9ec9
export PLANREG_SHARED_INIT="${shared_init}"
export PLANREG_INPUT_CACHE="${input_cache}"
export PLANREG_OUTPUT_DIR="${output_dir}"
export PLANREG_EXPERIMENT_NAME=formal_base_init_wm_continuation_epoch28_33_seed0
export NAVSIM_TRAIN_METRIC_CACHE="${metric_cache}"
export OPENSCENE_DATA_ROOT="${navsim_root}"
export NUPLAN_MAPS_ROOT="${maps_root}"
export RESUME_CHECKPOINT="${checkpoint}"
export DRIVEVLA_SCORE_RAY=0
export DRIVEVLA_SCORE_PROCESSES=8
export DRIVEVLA_SCORE_PARTITIONS=2
export DRIVEVLA_SCORE_START_METHOD=forkserver
export DRIVEVLA_BIND_RANK_CPUS=1
export DRIVEVLA_SYNC_TRAIN_METRICS=0
export DRIVEVLA_TRAIN_LOG_INTERVAL=1
export PLANREG_FORMAL_TIMING=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HOME="${HF_HOME:-/mnt/project/.cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/project/.cache/matplotlib}"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

command=(
  "${PLANREG_FORMAL_PYTHON_BIN}"
  "${REPO_ROOT}/navsim/planning/script/run_training_full.py"
  --config-name=formal_planreg_wm_training
  agent=episode_drive_planreg_wm_continuation_formal
  seed=0
  experiment_uid=formal_base_init_wm_continuation_epoch28_33_seed0
  experiment_name=formal_base_init_wm_continuation_epoch28_33_seed0
  "output_dir=${output_dir}"
  "cache_path=${input_cache}"
  "navsim_log_path=${navsim_root}/navsim_logs/trainval"
  "sensor_blobs_path=${navsim_root}/sensor_blobs/trainval"
  formal_training.enabled=false
  auto_resume=false
  # Quote the Hydra value itself because Lightning's default checkpoint
  # filenames contain '=' (for example epoch=27-step=22597.ckpt).
  "train_ckpt_path='${checkpoint}'"
  agent.batch_size=8
  agent.num_gpus=16
  agent.vlm_config.gradient_checkpointing=false
  agent.planning_registers.read_only_attention_backend=split_sdpa
  dataloader.params.batch_size=8
  dataloader.params.num_workers=4
  dataloader.params.multiprocessing_context=forkserver
  trainer.params.devices=8
  trainer.params.num_nodes=2
  trainer.params.strategy=ddp
  trainer.params.accumulate_grad_batches=1
  trainer.params.max_epochs=100
  "trainer.params.max_steps=${target_step}"
  trainer.params.limit_val_batches=0
  diagnostics.grad_log_interval=100
  diagnostics.register_log_interval=100
  hydra.output_subdir=null
)

mkdir -p "${output_dir}/run_metadata"
git -C "${REPO_ROOT}" rev-parse HEAD > "${output_dir}/run_metadata/git_commit.txt"
git -C "${REPO_ROOT}" status --short --branch > "${output_dir}/run_metadata/git_status.txt"
printf '%q ' "${command[@]}" > "${output_dir}/run_metadata/train_command.txt"
printf '\n' >> "${output_dir}/run_metadata/train_command.txt"
"${command[@]}" --cfg job --resolve > "${output_dir}/run_metadata/resolved_hydra_config.yaml"
"${PLANREG_FORMAL_PYTHON_BIN}" - <<PY
import hashlib, json
from pathlib import Path
checkpoint = Path(${checkpoint@Q}).resolve()
digest = hashlib.sha256()
with checkpoint.open("rb") as stream:
    for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
        digest.update(chunk)
metadata = {
    "source_checkpoint": str(checkpoint),
    "source_checkpoint_sha256": digest.hexdigest(),
    "origin_optimizer_step": ${origin_step},
    "steps_per_dataset_epoch": ${steps_per_epoch},
    "continuation_dataset_epochs": ${continuation_epochs},
    "continuation_optimizer_steps": ${continuation_steps},
    "target_optimizer_step": ${target_step},
    "global_batch_size": 128,
    "topology": "2 nodes x 8 GPUs x 8 samples/GPU",
    "world_model_enabled": True,
    "future_mode": "correct",
}
Path(${output_dir@Q}, "run_metadata", "continuation_protocol.json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
)
PY
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'Formal continuation dry run: target_step=%s output=%s\n' "${target_step}" "${output_dir}"
  exit 0
fi

program_args=("${command[@]:1}")
torchrun=(
  "${PLANREG_FORMAL_PYTHON_BIN}" -m torch.distributed.run
  --nnodes=2 --nproc-per-node=8
  "--master-addr=${master_addr}" "--master-port=${master_port}"
)
shared_environment=(
  "PLANREG_BASE_VLM_PATH=${PLANREG_BASE_VLM_PATH}"
  "PLANREG_FORMAL_VLM_PATH=${PLANREG_FORMAL_VLM_PATH}"
  "PLANREG_INITIALIZATION_VARIANT=${PLANREG_INITIALIZATION_VARIANT}"
  "PLANREG_VLM_CHECKPOINT_SHA256=${PLANREG_VLM_CHECKPOINT_SHA256}"
  "PLANREG_VLM_CONFIG_SHA256=${PLANREG_VLM_CONFIG_SHA256}"
  "PLANREG_SHARED_INIT=${PLANREG_SHARED_INIT}"
  "PLANREG_INPUT_CACHE=${PLANREG_INPUT_CACHE}"
  "PLANREG_OUTPUT_DIR=${PLANREG_OUTPUT_DIR}"
  "PLANREG_EXPERIMENT_NAME=${PLANREG_EXPERIMENT_NAME}"
  "NAVSIM_TRAIN_METRIC_CACHE=${NAVSIM_TRAIN_METRIC_CACHE}"
  "OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT}"
  "NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT}"
  "RESUME_CHECKPOINT=${RESUME_CHECKPOINT}"
  DRIVEVLA_SCORE_RAY=0 DRIVEVLA_SCORE_PROCESSES=8
  DRIVEVLA_SCORE_PARTITIONS=2 DRIVEVLA_SCORE_START_METHOD=forkserver
  DRIVEVLA_BIND_RANK_CPUS=1 DRIVEVLA_SYNC_TRAIN_METRICS=0
  DRIVEVLA_TRAIN_LOG_INTERVAL=1 PLANREG_FORMAL_TIMING=0
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  "PYTHONPATH=${PLANREG_FORMAL_PYTHONPATH}" PYTHONNOUSERSITE=1
  "HF_HOME=${HF_HOME}" "MPLCONFIGDIR=${MPLCONFIGDIR}"
  TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
)
remote_array=(
  env "${shared_environment[@]}" "${torchrun[@]}" --node-rank=1
  "${program_args[@]}"
)
printf -v remote_command '%q ' "${remote_array[@]}"
ssh -o BatchMode=yes "${peer}" "${remote_command}" \
  > "${output_dir}/run_metadata/node1.log" 2>&1 &
remote_pid=$!
sleep 3
local_exit=0
remote_exit=0
env "${shared_environment[@]}" "${torchrun[@]}" --node-rank=0 \
  "${program_args[@]}" 2>&1 | tee "${output_dir}/run_metadata/node0.log" \
  || local_exit=${PIPESTATUS[0]}
wait "${remote_pid}" || remote_exit=$?
if [[ "${local_exit}" -ne 0 || "${remote_exit}" -ne 0 ]]; then
  echo "Formal multi-node continuation failed: local=${local_exit} remote=${remote_exit}" >&2
  exit 1
fi

final_checkpoint="${output_dir}/checkpoints/last.ckpt"
[[ -f "${final_checkpoint}" ]] || { echo "Missing final continuation checkpoint: ${final_checkpoint}" >&2; exit 1; }
ln -f "${final_checkpoint}" "${output_dir}/checkpoints/continuation_epoch33_final.ckpt"
printf 'Formal continuation completed: %s\n' "${final_checkpoint}"
