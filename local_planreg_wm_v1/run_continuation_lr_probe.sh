#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=formal_runtime.sh
source "${SCRIPT_DIR}/formal_runtime.sh"
planreg_formal_runtime_setup "${REPO_ROOT}"

profile="${1:?Usage: $0 flat|targeted OUTPUT_DIR}"
output_dir="${2:?Usage: $0 flat|targeted OUTPUT_DIR}"
case "${profile}" in
  flat) agent_config=episode_drive_planreg_wm_continuation_probe_flat ;;
  targeted) agent_config=episode_drive_planreg_wm_continuation_probe_targeted ;;
  *) echo "Unknown continuation probe profile: ${profile}" >&2; exit 2 ;;
esac

checkpoint="${PLANREG_CONTINUATION_CHECKPOINT:-/mnt/project/DriveVLA-M0-formal-runs/formal_dual_init_gb128_asyncpdm_20260903/formal_base_init_wm_seed0/checkpoints/epoch_27_final.ckpt}"
vlm_path="${PLANREG_BASE_VLM_PATH:-/mnt/project/DriveVLA-M0-models/planreg-formal/InternVL3-2B-base-aligned}"
shared_init="${PLANREG_SHARED_INIT:-/mnt/project/DriveVLA-M0-models/planreg-formal/shared_planreg_init_seed0.pt}"
input_cache="${PLANREG_INPUT_CACHE:-/mnt/project/DriveVLA-M0-stage2/cache/feature_cache_navtrain_full}"
metric_cache="${PLANREG_TRAIN_METRIC_CACHE:-/mnt/project/DriveVLA-M0-stage2/cache/metric_cache_navtrain_full}"
navsim_root="${OPENSCENE_DATA_ROOT:-/mnt/project/DriveDreamer-Policy/navsim_raw}"
maps_root="${NUPLAN_MAPS_ROOT:-/mnt/navsim/maps}"
origin_step=21789
probe_steps="${PLANREG_PROBE_STEPS:-150}"
target_step=$((origin_step + probe_steps))
num_nodes="${PLANREG_PROBE_NUM_NODES:-2}"
if [[ "${num_nodes}" -eq 2 ]]; then
  accumulate_grad_batches=1
elif [[ "${num_nodes}" -eq 1 ]]; then
  accumulate_grad_batches=2
else
  echo "PLANREG_PROBE_NUM_NODES must be 1 or 2" >&2
  exit 2
fi

for required_file in "${checkpoint}" "${shared_init}" "${input_cache}/planreg_input_only_manifest.json"; do
  [[ -f "${required_file}" ]] || { echo "Missing probe input: ${required_file}" >&2; exit 2; }
done
for required_dir in "${vlm_path}" "${metric_cache}" "${navsim_root}/navsim_logs/trainval" "${navsim_root}/sensor_blobs/trainval" "${maps_root}"; do
  [[ -d "${required_dir}" ]] || { echo "Missing probe directory: ${required_dir}" >&2; exit 2; }
done
[[ ! -e "${output_dir}" ]] || { echo "Refusing to reuse probe output: ${output_dir}" >&2; exit 2; }

export PLANREG_BASE_VLM_PATH="${vlm_path}"
export PLANREG_FORMAL_VLM_PATH="${vlm_path}"
export PLANREG_INITIALIZATION_VARIANT=base
export PLANREG_VLM_CHECKPOINT_SHA256=0bd7cfa0ab23300304dd627abb09abbdc38748c8c8ff6c3209baf73a81fb421f
export PLANREG_VLM_CONFIG_SHA256=7617ced46ca592375c4d6394d9e7b0c8e3c2835f66b8d10c9977f0036caa9ec9
export PLANREG_SHARED_INIT="${shared_init}"
export PLANREG_INPUT_CACHE="${input_cache}"
export PLANREG_OUTPUT_DIR="${output_dir}"
export PLANREG_EXPERIMENT_NAME="continuation_lr_probe_${profile}"
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
  "agent=${agent_config}"
  seed=0
  "experiment_uid=continuation_lr_probe_${profile}"
  "experiment_name=continuation_lr_probe_${profile}"
  "output_dir=${output_dir}"
  "cache_path=${input_cache}"
  "navsim_log_path=${navsim_root}/navsim_logs/trainval"
  "sensor_blobs_path=${navsim_root}/sensor_blobs/trainval"
  formal_training.enabled=false
  auto_resume=false
  "train_ckpt_path=${checkpoint}"
  agent.batch_size=8
  agent.num_gpus=16
  agent.vlm_config.gradient_checkpointing=false
  agent.planning_registers.read_only_attention_backend=split_sdpa
  "agent.scheduler_args.continuation_optimizer_steps=${probe_steps}"
  dataloader.params.batch_size=8
  dataloader.params.num_workers=4
  dataloader.params.multiprocessing_context=forkserver
  trainer.params.devices=8
  "trainer.params.num_nodes=${num_nodes}"
  trainer.params.strategy=ddp
  "trainer.params.accumulate_grad_batches=${accumulate_grad_batches}"
  # The epoch loop state in an epoch-boundary checkpoint is already marked
  # complete. Use max_steps as the authoritative 150-step probe boundary.
  trainer.params.max_epochs=100
  "trainer.params.max_steps=${target_step}"
  trainer.params.limit_val_batches=0
  diagnostics.grad_log_interval=10
  diagnostics.register_log_interval=10
  hydra.output_subdir=null
)

mkdir -p "${output_dir}/run_metadata"
git -C "${REPO_ROOT}" rev-parse HEAD > "${output_dir}/run_metadata/git_commit.txt"
git -C "${REPO_ROOT}" status --short --branch > "${output_dir}/run_metadata/git_status.txt"
printf '%q ' "${command[@]}" > "${output_dir}/run_metadata/train_command.txt"
printf '\n' >> "${output_dir}/run_metadata/train_command.txt"
"${command[@]}" --cfg job --resolve > "${output_dir}/run_metadata/resolved_hydra_config.yaml"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'Continuation probe dry run: profile=%s nodes=%s target_step=%s output=%s\n' \
    "${profile}" "${num_nodes}" "${target_step}" "${output_dir}"
  exit 0
fi

if [[ "${num_nodes}" -eq 1 ]]; then
  "${command[@]}" 2>&1 | tee "${output_dir}/run_metadata/train.log"
  exit "${PIPESTATUS[0]}"
fi

peer="${PLANREG_PEER_HOST:-training-vla-zt2}"
master_addr="${PLANREG_MASTER_ADDR:-$(hostname -I | awk '{print $1}')}"
master_port="${PLANREG_MASTER_PORT:-29670}"
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
  echo "Continuation multi-node probe failed: local=${local_exit} remote=${remote_exit}" >&2
  exit 1
fi
