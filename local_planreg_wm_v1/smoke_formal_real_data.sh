#!/usr/bin/env bash
set -euo pipefail

seed="${1:-0}"
if ! [[ "${seed}" =~ ^[0-9]+$ ]]; then
  echo "Seed must be a non-negative integer" >&2
  exit 2
fi
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
# shellcheck source=../load_env.sh
source "${repo_root}/load_env.sh"
python_bin="${PYTHON_BIN:-/mnt/project/DriveVLA-M0-env/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  python_bin=/mnt/project/DriveVLA-M0-env/bin/python
fi
base_vlm="${PLANREG_BASE_VLM_PATH:-/mnt/project/DriveVLA-M0-models/planreg-formal/InternVL3-2B-base-aligned}"
shared_init="${PLANREG_SHARED_INIT:-/mnt/project/DriveVLA-M0-models/planreg-formal/shared_planreg_init_seed${seed}.pt}"
metric_cache="${PLANREG_TRAIN_METRIC_CACHE:-/mnt/project/DriveVLA-M0-stage2/cache/metric_cache_navtrain_full}"
navsim_data="${OPENSCENE_DATA_ROOT:-/mnt/project/DriveDreamer-Policy/navsim_raw}"
output="${PLANREG_FORMAL_REAL_SMOKE_ROOT:-/mnt/project/DriveVLA-M0-formal-runs/real_data_smoke_seed${seed}}"
audit="${PLANREG_VLM_AUDIT_REPORT:-${repo_root}/reports/planreg_wm_v1/formal_vlm_initialization_audit.json}"

for path in "${python_bin}" "${shared_init}" "${audit}"; do
  [[ -e "${path}" ]] || { echo "Missing formal smoke prerequisite: ${path}" >&2; exit 2; }
done
for path in "${base_vlm}" "${metric_cache}" "${navsim_data}/navsim_logs/trainval" "${navsim_data}/sensor_blobs/trainval" "${NUPLAN_MAPS_ROOT:-/mnt/navsim/maps}"; do
  [[ -d "${path}" ]] || { echo "Missing formal smoke directory: ${path}" >&2; exit 2; }
done
[[ ! -e "${output}" ]] || { echo "Refusing to overwrite ${output}" >&2; exit 2; }

checkpoint_sha="$(jq -er '.base.checkpoint_sha256' "${audit}")"
config_sha="$(jq -er '.base.config_sha256' "${audit}")"
export PLANREG_BASE_VLM_PATH="${base_vlm}"
export PLANREG_FORMAL_VLM_PATH="${base_vlm}"
export PLANREG_INITIALIZATION_VARIANT=base
export PLANREG_VLM_CHECKPOINT_SHA256="${checkpoint_sha}"
export PLANREG_VLM_CONFIG_SHA256="${config_sha}"
export PLANREG_SHARED_INIT="${shared_init}"
export PLANREG_INPUT_CACHE="${PLANREG_INPUT_CACHE:-/mnt/project/DriveVLA-M0-stage2/cache/feature_cache_navtrain_full}"
export PLANREG_OUTPUT_DIR="${output}/training"
export PLANREG_EXPERIMENT_NAME="formal_real_data_smoke_seed${seed}"
export NAVSIM_TRAIN_METRIC_CACHE="${metric_cache}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/navsim/maps}"
export DRIVEVLA_SCORE_RAY=0
export DRIVEVLA_SCORE_PROCESSES="${PLANREG_SMOKE_SCORER_PROCESSES:-4}"
export DRIVEVLA_SCORE_PARTITIONS=8
export DRIVEVLA_SCORE_START_METHOD=forkserver
export CUDA_VISIBLE_DEVICES="${PLANREG_SMOKE_GPU:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

mkdir -p "${output}/training/run_metadata"
command=(
  "${python_bin}" "${repo_root}/navsim/planning/script/run_training_full.py"
  --config-name=formal_planreg_wm_training
  agent=episode_drive_planreg_wm_formal_base
  train_test_split=navtrain
  "train_test_split.scene_filter.max_scenes=16"
  "navsim_log_path=${navsim_data}/navsim_logs/trainval"
  "sensor_blobs_path=${navsim_data}/sensor_blobs/trainval"
  "output_dir=${output}/training"
  "experiment_name=formal_real_data_smoke_seed${seed}"
  "seed=${seed}"
  use_cache_without_dataset=false
  cache_path=null
  force_cache_computation=false
  load_image_path=true
  data_protocol.include_val_in_train=false
  formal_training.enabled=false
  agent.world_model.require_all_horizons_valid=true
  agent.batch_size=2
  agent.num_gpus=1
  dataloader.params.batch_size=2
  dataloader.params.num_workers=0
  dataloader.params.persistent_workers=false
  ~dataloader.params.prefetch_factor
  trainer.params.devices=1
  trainer.params.num_nodes=1
  trainer.params.strategy=auto
  trainer.params.max_epochs=1
  trainer.params.max_steps=-1
  trainer.params.limit_train_batches=2
  trainer.params.limit_val_batches=1
  diagnostics.require_finite_loss_and_gradients=true
  hydra.output_subdir=null
)
printf '%q ' "${command[@]}" > "${output}/training/run_metadata/command.txt"
printf '\n' >> "${output}/training/run_metadata/command.txt"
"${command[@]}" --cfg job --resolve > "${output}/training/run_metadata/resolved_hydra_config.yaml"
"${command[@]}" 2>&1 | tee "${output}/training/run_metadata/train.log"

mapfile -t checkpoints < <(find "${output}/training" -name last.ckpt -print)
if [[ "${#checkpoints[@]}" -ne 1 ]]; then
  echo "Expected one smoke last.ckpt, found ${#checkpoints[@]}" >&2
  exit 2
fi
student="${output}/formal_real_data_smoke_student.ckpt"
"${python_bin}" "${repo_root}/scripts/export_planreg_student_checkpoint.py" \
  "${checkpoints[0]}" "${student}" \
  --resolved-config "${output}/training/run_metadata/resolved_hydra_config.yaml" \
  | tee "${output}/student_export.json"
"${python_bin}" "${repo_root}/scripts/smoke_planreg_student_inference.py" \
  "${student}" --formal --vlm-path "${base_vlm}" \
  --navsim-log-path "${navsim_data}/navsim_logs/trainval" \
  --sensor-blobs-path "${navsim_data}/sensor_blobs/trainval" \
  | tee "${output}/student_inference.json"
