#!/usr/bin/env bash
set -euo pipefail

PLANREG_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLANREG_REPO_ROOT="$(cd "${PLANREG_SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../load_env.sh
source "${PLANREG_REPO_ROOT}/load_env.sh"

_formal_json_value() {
  jq -er "$2" "$1"
}

_formal_require_file() {
  [[ -f "$1" ]] || { echo "Required formal artifact is missing: $1" >&2; return 2; }
}

_formal_require_directory() {
  [[ -d "$1" ]] || { echo "Required formal directory is missing: $1" >&2; return 2; }
}

_formal_assert_idle_gpus() {
  local host_label="$1"
  local expected="$2"
  local count
  count="$(nvidia-smi -L | wc -l)"
  if [[ "${count}" -lt "${expected}" ]]; then
    echo "${host_label} exposes ${count} GPUs; ${expected} are required" >&2
    return 2
  fi
  local busy
  busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)"
  if [[ -n "${busy}" ]]; then
    echo "${host_label} has active GPU compute processes; refusing to preempt: ${busy}" >&2
    return 2
  fi
}

formal_launch() {
  if [[ $# -ne 2 ]]; then
    echo "formal_launch VARIANT SEED" >&2
    return 2
  fi
  local variant="$1"
  local seed="$2"
  if [[ "${variant}" != "base" && "${variant}" != "driving_vqa" ]]; then
    echo "Unknown formal initialization variant: ${variant}" >&2
    return 2
  fi
  if ! [[ "${seed}" =~ ^[0-9]+$ ]]; then
    echo "Formal seed must be a non-negative integer: ${seed}" >&2
    return 2
  fi
  : "${PLANREG_LAYOUT_LOCK:?Set PLANREG_LAYOUT_LOCK to formal_training_layout_lock.json}"
  : "${PLANREG_SHARED_INIT:?Set PLANREG_SHARED_INIT to shared_planreg_init_seed${seed}.pt}"

  local python_bin="${PYTHON_BIN:-/mnt/project/DriveVLA-M0-env/bin/python}"
  if [[ ! -x "${python_bin}" ]]; then
    python_bin=/mnt/project/DriveVLA-M0-env/bin/python
  fi
  local vlm_audit="${PLANREG_VLM_AUDIT_REPORT:-${PLANREG_REPO_ROOT}/reports/planreg_wm_v1/formal_vlm_initialization_audit.json}"
  local input_cache="${PLANREG_INPUT_CACHE:-/mnt/project/DriveVLA-M0-stage2/cache/feature_cache_navtrain_full}"
  local metric_cache="${PLANREG_TRAIN_METRIC_CACHE:-/mnt/project/DriveVLA-M0-stage2/cache/metric_cache_navtrain_full}"
  local navsim_data="${OPENSCENE_DATA_ROOT:-/mnt/project/DriveDreamer-Policy/navsim_raw}"
  local maps_root="${NUPLAN_MAPS_ROOT:-/mnt/navsim/maps}"
  local run_root="${PLANREG_FORMAL_RUN_ROOT:-/mnt/project/DriveVLA-M0-formal-runs/training}"

  _formal_require_file "${PLANREG_LAYOUT_LOCK}"
  _formal_require_file "${PLANREG_SHARED_INIT}"
  _formal_require_file "${vlm_audit}"
  _formal_require_file "${input_cache}/planreg_input_only_manifest.json"
  _formal_require_directory "${metric_cache}"
  _formal_require_directory "${navsim_data}/navsim_logs/trainval"
  _formal_require_directory "${navsim_data}/sensor_blobs/trainval"
  _formal_require_directory "${maps_root}"

  local base_vlm vqa_vlm
  base_vlm="${PLANREG_BASE_VLM_PATH:-$(_formal_json_value "${vlm_audit}" '.base.checkpoint_path')}"
  vqa_vlm="${PLANREG_VQA_VLM_PATH:-$(_formal_json_value "${vlm_audit}" '.driving_vqa.checkpoint_path')}"
  if [[ "${variant}" == "base" && -z "${PLANREG_BASE_VLM_PATH:-}" ]]; then
    echo "Set PLANREG_BASE_VLM_PATH for the Base formal run" >&2
    return 2
  fi
  if [[ "${variant}" == "driving_vqa" && -z "${PLANREG_VQA_VLM_PATH:-}" ]]; then
    echo "Set PLANREG_VQA_VLM_PATH for the Driving-VQA formal run" >&2
    return 2
  fi
  _formal_require_directory "${base_vlm}"
  _formal_require_directory "${vqa_vlm}"

  local base_checkpoint_sha base_config_sha vqa_checkpoint_sha vqa_config_sha
  base_checkpoint_sha="$(_formal_json_value "${vlm_audit}" '.base.checkpoint_sha256')"
  base_config_sha="$(_formal_json_value "${vlm_audit}" '.base.config_sha256')"
  vqa_checkpoint_sha="$(_formal_json_value "${vlm_audit}" '.driving_vqa.checkpoint_sha256')"
  vqa_config_sha="$(_formal_json_value "${vlm_audit}" '.driving_vqa.config_sha256')"

  local selected_layout gpu_count per_gpu_batch global_batch num_nodes devices_per_node
  local num_workers scorer_processes
  selected_layout="$(_formal_json_value "${PLANREG_LAYOUT_LOCK}" '.selected_layout')"
  gpu_count="$(_formal_json_value "${PLANREG_LAYOUT_LOCK}" '.gpu_count')"
  per_gpu_batch="$(_formal_json_value "${PLANREG_LAYOUT_LOCK}" '.per_gpu_batch_size')"
  global_batch="$(_formal_json_value "${PLANREG_LAYOUT_LOCK}" '.global_batch_size')"
  num_nodes="$(_formal_json_value "${PLANREG_LAYOUT_LOCK}" '.num_nodes')"
  devices_per_node="$(_formal_json_value "${PLANREG_LAYOUT_LOCK}" '.devices_per_node')"
  num_workers="$(_formal_json_value "${PLANREG_LAYOUT_LOCK}" '.num_workers_per_rank')"
  scorer_processes="$(_formal_json_value "${PLANREG_LAYOUT_LOCK}" '.scorer_processes_per_rank')"
  if [[ "$((gpu_count * per_gpu_batch))" -ne "${global_batch}" ]]; then
    echo "Layout lock has inconsistent global batch" >&2
    return 2
  fi

  local base_name="formal_base_init_wm_seed${seed}"
  local vqa_name="formal_vqa_init_wm_seed${seed}"
  local base_output="${run_root}/${base_name}"
  local vqa_output="${run_root}/${vqa_name}"
  local experiment_name output_dir vlm_path checkpoint_sha config_sha agent_config
  if [[ "${variant}" == "base" ]]; then
    experiment_name="${base_name}"
    output_dir="${base_output}"
    vlm_path="${base_vlm}"
    checkpoint_sha="${base_checkpoint_sha}"
    config_sha="${base_config_sha}"
    agent_config=episode_drive_planreg_wm_formal_base
  else
    experiment_name="${vqa_name}"
    output_dir="${vqa_output}"
    vlm_path="${vqa_vlm}"
    checkpoint_sha="${vqa_checkpoint_sha}"
    config_sha="${vqa_config_sha}"
    agent_config=episode_drive_planreg_wm_formal_vqa
  fi

  if [[ -z "${RESUME_CHECKPOINT:-}" ]]; then
    if [[ -e "${output_dir}" ]]; then
      echo "Refusing to reuse formal output directory without RESUME_CHECKPOINT: ${output_dir}" >&2
      return 2
    fi
  elif [[ ! -d "${output_dir}" ]]; then
    echo "Formal resume output directory does not exist: ${output_dir}" >&2
    return 2
  fi

  export PLANREG_BASE_VLM_PATH="${base_vlm}"
  export PLANREG_VQA_VLM_PATH="${vqa_vlm}"
  export PLANREG_FORMAL_VLM_PATH="${vlm_path}"
  export PLANREG_INITIALIZATION_VARIANT="${variant}"
  export PLANREG_VLM_CHECKPOINT_SHA256="${checkpoint_sha}"
  export PLANREG_VLM_CONFIG_SHA256="${config_sha}"
  export PLANREG_INPUT_CACHE="${input_cache}"
  export PLANREG_OUTPUT_DIR="${output_dir}"
  export PLANREG_EXPERIMENT_NAME="${experiment_name}"
  export NAVSIM_TRAIN_METRIC_CACHE="${metric_cache}"
  export OPENSCENE_DATA_ROOT="${navsim_data}"
  export NUPLAN_MAPS_ROOT="${maps_root}"
  export DRIVEVLA_SCORE_RAY=0
  export DRIVEVLA_SCORE_PROCESSES="${scorer_processes}"
  export DRIVEVLA_SCORE_PARTITIONS=8
  export DRIVEVLA_SCORE_START_METHOD=forkserver
  export DRIVEVLA_BIND_RANK_CPUS=1
  export DRIVEVLA_SYNC_TRAIN_METRICS=0
  export DRIVEVLA_TRAIN_LOG_INTERVAL="${DRIVEVLA_TRAIN_LOG_INTERVAL:-10}"
  export PLANREG_FORMAL_TIMING=0
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

  local common_hydra_args=(
    seed="${seed}"
    "cache_path=${input_cache}"
    "navsim_log_path=${navsim_data}/navsim_logs/trainval"
    "sensor_blobs_path=${navsim_data}/sensor_blobs/trainval"
    "agent.batch_size=${per_gpu_batch}"
    "agent.num_gpus=${gpu_count}"
    "dataloader.params.batch_size=${per_gpu_batch}"
    "dataloader.params.num_workers=${num_workers}"
    dataloader.params.multiprocessing_context=forkserver
    "trainer.params.devices=${devices_per_node}"
    "trainer.params.num_nodes=${num_nodes}"
    trainer.params.strategy=ddp
    hydra.output_subdir=null
  )

  # Compose both launch configs under one file lock, then prove that only the
  # declared VLM identity fields differ. Both launchers use this same path.
  local pair_root="${run_root}/formal_config_pair_seed${seed}_${selected_layout}"
  mkdir -p "${pair_root}"
  exec 9>"${pair_root}/.compose.lock"
  flock 9
  local base_resolved="${pair_root}/formal_base_resolved.yaml"
  local vqa_resolved="${pair_root}/formal_vqa_resolved.yaml"
  local pair_audit="${pair_root}/formal_config_pair_audit.json"
  env \
    PLANREG_INITIALIZATION_VARIANT=base \
    PLANREG_FORMAL_VLM_PATH="${base_vlm}" \
    PLANREG_VLM_CHECKPOINT_SHA256="${base_checkpoint_sha}" \
    PLANREG_VLM_CONFIG_SHA256="${base_config_sha}" \
    PLANREG_OUTPUT_DIR="${base_output}" \
    PLANREG_EXPERIMENT_NAME="${base_name}" \
    "${python_bin}" "${PLANREG_REPO_ROOT}/navsim/planning/script/run_training_full.py" \
    --config-name=formal_planreg_wm_training \
    agent=episode_drive_planreg_wm_formal_base "${common_hydra_args[@]}" \
    "experiment_name=${base_name}" "output_dir=${base_output}" --cfg job --resolve \
    > "${base_resolved}.tmp"
  mv "${base_resolved}.tmp" "${base_resolved}"
  env \
    PLANREG_INITIALIZATION_VARIANT=driving_vqa \
    PLANREG_FORMAL_VLM_PATH="${vqa_vlm}" \
    PLANREG_VLM_CHECKPOINT_SHA256="${vqa_checkpoint_sha}" \
    PLANREG_VLM_CONFIG_SHA256="${vqa_config_sha}" \
    PLANREG_OUTPUT_DIR="${vqa_output}" \
    PLANREG_EXPERIMENT_NAME="${vqa_name}" \
    "${python_bin}" "${PLANREG_REPO_ROOT}/navsim/planning/script/run_training_full.py" \
    --config-name=formal_planreg_wm_training \
    agent=episode_drive_planreg_wm_formal_vqa "${common_hydra_args[@]}" \
    "experiment_name=${vqa_name}" "output_dir=${vqa_output}" --cfg job --resolve \
    > "${vqa_resolved}.tmp"
  mv "${vqa_resolved}.tmp" "${vqa_resolved}"
  "${python_bin}" "${PLANREG_REPO_ROOT}/scripts/audit_formal_config_pair.py" \
    --base "${base_resolved}" --driving-vqa "${vqa_resolved}" \
    --output "${pair_audit}" > "${pair_root}/formal_config_pair_audit.stdout.json"
  flock -u 9

  local preflight_dir
  preflight_dir="$(mktemp -d)"
  local preflight_identity="${preflight_dir}/formal_run_identity.json"
  local preflight_args=(
    --variant "${variant}"
    --vlm-path "${vlm_path}"
    --vlm-audit "${vlm_audit}"
    --shared-init "${PLANREG_SHARED_INIT}"
    --layout-lock "${PLANREG_LAYOUT_LOCK}"
    --input-cache "${input_cache}"
    --output-dir "${output_dir}"
    --experiment-name "${experiment_name}"
    --seed "${seed}"
    --repo-root "${PLANREG_REPO_ROOT}"
    --metadata-output "${preflight_identity}"
  )
  if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    preflight_args+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
  fi
  "${python_bin}" "${PLANREG_REPO_ROOT}/scripts/validate_formal_training_prerequisites.py" \
    "${preflight_args[@]}" > "${preflight_dir}/preflight.stdout.json"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "FORMAL_TRAIN_DRY_RUN variant=${variant} seed=${seed} layout=${selected_layout} global_batch=${global_batch}"
    rm -rf "${preflight_dir}"
    return 0
  fi

  mkdir -p "${output_dir}/run_metadata"
  if [[ -z "${RESUME_CHECKPOINT:-}" ]]; then
    cp "${preflight_identity}" "${output_dir}/run_metadata/formal_run_identity.json"
  else
    cp "${preflight_identity}" "${output_dir}/run_metadata/resume_preflight_$(date -u +%Y%m%dT%H%M%SZ).json"
  fi
  rm -rf "${preflight_dir}"
  cp "${pair_audit}" "${output_dir}/run_metadata/formal_config_pair_audit.json"
  cp "${PLANREG_LAYOUT_LOCK}" "${output_dir}/run_metadata/formal_training_layout_lock.json"
  cp "${input_cache}/planreg_input_only_manifest.json" "${output_dir}/run_metadata/input_only_cache_manifest.json"
  cp "${vlm_audit}" "${output_dir}/run_metadata/formal_vlm_initialization_audit.json"
  if [[ "${variant}" == "base" ]]; then
    cp "${base_resolved}" "${output_dir}/run_metadata/resolved_hydra_config.yaml"
  else
    cp "${vqa_resolved}" "${output_dir}/run_metadata/resolved_hydra_config.yaml"
  fi
  git -C "${PLANREG_REPO_ROOT}" rev-parse HEAD > "${output_dir}/run_metadata/git_commit.txt"
  git -C "${PLANREG_REPO_ROOT}" status --short --branch > "${output_dir}/run_metadata/git_status.txt"
  env | LC_ALL=C sort | sed -E \
    's/^([^=]*(TOKEN|SECRET|PASSWORD|API_KEY)[^=]*)=.*/\1=<redacted>/I' \
    > "${output_dir}/run_metadata/environment.txt"

  _formal_assert_idle_gpus "$(hostname)" "${devices_per_node}"
  if [[ "${num_nodes}" -eq 2 ]]; then
    if [[ "$(hostname)" != *"vla-zt-worker-0"* || "$(hostname)" == *"vla-zt2"* ]]; then
      echo "A 16-GPU formal run must be coordinated from training-vla-zt" >&2
      return 2
    fi
    ssh -o BatchMode=yes "${PLANREG_PEER_HOST:-training-vla-zt2}" \
      "nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | grep -q . && exit 42 || test \$(nvidia-smi -L | wc -l) -ge ${devices_per_node}"
  fi

  local training_hydra_args=(
    --config-name=formal_planreg_wm_training
    "agent=${agent_config}"
    "${common_hydra_args[@]}"
    "experiment_name=${experiment_name}"
    "output_dir=${output_dir}"
  )
  printf '%q ' "${python_bin}" "${PLANREG_REPO_ROOT}/navsim/planning/script/run_training_full.py" "${training_hydra_args[@]}" \
    > "${output_dir}/run_metadata/train_command.txt"
  printf '\n' >> "${output_dir}/run_metadata/train_command.txt"

  if [[ "${num_nodes}" -eq 1 ]]; then
    "${python_bin}" "${PLANREG_REPO_ROOT}/navsim/planning/script/run_training_full.py" \
      "${training_hydra_args[@]}" 2>&1 | tee "${output_dir}/run_metadata/train.log"
  else
    local peer="${PLANREG_PEER_HOST:-training-vla-zt2}"
    local master_addr="${PLANREG_MASTER_ADDR:-$(hostname -I | awk '{print $1}')}"
    local master_port="${PLANREG_MASTER_PORT:-29630}"
    local shared_environment=(
      "PLANREG_BASE_VLM_PATH=${base_vlm}"
      "PLANREG_VQA_VLM_PATH=${vqa_vlm}"
      "PLANREG_FORMAL_VLM_PATH=${vlm_path}"
      "PLANREG_INITIALIZATION_VARIANT=${variant}"
      "PLANREG_VLM_CHECKPOINT_SHA256=${checkpoint_sha}"
      "PLANREG_VLM_CONFIG_SHA256=${config_sha}"
      "PLANREG_SHARED_INIT=${PLANREG_SHARED_INIT}"
      "PLANREG_INPUT_CACHE=${input_cache}"
      "PLANREG_OUTPUT_DIR=${output_dir}"
      "PLANREG_EXPERIMENT_NAME=${experiment_name}"
      "NAVSIM_TRAIN_METRIC_CACHE=${metric_cache}"
      "OPENSCENE_DATA_ROOT=${navsim_data}"
      "NUPLAN_MAPS_ROOT=${maps_root}"
      DRIVEVLA_SCORE_RAY=0
      "DRIVEVLA_SCORE_PROCESSES=${scorer_processes}"
      DRIVEVLA_SCORE_PARTITIONS=8
      DRIVEVLA_SCORE_START_METHOD=forkserver
      DRIVEVLA_BIND_RANK_CPUS=1 DRIVEVLA_SYNC_TRAIN_METRICS=0
      "DRIVEVLA_TRAIN_LOG_INTERVAL=${DRIVEVLA_TRAIN_LOG_INTERVAL}"
      PLANREG_FORMAL_TIMING=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
      "PYTHONPATH=${PLANREG_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
      "HF_HOME=${HF_HOME}" "MPLCONFIGDIR=${MPLCONFIGDIR}"
      "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
      "HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
      "TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM}"
    )
    if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
      shared_environment+=("RESUME_CHECKPOINT=${RESUME_CHECKPOINT}")
    fi
    local torchrun=(
      "${python_bin}" -m torch.distributed.run --nnodes=2 --nproc-per-node=8
      --master-addr="${master_addr}" --master-port="${master_port}"
    )
    local remote_array=(
      env "${shared_environment[@]}" "${torchrun[@]}" --node-rank=1
      "${PLANREG_REPO_ROOT}/navsim/planning/script/run_training_full.py"
      "${training_hydra_args[@]}"
    )
    local remote_command
    printf -v remote_command '%q ' "${remote_array[@]}"
    ssh -o BatchMode=yes "${peer}" "${remote_command}" \
      > "${output_dir}/run_metadata/node1.log" 2>&1 &
    local remote_pid=$!
    sleep 3
    local local_exit=0 remote_exit=0
    env "${shared_environment[@]}" "${torchrun[@]}" --node-rank=0 \
      "${PLANREG_REPO_ROOT}/navsim/planning/script/run_training_full.py" \
      "${training_hydra_args[@]}" 2>&1 | tee "${output_dir}/run_metadata/node0.log" || local_exit=${PIPESTATUS[0]}
    wait "${remote_pid}" || remote_exit=$?
    if [[ "${local_exit}" -ne 0 || "${remote_exit}" -ne 0 ]]; then
      echo "Formal multi-node training failed: local=${local_exit} remote=${remote_exit}" >&2
      return 1
    fi
  fi

  local final_checkpoint="${output_dir}/checkpoints/epoch_27_final.ckpt"
  _formal_require_file "${final_checkpoint}"
  local deploy_dir="${output_dir}/deploy"
  mkdir -p "${deploy_dir}"
  local student_name
  if [[ "${variant}" == "base" ]]; then
    student_name=formal_base_init_wm_epoch27_student.ckpt
  else
    student_name=formal_vqa_init_wm_epoch27_student.ckpt
  fi
  "${python_bin}" "${PLANREG_REPO_ROOT}/scripts/export_planreg_student_checkpoint.py" \
    "${final_checkpoint}" "${deploy_dir}/${student_name}" \
    --resolved-config "${output_dir}/run_metadata/resolved_hydra_config.yaml" \
    > "${output_dir}/run_metadata/student_export.json"
  "${python_bin}" "${PLANREG_REPO_ROOT}/scripts/export_planreg_student_checkpoint.py" \
    --verify "${deploy_dir}/${student_name}"
}
