#!/usr/bin/env bash
set -euo pipefail

PLANREG_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLANREG_REPO_ROOT="$(cd "${PLANREG_SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../load_env.sh
source "${PLANREG_REPO_ROOT}/load_env.sh"
# shellcheck source=formal_runtime.sh
source "${PLANREG_SCRIPT_DIR}/formal_runtime.sh"

_formal_benchmark_assert_idle_local() {
  local label="$1"
  local expected="$2"
  local visible
  visible="$(nvidia-smi -L | wc -l)"
  if [[ "${visible}" -lt "${expected}" ]]; then
    echo "${label} exposes ${visible} GPUs; ${expected} are required" >&2
    return 2
  fi
  local busy
  busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)"
  if [[ -n "${busy}" ]]; then
    echo "${label} has active GPU compute processes; refusing to preempt: ${busy}" >&2
    return 2
  fi
}

formal_benchmark_layout() {
  if [[ $# -ne 5 ]]; then
    echo "formal_benchmark_layout LAYOUT GPU_COUNT PER_GPU_BATCH NUM_NODES SCORER_PROCESSES" >&2
    return 2
  fi
  local layout="$1"
  local training_config=formal_planreg_wm_training
  local agent_config=episode_drive_planreg_wm_formal_base
  if [[ "${PLANREG_PROTOCOL_VERSION:-v1}" == v1p1 ]]; then
    training_config=formal_planreg_wm_v1p1_training
    agent_config=episode_drive_planreg_wm_v1p1_base
    export PLANREG_PROMPT_VERSION=single_front_v1p1
    : "${PLANREG_SHARED_INIT:?V1.1 requires a newly generated FP32 shared init}"
    : "${PLANREG_INPUT_CACHE:?V1.1 requires a separate prompt-versioned input cache}"
    [[ "$(jq -r '.prompt_version' "${PLANREG_INPUT_CACHE}/planreg_input_only_manifest.json")" == single_front_v1p1 ]] || {
      echo 'Stale prompt input cache; rebuild in a separate V1.1 directory' >&2; return 2;
    }
  fi
  local gpu_count="$2"
  local per_gpu_batch="$3"
  local num_nodes="$4"
  local scorer_processes="$5"
  local scorer_partitions="${PLANREG_BENCHMARK_SCORE_PARTITIONS:-8}"
  local devices_per_node=8
  local global_batch=$((gpu_count * per_gpu_batch))
  local num_workers="${PLANREG_BENCHMARK_NUM_WORKERS:-4}"
  local warmup_steps="${PLANREG_BENCHMARK_WARMUP_STEPS:-20}"
  local timed_steps="${PLANREG_BENCHMARK_TIMED_STEPS:-300}"
  local gradient_checkpointing="${PLANREG_BENCHMARK_GRADIENT_CHECKPOINTING:-true}"
  local attention_backend="${PLANREG_BENCHMARK_ATTENTION_BACKEND:-split_sdpa}"
  if ! [[ "${warmup_steps}" =~ ^[0-9]+$ && "${timed_steps}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Benchmark warmup/timed steps must be non-negative/positive integers" >&2
    return 2
  fi
  if ! [[ "${scorer_partitions}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PLANREG_BENCHMARK_SCORE_PARTITIONS must be a positive integer" >&2
    return 2
  fi
  if [[ "${gradient_checkpointing}" != "true" && "${gradient_checkpointing}" != "false" ]]; then
    echo "PLANREG_BENCHMARK_GRADIENT_CHECKPOINTING must be true or false" >&2
    return 2
  fi
  if [[ "${attention_backend}" != "eager" && "${attention_backend}" != "split_sdpa" ]]; then
    echo "PLANREG_BENCHMARK_ATTENTION_BACKEND must be eager or split_sdpa" >&2
    return 2
  fi
  planreg_formal_runtime_setup "${PLANREG_REPO_ROOT}"
  local python_bin="${PLANREG_FORMAL_PYTHON_BIN}"
  local base_vlm="${PLANREG_BASE_VLM_PATH:-/mnt/project/DriveVLA-M0-models/planreg-formal/InternVL3-2B-base-aligned}"
  local shared_init="${PLANREG_SHARED_INIT:-/mnt/project/DriveVLA-M0-models/planreg-formal/shared_planreg_init_seed0.pt}"
  local input_cache="${PLANREG_INPUT_CACHE:-/mnt/project/DriveVLA-M0-stage2/cache/feature_cache_navtrain_full}"
  local vlm_audit="${PLANREG_VLM_AUDIT_REPORT:-${PLANREG_REPO_ROOT}/reports/planreg_wm_v1/formal_vlm_initialization_audit.json}"
  local metric_cache="${PLANREG_TRAIN_METRIC_CACHE:-/mnt/project/DriveVLA-M0-stage2/cache/metric_cache_navtrain_full}"
  local navsim_data="${OPENSCENE_DATA_ROOT:-/mnt/project/DriveDreamer-Policy/navsim_raw}"
  local report_root="${PLANREG_BENCHMARK_REPORT_ROOT:-${PLANREG_REPO_ROOT}/reports/planreg_wm_v1/throughput}"
  local report_dir="${report_root}/${layout}"
  local metrics_path="${report_dir}/metrics.json"
  local run_root="${PLANREG_BENCHMARK_RUN_ROOT:-/mnt/project/DriveVLA-M0-formal-runs/throughput}"
  local output_dir="${run_root}/${layout}"
  local timeout_duration="${PLANREG_BENCHMARK_TIMEOUT:-24h}"

  for path in "${python_bin}" "${shared_init}" "${vlm_audit}" "${input_cache}/planreg_input_only_manifest.json"; do
    if [[ ! -e "${path}" ]]; then
      echo "Formal benchmark prerequisite is missing: ${path}" >&2
      return 2
    fi
  done
  for directory in "${base_vlm}" "${input_cache}" "${metric_cache}" "${navsim_data}/navsim_logs/trainval" "${navsim_data}/sensor_blobs/trainval" "${NUPLAN_MAPS_ROOT:-/mnt/navsim/maps}"; do
    if [[ ! -d "${directory}" ]]; then
      echo "Formal benchmark directory is missing: ${directory}" >&2
      return 2
    fi
  done
  if [[ -e "${output_dir}" || -e "${metrics_path}" ]]; then
    echo "Refusing to overwrite benchmark output for ${layout}: ${output_dir} or ${metrics_path}" >&2
    return 2
  fi
  _formal_benchmark_assert_idle_local "$(hostname)" 8
  if [[ "${num_nodes}" == "2" ]]; then
    local peer_for_idle="${PLANREG_PEER_HOST:-training-vla-zt2}"
    ssh -o BatchMode=yes "${peer_for_idle}" \
      'visible=$(nvidia-smi -L | wc -l); busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '\''/^[[:space:]]*$/d'\'' || true); [[ "$visible" -ge 8 && -z "$busy" ]]' \
      || { echo "${peer_for_idle} is unavailable, lacks 8 GPUs, or has an active GPU process" >&2; return 2; }
  fi

  local base_checkpoint_sha base_config_sha
  base_checkpoint_sha="$(jq -er '.base.checkpoint_sha256' "${vlm_audit}")"
  base_config_sha="$(jq -er '.base.config_sha256' "${vlm_audit}")"
  export PLANREG_FORMAL_VLM_PATH="${base_vlm}"
  mkdir -p "${output_dir}/run_metadata" "${report_dir}"
  local runtime_node0="${output_dir}/run_metadata/formal_runtime_node0.json"
  planreg_formal_runtime_audit_local "${PLANREG_REPO_ROOT}" "${runtime_node0}"
  if [[ "${num_nodes}" == "2" ]]; then
    local runtime_node1="${output_dir}/run_metadata/formal_runtime_node1.json"
    local runtime_pair="${output_dir}/run_metadata/formal_runtime_pair_audit.json"
    planreg_formal_runtime_audit_remote \
      "${PLANREG_PEER_HOST:-training-vla-zt2}" "${PLANREG_REPO_ROOT}" \
      "${runtime_node1}"
    planreg_formal_runtime_compare \
      "${PLANREG_REPO_ROOT}" "${runtime_node0}" "${runtime_node1}" \
      "${runtime_pair}"
  fi
  git -C "${PLANREG_REPO_ROOT}" rev-parse HEAD > "${output_dir}/run_metadata/git_commit.txt"
  git -C "${PLANREG_REPO_ROOT}" status --short --branch > "${output_dir}/run_metadata/git_status.txt"
  env | LC_ALL=C sort | sed -E \
    's/^([^=]*(TOKEN|SECRET|PASSWORD|API_KEY)[^=]*)=.*/\1=<redacted>/I' \
    > "${output_dir}/run_metadata/environment.txt"

  export PLANREG_BASE_VLM_PATH="${base_vlm}"
  export PLANREG_FORMAL_VLM_PATH="${base_vlm}"
  export PLANREG_INITIALIZATION_VARIANT=base
  export PLANREG_VLM_CHECKPOINT_SHA256="${base_checkpoint_sha}"
  export PLANREG_VLM_CONFIG_SHA256="${base_config_sha}"
  export PLANREG_SHARED_INIT="${shared_init}"
  export PLANREG_INPUT_CACHE="${input_cache}"
  export PLANREG_OUTPUT_DIR="${output_dir}"
  export PLANREG_EXPERIMENT_NAME="benchmark_formal_${layout}"
  export NAVSIM_TRAIN_METRIC_CACHE="${metric_cache}"
  export OPENSCENE_DATA_ROOT="${navsim_data}"
  export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/navsim/maps}"
  export DRIVEVLA_SCORE_RAY=0
  export DRIVEVLA_SCORE_PROCESSES="${scorer_processes}"
  export DRIVEVLA_SCORE_PARTITIONS="${scorer_partitions}"
  export DRIVEVLA_SCORE_START_METHOD=forkserver
  export DRIVEVLA_BIND_RANK_CPUS=1
  export DRIVEVLA_SYNC_TRAIN_METRICS=0
  export DRIVEVLA_TRAIN_LOG_INTERVAL=20
  export PLANREG_FORMAL_TIMING=1
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

  local hydra_args=(
    --config-name="${training_config}"
    agent="${agent_config}"
    "experiment_name=benchmark_formal_${layout}"
    "output_dir=${output_dir}"
    seed=0
    "cache_path=${input_cache}"
    "navsim_log_path=${navsim_data}/navsim_logs/trainval"
    "sensor_blobs_path=${navsim_data}/sensor_blobs/trainval"
    "agent.batch_size=${per_gpu_batch}"
    "agent.num_gpus=${gpu_count}"
    "agent.vlm_config.gradient_checkpointing=${gradient_checkpointing}"
    "agent.planning_registers.read_only_attention_backend=${attention_backend}"
    "dataloader.params.batch_size=${per_gpu_batch}"
    "dataloader.params.num_workers=${num_workers}"
    dataloader.params.multiprocessing_context=forkserver
    "trainer.params.devices=${devices_per_node}"
    "trainer.params.num_nodes=${num_nodes}"
    trainer.params.strategy=ddp
    throughput_benchmark.enabled=true
    "throughput_benchmark.layout_name=${layout}"
    "throughput_benchmark.output_path=${metrics_path}"
    "throughput_benchmark.warmup_steps=${warmup_steps}"
    "throughput_benchmark.timed_steps=${timed_steps}"
    "throughput_benchmark.scorer_processes_per_rank=${scorer_processes}"
    "throughput_benchmark.scorer_partitions_per_scene=${scorer_partitions}"
    hydra.output_subdir=null
  )
  printf '%q ' "${python_bin}" "${PLANREG_REPO_ROOT}/navsim/planning/script/run_training_full.py" "${hydra_args[@]}" \
    > "${output_dir}/run_metadata/benchmark_command.txt"
  printf '\n' >> "${output_dir}/run_metadata/benchmark_command.txt"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "FORMAL_BENCHMARK_DRY_RUN layout=${layout} global_batch=${global_batch} gradient_checkpointing=${gradient_checkpointing} attention_backend=${attention_backend} warmup=${warmup_steps} timed=${timed_steps}"
    cat "${output_dir}/run_metadata/benchmark_command.txt"
    return 0
  fi

  local exit_code=0
  if [[ "${num_nodes}" == "1" ]]; then
    timeout "${timeout_duration}" "${python_bin}" \
      "${PLANREG_REPO_ROOT}/navsim/planning/script/run_training_full.py" \
      "${hydra_args[@]}" 2>&1 | tee "${output_dir}/run_metadata/train.log" || exit_code=${PIPESTATUS[0]}
  else
    local coordinator="${PLANREG_COORDINATOR_HOST:-training-vla-zt}"
    local peer="${PLANREG_PEER_HOST:-training-vla-zt2}"
    local current_host
    current_host="$(hostname)"
    if [[ "${current_host}" != *"vla-zt-worker-0"* || "${current_host}" == *"vla-zt2"* ]]; then
      echo "Run 16-GPU benchmark from ${coordinator}; current host is ${current_host}" >&2
      return 2
    fi
    local master_addr="${PLANREG_MASTER_ADDR:-$(hostname -I | awk '{print $1}')}"
    local master_port="${PLANREG_MASTER_PORT:-29620}"
    local common_env=(
      "PLANREG_BASE_VLM_PATH=${base_vlm}"
      "PLANREG_FORMAL_VLM_PATH=${base_vlm}"
      PLANREG_INITIALIZATION_VARIANT=base
      "PLANREG_VLM_CHECKPOINT_SHA256=${base_checkpoint_sha}"
      "PLANREG_VLM_CONFIG_SHA256=${base_config_sha}"
      "PLANREG_SHARED_INIT=${shared_init}"
      "PLANREG_PROMPT_VERSION=${PLANREG_PROMPT_VERSION:-legacy}"
      "PLANREG_INPUT_CACHE=${input_cache}"
      "PLANREG_OUTPUT_DIR=${output_dir}"
      "PLANREG_EXPERIMENT_NAME=benchmark_formal_${layout}"
      "NAVSIM_TRAIN_METRIC_CACHE=${metric_cache}"
      "OPENSCENE_DATA_ROOT=${navsim_data}"
      "NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT}"
      DRIVEVLA_SCORE_RAY=0
      "DRIVEVLA_SCORE_PROCESSES=${scorer_processes}"
      "DRIVEVLA_SCORE_PARTITIONS=${scorer_partitions}"
      DRIVEVLA_SCORE_START_METHOD=forkserver
      DRIVEVLA_BIND_RANK_CPUS=1
      DRIVEVLA_SYNC_TRAIN_METRICS=0
      DRIVEVLA_TRAIN_LOG_INTERVAL=20
      PLANREG_FORMAL_TIMING=1
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
      "PYTHONPATH=${PLANREG_FORMAL_PYTHONPATH}" PYTHONNOUSERSITE=1
      "HF_HOME=${HF_HOME}" "MPLCONFIGDIR=${MPLCONFIGDIR}"
      "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
      "HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
      "TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM}"
    )
    local torchrun_base=(
      "${python_bin}" -m torch.distributed.run
      --nnodes=2 --nproc-per-node=8
      --master-addr="${master_addr}" --master-port="${master_port}"
    )
    local remote_array=(
      env "${common_env[@]}" "${torchrun_base[@]}" --node-rank=1
      "${PLANREG_REPO_ROOT}/navsim/planning/script/run_training_full.py"
      "${hydra_args[@]}"
    )
    local remote_command
    printf -v remote_command '%q ' "${remote_array[@]}"
    timeout "${timeout_duration}" ssh -o BatchMode=yes "${peer}" "${remote_command}" \
      > "${output_dir}/run_metadata/node1.log" 2>&1 &
    local remote_pid=$!
    sleep 3
    timeout "${timeout_duration}" env "${common_env[@]}" \
      "${torchrun_base[@]}" --node-rank=0 \
      "${PLANREG_REPO_ROOT}/navsim/planning/script/run_training_full.py" \
      "${hydra_args[@]}" 2>&1 | tee "${output_dir}/run_metadata/node0.log" || exit_code=${PIPESTATUS[0]}
    local remote_exit=0
    wait "${remote_pid}" || remote_exit=$?
    if [[ "${exit_code}" -eq 0 && "${remote_exit}" -ne 0 ]]; then
      exit_code="${remote_exit}"
    fi
  fi

  if [[ "${exit_code}" -ne 0 ]]; then
    local deadlock_arg=()
    if [[ "${exit_code}" -eq 124 ]]; then
      deadlock_arg=(--deadlock)
    fi
    "${python_bin}" "${PLANREG_REPO_ROOT}/scripts/mark_formal_benchmark_failure.py" \
      --output "${metrics_path}" --layout "${layout}" --exit-code "${exit_code}" \
      --global-batch "${global_batch}" --gpu-count "${gpu_count}" \
      --per-gpu-batch "${per_gpu_batch}" \
      --scorer-processes-per-rank "${scorer_processes}" \
      --scorer-partitions-per-scene "${scorer_partitions}" \
      --num-workers-per-rank "${num_workers}" "${deadlock_arg[@]}"
    return "${exit_code}"
  fi
  if [[ ! -f "${metrics_path}" ]]; then
    echo "Benchmark exited successfully but did not create ${metrics_path}" >&2
    return 2
  fi
}
