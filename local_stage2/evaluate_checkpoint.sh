#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 CHECKPOINT EXPERIMENT_NAME" >&2
  exit 2
fi

checkpoint_path="$1"
experiment_name="$2"
evaluation_root="${DRIVEVLA_STAGE2_RUN_ROOT}/ke/${experiment_name}"
evaluation_marker="${evaluation_root}/.evaluation_started"
mkdir -p "${evaluation_root}"
touch "${evaluation_marker}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export DRIVEVLA_SCORE_RAY=0
export INTERNVL_VERBOSE_DYNAMIC_BATCH=0

# The synthetic pressure service may reclaim the GPUs between training exit
# and evaluator initialization. The user authorized stopping this exact task;
# keep the same narrow guard active for the complete evaluation.
stress_guard_pid=""
if [[ "${DRIVEVLA_KILL_GPU_STRESS:-1}" == "1" ]]; then
  guard_gpu_stress() {
    while true; do
      mapfile -t stress_pids < <(pgrep -f '/mnt/project/gpu_stress[.]py' || true)
      if (( ${#stress_pids[@]} > 0 )); then
        targets=()
        for stress_pid in "${stress_pids[@]}"; do
          [[ -r "/proc/${stress_pid}/cmdline" ]] || continue
          stress_command="$(tr '\0' ' ' < "/proc/${stress_pid}/cmdline")"
          [[ "${stress_command}" == *"/mnt/project/gpu_stress.py"* ]] || continue
          targets+=("${stress_pid}")
          mapfile -t children < <(pgrep -P "${stress_pid}" || true)
          targets+=("${children[@]}")
        done
        if (( ${#targets[@]} > 0 )); then
          printf 'Stopping authorized gpu_stress.py processes: %s\n' "${targets[*]}"
          kill "${targets[@]}" 2>/dev/null || true
          sleep 1
          for target_pid in "${targets[@]}"; do
            if kill -0 "${target_pid}" 2>/dev/null; then
              kill -9 "${target_pid}" 2>/dev/null || true
            fi
          done
        fi
      fi
      sleep 2
    done
  }
  guard_gpu_stress &
  stress_guard_pid=$!
  trap 'kill "${stress_guard_pid}" 2>/dev/null || true' EXIT
fi

"${DRIVEVLA_PYTHON}" "${DRIVEVLA_REPO_ROOT}/navsim/planning/script/run_pdm_score_multi_gpu.py" \
  train_test_split=navtest \
  agent=episode_drive \
  "agent.checkpoint_path=${checkpoint_path}" \
  agent.stage1_checkpoint_path=null \
  "experiment_name=${experiment_name}" \
  load_image_path=true \
  dataloader.params.batch_size=2 \
  +trainer.params.devices=8 \
  trainer.params.strategy=ddp \
  agent.action_head_config.proposal_num=64 \
  agent.action_head_config.refiner_ls_values=0.0 \
  agent.action_head_config.image_backbone.focus_front_cam=false \
  agent.action_head_config.one_token_per_traj=true \
  agent.action_head_config.refiner_num_heads=1 \
  agent.action_head_config.tf_d_model=256 \
  agent.action_head_config.tf_d_ffn=1024 \
  agent.action_head_config.area_pred=false \
  agent.action_head_config.agent_pred=false \
  agent.action_head_config.ref_num=4 \
  agent.action_head_config.noc=1 \
  agent.action_head_config.dac=1 \
  agent.action_head_config.ddc=0.0 \
  agent.action_head_config.ttc=5 \
  agent.action_head_config.ep=5 \
  agent.action_head_config.comfort=2 \
  agent.vlm_config.cam_type=single \
  agent.vlm_config.cache_hidden_state=false \
  agent.vlm_config.cache_mode=false \
  agent.vlm_config.freeze_backbone=true \
  agent.vlm_config.vlm_type=internvl \
  "agent.vlm_config.vlm_path=${DRIVEVLA_VLM_DIR}" \
  agent.vlm_config.initialize_from_config=true \
  agent.vlm_config.use_flash_attn=false \
  agent.vlm_config.extra_token_count=8 \
  agent.vlm_config.target_vocab_size=151682 \
  agent.lora_config.use_lora=true \
  'agent.lora_config.lora_target_modules=[attn.qkv,attn.proj,q_proj,v_proj,k_proj,o_proj]' \
  "metric_cache_path=${DRIVEVLA_NAVTEST_METRIC_CACHE}" \
  "navsim_log_path=${DRIVEVLA_DATA_ROOT}/navsim_logs/test" \
  "sensor_blobs_path=${DRIVEVLA_SENSOR_ROOT}/test" \
  worker=ray_distributed_no_torch \
  worker.threads_per_node=64 \
  worker.log_to_driver=false \
  logger_level=warning

mapfile -t candidate_csvs < <(
  find "${evaluation_root}" -type f -name '*.csv' -newer "${evaluation_marker}" -print
)
if (( ${#candidate_csvs[@]} != 1 )); then
  echo "Evaluation produced ${#candidate_csvs[@]} new result CSVs under ${evaluation_root}; expected one" >&2
  exit 1
fi
candidate_csv="${candidate_csvs[0]}"
if [[ ! -f "${DRIVEVLA_PUBLIC_BASE_CSV}" ]]; then
  echo "Public Base reference CSV is missing: ${DRIVEVLA_PUBLIC_BASE_CSV}" >&2
  exit 1
fi

comparison_json="${evaluation_root}/comparison.json"
"${DRIVEVLA_PYTHON}" "${DRIVEVLA_REPO_ROOT}/local_stage2/summarize_navtest.py" \
  "${candidate_csv}" \
  "${DRIVEVLA_PUBLIC_BASE_CSV}" \
  "${comparison_json}" \
  --checkpoint "${checkpoint_path}"
printf 'NAVTEST_COMPARISON csv=%s report=%s\n' \
  "${candidate_csv}" "${comparison_json}"
