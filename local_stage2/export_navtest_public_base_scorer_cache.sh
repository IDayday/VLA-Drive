#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 CUDA_DEVICE_CSV EXPERIMENT_NAME [HYDRA_OVERRIDES...]" >&2
  exit 2
fi

device_csv="$1"
experiment_name="$2"
shift 2

checkpoint="${DRIVEVLA_PUBLIC_CHECKPOINT:-/mnt/project/DriveVLA-M0-modelscope/best-epoch_26-step_174312.server_merged.ckpt}"
reference_cache="${DRIVEVLA_PUBLIC_NAVTEST_PROPOSALS:-${DRIVEVLA_STAGE2_RUN_ROOT}/ke_candidate_audit/public_base_navtest_proposal_full_fp32/proposal_predictions.pkl}"

if [[ ! -f "${reference_cache}" ]]; then
  echo "Locked FP32 public proposal cache not found: ${reference_cache}" >&2
  exit 2
fi

bash "${DRIVEVLA_REPO_ROOT}/local_stage2/run_navtest_proposal_audit.sh" \
  "${checkpoint}" \
  navsim.agents.EpisodeDrive.episodedrive_agent.EpisodeDriveAgent \
  "${experiment_name}" \
  "${device_csv}" \
  +agent.action_head_config.return_scorer_features=true \
  +agent.action_head_config.return_memory_fields=true \
  +proposal_audit_skip_cpu_scoring=true \
  "+proposal_audit_reference_predictions_path=${reference_cache}" \
  worker.threads_per_node=1 \
  "$@"
