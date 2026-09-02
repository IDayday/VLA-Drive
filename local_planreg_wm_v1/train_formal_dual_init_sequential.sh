#!/usr/bin/env bash
set -euo pipefail

# One selected 16-GPU layout occupies both vla-zt and vla-zt2.  Keep the
# scientifically paired runs on that same lock by queueing Driving-VQA only
# after the Base run and its student export finish successfully.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
seed="${1:-0}"

: "${PLANREG_LAYOUT_LOCK:?Set PLANREG_LAYOUT_LOCK}"
: "${PLANREG_SHARED_INIT:?Set PLANREG_SHARED_INIT}"
: "${PLANREG_BASE_VLM_PATH:?Set PLANREG_BASE_VLM_PATH}"
: "${PLANREG_VQA_VLM_PATH:?Set PLANREG_VQA_VLM_PATH}"

echo "PLANREG_FORMAL_SEQUENCE state=starting_base seed=${seed} utc=$(date -u +%FT%TZ)"
bash "${script_dir}/train_formal_base_init_wm.sh" "${seed}"
echo "PLANREG_FORMAL_SEQUENCE state=base_complete_starting_vqa seed=${seed} utc=$(date -u +%FT%TZ)"
bash "${script_dir}/train_formal_vqa_init_wm.sh" "${seed}"
echo "PLANREG_FORMAL_SEQUENCE state=complete seed=${seed} utc=$(date -u +%FT%TZ)"
