#!/usr/bin/env bash
# Incrementally evaluate all Phase-3 seed-42 checkpoints in a separate suite.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${project_root}/env.sh"

export FIELD2PLAN_EVAL_EXPERIMENTS="${FIELD2PLAN_EVAL_EXPERIMENTS:-p3_dyn_nosup_noaccess,p3_dyn_only_real,p3_geo_dyn_real,p3_dyn_sup_noaccess,p3_dyn_access_nosup,p3_geo_dyn_temporal_shuffle,p3_geo_dyn_batch_shuffle}"
export FIELD2PLAN_EXPERIMENT_SEED="${FIELD2PLAN_EXPERIMENT_SEED:-42}"

infer_seed="${INFER_SEED:-20260808}"
protocol_id="navsim_v1_1_pdms_ws2_seed${infer_seed}"
orchestration_revision="${EVAL_ORCHESTRATION_REVISION:-distenvfix-v1}"
shared_root="${DRIVEDREAMER_SHARED_ROOT:-/mnt/zhangt_workspace/project/DriveDreamer-Policy}"
export EVAL_ARTIFACT_ROOT="${EVAL_ARTIFACT_ROOT:-${shared_root}/navsim_exp/field2plan_phase3_eval_16gpu_live/${protocol_id}-${orchestration_revision}}"
export PRED_ROOT="${PRED_ROOT:-${shared_root}/navsim_planning_results/field2plan_phase3_all_ckpts_${protocol_id}-${orchestration_revision}}"

exec bash "${project_root}/scripts/field2plan/10_eval_all_ckpts_16gpu.sh" "$@"
