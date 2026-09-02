#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

ORACLE_RUN="${ORACLE_RUN:-/mnt/workspace/project/VLA-Drive-cf-effect-oracle/experiments/cf_effect_wote_oracle_effect/oracle-effect-v2-20260829}"
RUN_ROOT="${RUN_ROOT:-experiments/cf_effect_wote_direct_rehab/direct-rehab-v1-20260901/representation-ablation-v1}"
ACCESS_POLICY="reports/cf_effect_wote_direct_rehab/ACCESS_POLICY.json"
ACCESS_LOG="experiments/cf_effect_wote_direct_rehab/direct-rehab-v1-20260901/audit/access_log.jsonl"

if [[ -e "${RUN_ROOT}" ]]; then
  echo "refusing existing representation-ablation output: ${RUN_ROOT}" >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}"

representations=(
  trajectory_only
  old_spatial_xattn
  pretrained_candidate_query
  path_aligned_current
  hybrid_current
  wote_current_only_rollout
)

for representation in "${representations[@]}"; do
  PYTHONPATH=. python -m research.cf_effect_gate_wote.src.direct_rehab_training train \
    --train-feature-root "${ORACLE_RUN}/features-train" \
    --train-label-root "${ORACLE_RUN}/labels-train" \
    --train-tokens research/cf_effect_gate_wote/configs/splits/direct_rehab_train_1024.txt \
    --train-limit 512 \
    --val-feature-root "${ORACLE_RUN}/features-val" \
    --val-label-root "${ORACLE_RUN}/labels-val" \
    --val-tokens research/cf_effect_gate_wote/configs/splits/direct_rehab_val_256.txt \
    --val-limit 256 \
    --representation "${representation}" \
    --objective O0 \
    --seed 0 \
    --learning-rate 0.0003 \
    --weight-decay 0.0001 \
    --batch-scenes 4 \
    --candidate-chunk 64 \
    --max-epochs 15 \
    --patience 4 \
    --safety-lambda 0.5 \
    --gradient-clip-norm 1.0 \
    --device cuda \
    --output "${RUN_ROOT}/${representation}.pt" \
    --access-policy "${ACCESS_POLICY}" \
    --access-log "${ACCESS_LOG}"
done
