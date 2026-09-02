#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

ORACLE_RUN="${ORACLE_RUN:-/mnt/workspace/project/VLA-Drive-cf-effect-oracle/experiments/cf_effect_wote_oracle_effect/oracle-effect-v2-20260829}"
RUN_ROOT="${RUN_ROOT:-experiments/cf_effect_wote_direct_rehab/direct-rehab-v1-20260901/confirmation-v1}"
ACCESS_POLICY="reports/cf_effect_wote_direct_rehab/ACCESS_POLICY.json"
ACCESS_LOG="experiments/cf_effect_wote_direct_rehab/direct-rehab-v1-20260901/audit/access_log.jsonl"

if [[ -e "${RUN_ROOT}" ]]; then
  echo "refusing existing confirmation output: ${RUN_ROOT}" >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}"

representations=(hybrid_current pretrained_candidate_query)
seeds=(0 1 2)

for representation in "${representations[@]}"; do
  for seed in "${seeds[@]}"; do
    PYTHONPATH=. python -m research.cf_effect_gate_wote.src.direct_rehab_training train \
      --train-feature-root "${ORACLE_RUN}/features-train" \
      --train-label-root "${ORACLE_RUN}/labels-train" \
      --train-tokens research/cf_effect_gate_wote/configs/splits/direct_rehab_train_1024.txt \
      --train-limit 1024 \
      --val-feature-root "${ORACLE_RUN}/features-val" \
      --val-label-root "${ORACLE_RUN}/labels-val" \
      --val-tokens research/cf_effect_gate_wote/configs/splits/direct_rehab_val_256.txt \
      --val-limit 256 \
      --representation "${representation}" \
      --objective O0 \
      --seed "${seed}" \
      --learning-rate 0.0003 \
      --weight-decay 0.0001 \
      --batch-scenes 4 \
      --candidate-chunk 64 \
      --max-epochs 20 \
      --patience 4 \
      --safety-lambda 0.5 \
      --gradient-clip-norm 1.0 \
      --device cuda \
      --output "${RUN_ROOT}/${representation}-seed${seed}.pt" \
      --access-policy "${ACCESS_POLICY}" \
      --access-log "${ACCESS_LOG}"
  done
done
