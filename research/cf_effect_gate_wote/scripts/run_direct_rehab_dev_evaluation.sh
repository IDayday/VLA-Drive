#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

ORACLE_RUN="${ORACLE_RUN:-/mnt/workspace/project/VLA-Drive-cf-effect-oracle/experiments/cf_effect_wote_oracle_effect/oracle-effect-v2-20260829}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-experiments/cf_effect_wote_direct_rehab/direct-rehab-v1-20260901/confirmation-v1}"
RUN_ROOT="${RUN_ROOT:-experiments/cf_effect_wote_direct_rehab/direct-rehab-v1-20260901/dev-evaluation-v1}"
ACCESS_POLICY="reports/cf_effect_wote_direct_rehab/ACCESS_POLICY.json"
ACCESS_LOG="experiments/cf_effect_wote_direct_rehab/direct-rehab-v1-20260901/audit/access_log.jsonl"

if [[ -e "${RUN_ROOT}" ]]; then
  echo "refusing existing dev-evaluation output: ${RUN_ROOT}" >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}"

for representation in hybrid_current pretrained_candidate_query; do
  for seed in 0 1 2; do
    PYTHONPATH=. python -m research.cf_effect_gate_wote.src.direct_rehab_training evaluate \
      --feature-root "${ORACLE_RUN}/features-test" \
      --label-root "${ORACLE_RUN}/labels-test" \
      --tokens research/cf_effect_gate_wote/configs/splits/direct_rehab_dev_512.txt \
      --checkpoint "${CHECKPOINT_ROOT}/${representation}-seed${seed}.pt" \
      --batch-scenes 4 \
      --candidate-chunk 64 \
      --phase development \
      --device cuda \
      --output "${RUN_ROOT}/${representation}-seed${seed}.json" \
      --access-policy "${ACCESS_POLICY}" \
      --access-log "${ACCESS_LOG}"
  done
done
