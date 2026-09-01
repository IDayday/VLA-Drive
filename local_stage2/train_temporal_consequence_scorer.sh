#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/local_stage2/common.sh"

source_root="${DRIVEVLA_PUBLIC_SCORER_CACHE:-${DRIVEVLA_STAGE2_RUN_ROOT}/scorer_pdms93/public_base_features_full_v1}"
factor_root="${DRIVEVLA_PUBLIC_SCORER_LABELS:-${DRIVEVLA_STAGE2_RUN_ROOT}/scorer_pdms93/public_base_labels_full_v1}"
consequence_root="${DRIVEVLA_PUBLIC_CONSEQUENCE_LABELS:-${DRIVEVLA_STAGE2_RUN_ROOT}/scorer_pdms93/public_base_consequence_labels_top16_v1}"

exec "${DRIVEVLA_PYTHON}" \
  "${repo_root}/local_stage2/train_temporal_consequence_scorer.py" \
  --source-root "${source_root}" \
  --factor-root "${factor_root}" \
  --consequence-root "${consequence_root}" \
  --base-checkpoint "${DRIVEVLA_PUBLIC_BASE}" \
  "$@"
