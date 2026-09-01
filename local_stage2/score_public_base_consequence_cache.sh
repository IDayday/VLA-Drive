#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/local_stage2/common.sh"

source_root="${DRIVEVLA_PUBLIC_SCORER_CACHE:-${DRIVEVLA_STAGE2_RUN_ROOT}/scorer_pdms93/public_base_features_full_v1}"
factor_root="${DRIVEVLA_PUBLIC_SCORER_LABELS:-${DRIVEVLA_STAGE2_RUN_ROOT}/scorer_pdms93/public_base_labels_full_v1}"
output_root="${DRIVEVLA_PUBLIC_CONSEQUENCE_LABELS:-${DRIVEVLA_STAGE2_RUN_ROOT}/scorer_pdms93/public_base_consequence_labels_top16_v1}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

exec "${DRIVEVLA_PYTHON}" \
  "${repo_root}/local_stage2/score_public_base_consequence_cache.py" \
  --source-root "${source_root}" \
  --factor-root "${factor_root}" \
  --output-root "${output_root}" \
  --metric-cache "${DRIVEVLA_NAVTRAIN_METRIC_CACHE}" \
  "$@"
