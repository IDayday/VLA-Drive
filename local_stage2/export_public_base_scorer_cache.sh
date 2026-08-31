#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/local_stage2/common.sh"

python_bin="${DRIVEVLA_PYTHON}"
checkpoint="${DRIVEVLA_PUBLIC_BASE}"
feature_cache="${DRIVEVLA_NAVTRAIN_LONG2_FEATURE_CACHE}"
output_dir="${DRIVEVLA_PUBLIC_SCORER_CACHE:-${DRIVEVLA_STAGE2_RUN_ROOT}/scorer_pdms93/public_base_features}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
# Proposal export never calls the training-time oracle scorer.  Prevent the
# base agent constructor from launching an otherwise idle local Ray cluster.
export DRIVEVLA_SCORE_RAY=0
export DRIVEVLA_SCORE_PROCESSES=0

exec "${python_bin}" "${repo_root}/local_stage2/export_public_base_scorer_cache.py" \
  --repo-root "${repo_root}" \
  --checkpoint "${checkpoint}" \
  --feature-cache "${feature_cache}" \
  --output-dir "${output_dir}" \
  "$@"
