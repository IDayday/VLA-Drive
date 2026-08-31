#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/local_stage2/common.sh"

source_root="${DRIVEVLA_PUBLIC_SCORER_CACHE:-${DRIVEVLA_STAGE2_RUN_ROOT}/scorer_pdms93/public_base_features_full_v1}"
label_root="${DRIVEVLA_PUBLIC_SCORER_LABELS:-${DRIVEVLA_STAGE2_RUN_ROOT}/scorer_pdms93/public_base_labels_full_v1}"
output_dir="${DRIVEVLA_RESIDUAL_SCORER_OUTPUT:-${DRIVEVLA_STAGE2_RUN_ROOT}/scorer_pdms93/residual_local_seed2}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

exec "${DRIVEVLA_PYTHON}" "${repo_root}/local_stage2/train_public_base_residual_scorer.py" \
  --repo-root "${repo_root}" \
  --source-root "${source_root}" \
  --label-root "${label_root}" \
  --base-checkpoint "${DRIVEVLA_PUBLIC_BASE}" \
  --output-dir "${output_dir}" \
  "$@"
