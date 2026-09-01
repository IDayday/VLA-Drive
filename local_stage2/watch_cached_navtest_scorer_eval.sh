#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "Usage: $0 FEATURE_DIR CANDIDATE_MATRIX OUTPUT_DIR ARTIFACT_MANIFEST GPU [BOOTSTRAP_ITERATIONS]" >&2
  exit 2
fi

feature_dir="$1"
candidate_matrix="$2"
output_dir="$3"
artifact_manifest="$4"
gpu="$5"
bootstrap_iterations="${6:-10000}"
python_bin="${DRIVEVLA_EXACT_PYTHON:-/mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/envs/navsim_py39_exact/bin/python}"
poll_seconds="${DRIVEVLA_EVAL_POLL_SECONDS:-30}"

if [[ ! -d "${feature_dir}" ]]; then
  echo "Feature directory does not exist: ${feature_dir}" >&2
  exit 3
fi
if [[ ! -f "${candidate_matrix}" ]]; then
  echo "Candidate matrix does not exist: ${candidate_matrix}" >&2
  exit 3
fi
if [[ ! -f "${artifact_manifest}" ]]; then
  echo "Artifact manifest does not exist: ${artifact_manifest}" >&2
  exit 3
fi
if [[ -d "${output_dir}" ]] && [[ -n "$(find "${output_dir}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite scorer evaluation: ${output_dir}" >&2
  exit 3
fi

while [[ ! -f "${feature_dir}/proposal_cache_manifest.json" ]] \
  || [[ ! -f "${feature_dir}/proposal_predictions.pkl" ]]; do
  printf 'NAVTEST_SCORER_WAIT utc=%s feature_dir=%s\n' \
    "$(date -u +%FT%TZ)" "${feature_dir}"
  sleep "${poll_seconds}"
done

export PYTHONPATH="${DRIVEVLA_REPO_ROOT}:${DRIVEVLA_REPO_ROOT}/nuplan-devkit:/mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/envs/transformers_4_48_3:/mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/envs/lightning_2_2_1:/mnt/project/DriveVLA-M0-env/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${gpu}"

exec "${python_bin}" \
  "${DRIVEVLA_REPO_ROOT}/local_stage2/evaluate_cached_navtest_scorers.py" \
  --feature-cache "${feature_dir}/proposal_predictions.pkl" \
  --feature-manifest "${feature_dir}/proposal_cache_manifest.json" \
  --candidate-matrix "${candidate_matrix}" \
  --output-dir "${output_dir}" \
  --artifact-manifest "${artifact_manifest}" \
  --device cuda \
  --batch-size 128 \
  --bootstrap-iterations "${bootstrap_iterations}"
