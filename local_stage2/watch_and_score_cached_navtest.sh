#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 PREDICTIONS_PKL OUTPUT_DIR CPU_WORKERS [CHECKPOINT] [HYDRA_OVERRIDES...]" >&2
  exit 2
fi

predictions_path="$1"
output_dir="$2"
cpu_workers="$3"
checkpoint_path="${4:-}"
if [[ $# -ge 4 ]]; then
  shift 4
else
  shift 3
fi

poll_seconds="${DRIVEVLA_SCORE_WATCH_POLL_SECONDS:-15}"
echo "Waiting for proposal cache: ${predictions_path}"
until [[ -f "${predictions_path}" ]]; do
  sleep "${poll_seconds}"
done

if [[ -f "${output_dir}/summary.json" ]]; then
  echo "Completed summary already exists: ${output_dir}/summary.json"
  exit 0
fi

extra_overrides=()
if [[ -n "${checkpoint_path}" ]]; then
  escaped_checkpoint="${checkpoint_path//=/\\=}"
  extra_overrides+=("+proposal_score_checkpoint_path=${escaped_checkpoint}")
fi

export DRIVEVLA_SCORE_AGGREGATE_WHEN_COMPLETE=true
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/score_cached_navtest_proposals.sh" \
  "${predictions_path}" \
  "${output_dir}" \
  1 \
  0 \
  "${cpu_workers}" \
  "${extra_overrides[@]}" \
  "$@"
