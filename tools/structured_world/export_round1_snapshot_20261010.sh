#!/usr/bin/env bash
set -euo pipefail

# CPU-only export from actual local artifacts. Require a fresh destination.
snapshot_output=${1:?Usage: bash tools/structured_world/export_round1_snapshot_20261010.sh NEW_OUTPUT_DIRECTORY}
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
artifact_root=/mnt/project/structured-world-fgtr-round1-artifacts/20261008
python "$repo_root/tools/structured_world/export_experiment_snapshot.py" \
  --run "$artifact_root/formal_navsim_zt2_G1_G0_v3_io/G1_FULL_UNIFORM" \
  --run "$artifact_root/formal_navsim_local_G3_G2_v3_io/G3_EVENT_LOCAL" \
  --run "$artifact_root/nuscenes_full8_deployment_20261009/G1_FULL_UNIFORM_formal/G1_FULL_UNIFORM" \
  --run "$artifact_root/nuscenes_full8_deployment_20261009/G3_EVENT_LOCAL_formal/G3_EVENT_LOCAL" \
  --navsim-reference S0=/mnt/project/action-video-foresight-artifacts/20261002/scores_reconciled_development_v1/formal_S0_seed42_full100k_v2_dev50000_seed42/scenes.csv \
  --navsim-reference S3=/mnt/project/action-video-foresight-artifacts/20261002/scores_reconciled_development_v1/formal_S3_seed42_full100k_v2_dev50000_seed42/scenes.csv \
  --navsim-reference V_QUERY=/mnt/project/planning-interface-transfer-artifacts/20261005/scores/formal_V_QUERY_seed42_full100k_v2_dev50000_seed42/scenes.csv \
  --output "$snapshot_output"
