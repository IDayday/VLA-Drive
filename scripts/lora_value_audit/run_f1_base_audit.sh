#!/usr/bin/env bash
set -euo pipefail

output_dir="${1:?usage: run_f1_base_audit.sh OUTPUT_DIR [MAX_SCENES]}"
max_scenes="${2:-0}"
repo_root=/mnt/project/DriveVLA-M0-lora-value-audit-20260902
args=(
  -m tools.lora_value_audit.analyze_base_candidates
  --proposal-pickle /mnt/project/DriveVLA-M0-stage2/runs/drivor_native_proposals/original25_navtest_fp32_v1_4shard/proposal_predictions.pkl
  --candidate-matrix /mnt/project/DriveVLA-M0-stage2/runs/drivor_native_proposals/original25_navtest_fp32_scored_v1/candidate_scores.npz
  --status-replay-root /root/scorer_pdms93_cache/drivor_scene_replay_public_base_navtest_full_v3_local_4shard
  --output-dir "$output_dir"
  --bootstrap-replicates 10000
  --seed 20260902
)
if [[ "$max_scenes" -gt 0 ]]; then
  args+=(--max-scenes "$max_scenes")
fi
cd "$repo_root"
/root/miniconda3/envs/ddp/bin/python "${args[@]}"
