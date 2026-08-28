#!/usr/bin/env bash
set -euo pipefail

# Isolation preflight (run from the original checkout).
git status --short
git rev-parse HEAD
git branch --show-current
git worktree add -b feature/navsim-candidate-relative-feasibility-audit \
  /mnt/workspace/project/DriveDreamer-Policy-navsim-candidate-relative-audit \
  1482f1da87e31907b549f09836a38f99fd18f200

# All commands below ran from the isolated worktree. Data and official caches
# were opened read-only; outputs were scoped to reports/navsim_candidate_relative_audit.
export PYTHONPATH=.
export MPLBACKEND=Agg

python -m tools.navsim_candidate_relative_audit.inspect_environment \
  --split train --max-scenes 8 --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.inspect_scenes \
  --mode smoke --split train --max-scenes 8 --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.validate_alignment \
  --split train --max-scenes 8 --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.candidate_generator \
  --split train --max-scenes 8 --num-candidates 12 --traffic-policy non_reactive \
  --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.score_candidates \
  --split train --max-scenes 8 --traffic-policy non_reactive --verify-runs 2 \
  --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.build_candidate_relative_targets \
  --split train --max-scenes 8 --max-actors 16 --traffic-policy non_reactive \
  --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.analyze_target_diversity \
  --split train --max-scenes 8 --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.build_soft_contrastive_labels \
  --split train --max-scenes 8 --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.run_oracle_probe \
  --split train --max-scenes 8 --max-scenes-per-log 1 --num-candidates 12 \
  --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.audit_future_visual_anchor \
  --split train --max-scenes 8 --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.audit_v2_extensions \
  --split train --max-scenes 1 --skip-track-rerun --synthetic-metadata-samples 16 \
  --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.visualize_audit \
  --split train --max-scenes 8 --output-dir reports/navsim_candidate_relative_audit

# Expanded field, candidate, consequence, soft-label, and visual audit.
python -m tools.navsim_candidate_relative_audit.inspect_scenes \
  --mode audit --split train --max-scenes 64 --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.candidate_generator \
  --split train --max-scenes 64 --num-candidates 12 --traffic-policy non_reactive \
  --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.score_candidates \
  --split train --max-scenes 64 --traffic-policy non_reactive --verify-runs 2 \
  --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.build_candidate_relative_targets \
  --split train --max-scenes 64 --max-actors 16 --traffic-policy non_reactive \
  --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.analyze_target_diversity \
  --split train --max-scenes 64 --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.build_soft_contrastive_labels \
  --split train --max-scenes 64 --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.audit_future_visual_anchor \
  --split train --max-scenes 12 --output-dir reports/navsim_candidate_relative_audit

# Formal planning-utility probe and v2 extension audit.
python -m tools.navsim_candidate_relative_audit.run_oracle_probe \
  --split train --max-scenes 500 --max-scenes-per-log 8 --num-candidates 12 \
  --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.audit_v2_extensions \
  --split train --max-scenes 32 --synthetic-metadata-samples 512 \
  --output-dir reports/navsim_candidate_relative_audit

# Requested 500-scene field statistics pass.
python -m tools.navsim_candidate_relative_audit.inspect_scenes \
  --mode statistics --split train --max-scenes 500 --scene-object-samples 8 \
  --sensor-scene-samples 12 --output-dir reports/navsim_candidate_relative_audit

# Final assembly and verification (run after the long statistics/reactive jobs).
python -m tools.navsim_candidate_relative_audit.visualize_audit \
  --split train --max-scenes 64 --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.generate_report \
  --split train --max-scenes 500 --output-dir reports/navsim_candidate_relative_audit
pytest -q tests/test_navsim_candidate_relative_audit.py
# Existing action-effect tests require the nested local NAVSIM 2.0 source root.
PYTHONPATH=.:navsim pytest -q tests/action_effect tests/test_navsim_candidate_relative_audit.py
python -m compileall -q tools/navsim_candidate_relative_audit
bash -n tools/navsim_candidate_relative_audit/run_audit.sh
git diff --check
