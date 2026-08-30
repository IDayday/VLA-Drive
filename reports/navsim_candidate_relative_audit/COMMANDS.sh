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

# Predicted-consequence probe requested after the oracle audit. Model/strength
# selection uses train OOF candidate fidelity; validation logs remain untouched.
python -m tools.navsim_candidate_relative_audit.run_predicted_consequence_probe \
  --split train --max-scenes 500 --num-candidates 12 --max-scenes-per-log 8 \
  --output-dir reports/navsim_candidate_relative_audit \
  --effect-models ridge,extra_trees,mlp_raw,mlp_delta,mlp_delta_strong \
  --extra-trees-estimators 64 --extra-trees-max-depth 14 \
  --extra-trees-min-samples-leaf 4 --extra-trees-max-features 0.7 --jobs 8 \
  --mlp-hidden-dims 512,256,128 --mlp-learning-rate 0.001 \
  --mlp-weight-decay 0.0001 --mlp-max-epochs 160 --mlp-patience 20 \
  --mlp-delta-weight 1.0 --mlp-delta-weight-strong 4.0 \
  --mlp-overfit-scenes 8 --mlp-overfit-epochs 400 \
  --mlp-device cuda --bootstrap-samples 2000

# Fixed-seed repeat used the same arguments, wrote to an isolated temporary
# output, and compared against the formal result at 1e-6 schema precision.
python -m tools.navsim_candidate_relative_audit.run_predicted_consequence_probe \
  --split train --max-scenes 500 --num-candidates 12 --max-scenes-per-log 8 \
  --output-dir /tmp/navsim_predicted_mlp_repeat \
  --effect-models ridge,extra_trees,mlp_raw,mlp_delta,mlp_delta_strong \
  --extra-trees-estimators 64 --extra-trees-max-depth 14 \
  --extra-trees-min-samples-leaf 4 --extra-trees-max-features 0.7 --jobs 8 \
  --mlp-hidden-dims 512,256,128 --mlp-learning-rate 0.001 \
  --mlp-weight-decay 0.0001 --mlp-max-epochs 160 --mlp-patience 20 \
  --mlp-delta-weight 1.0 --mlp-delta-weight-strong 4.0 \
  --mlp-overfit-scenes 8 --mlp-overfit-epochs 400 \
  --mlp-device cuda --bootstrap-samples 2000 \
  --determinism-reference reports/navsim_candidate_relative_audit

# Data-scale control: the validation decision remains log-disjoint, while the
# training pool grows from 500 to 2,000 scenes.
python -m tools.navsim_candidate_relative_audit.run_predicted_consequence_probe \
  --split train --max-scenes 2000 --num-candidates 12 --max-scenes-per-log 8 \
  --output-dir reports/navsim_candidate_relative_audit/predicted_consequence_runs/scale_2000 \
  --effect-models extra_trees,mlp_delta,mlp_delta_strong \
  --extra-trees-estimators 64 --extra-trees-max-depth 14 \
  --extra-trees-min-samples-leaf 4 --extra-trees-max-features 0.7 --jobs 8 \
  --mlp-hidden-dims 512,256,128 --mlp-learning-rate 0.001 \
  --mlp-weight-decay 0.0001 --mlp-max-epochs 160 --mlp-patience 20 \
  --mlp-delta-weight 1.0 --mlp-delta-weight-strong 4.0 \
  --mlp-overfit-scenes 8 --mlp-overfit-epochs 400 \
  --mlp-device cuda --bootstrap-samples 2000
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
