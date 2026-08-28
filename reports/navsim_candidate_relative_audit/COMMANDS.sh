#!/usr/bin/env bash
set -euo pipefail

python -m tools.navsim_candidate_relative_audit.inspect_environment --split trainval --mode smoke --max-scenes 8 --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.inspect_scenes --split trainval --mode smoke --max-scenes 8 --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.validate_alignment --split trainval --mode smoke --max-scenes 8 --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.validate_alignment --split trainval --mode smoke --max-scenes 8 --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.candidate_generator --split trainval --mode smoke --max-scenes 8 --num-candidates 12 --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.score_candidates --split trainval --mode smoke --max-scenes 8 --num-candidates 12 --sanity-scenes 2 --traffic-policy non_reactive --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.build_candidate_relative_targets --split trainval --mode smoke --max-scenes 8 --num-candidates 12 --max-actors 16 --max-shared-actors 64 --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.analyze_target_diversity --split trainval --mode smoke --max-scenes 8 --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.candidate_generator --split trainval --mode smoke --max-scenes 8 --num-candidates 12 --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.score_candidates --split trainval --mode smoke --max-scenes 8 --num-candidates 12 --sanity-scenes 2 --traffic-policy non_reactive --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.build_candidate_relative_targets --split trainval --mode smoke --max-scenes 8 --num-candidates 12 --max-actors 16 --max-shared-actors 64 --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.analyze_target_diversity --split trainval --mode smoke --max-scenes 8 --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.build_soft_contrastive_labels --split trainval --mode smoke --max-scenes 8 --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.run_oracle_probe --split trainval --mode smoke --max-scenes 8 --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.analyze_target_diversity --split trainval --max-scenes 8 --mode smoke --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.audit_future_visual_anchor --split trainval --max-scenes 8 --mode smoke --num-figures 8 --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.audit_v2_extensions --split trainval --max-scenes 8 --mode smoke --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.visualize_audit --split trainval --max-scenes 8 --mode smoke --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.generate_final_report --split trainval --max-scenes 8 --mode smoke --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.inspect_environment --split trainval --max-scenes 500 --output-dir reports/navsim_candidate_relative_audit --mode full
python -m tools.navsim_candidate_relative_audit.inspect_scenes --split trainval --max-scenes 500 --output-dir reports/navsim_candidate_relative_audit --mode full
python -m tools.navsim_candidate_relative_audit.validate_alignment --split trainval --max-scenes 64 --output-dir reports/navsim_candidate_relative_audit --mode full
python -m tools.navsim_candidate_relative_audit.candidate_generator --split trainval --max-scenes 500 --num-candidates 12 --output-dir reports/navsim_candidate_relative_audit --mode full
python -m tools.navsim_candidate_relative_audit.score_candidates --split trainval --max-scenes 500 --num-candidates 12 --output-dir reports/navsim_candidate_relative_audit --mode full --traffic-policy non_reactive --sanity-scenes 2
python -m tools.navsim_candidate_relative_audit.build_candidate_relative_targets --split trainval --max-scenes 500 --num-candidates 12 --output-dir reports/navsim_candidate_relative_audit --mode full --max-actors 16 --max-shared-actors 64
python -m tools.navsim_candidate_relative_audit.analyze_target_diversity --split trainval --max-scenes 500 --output-dir reports/navsim_candidate_relative_audit --mode full
python -m tools.navsim_candidate_relative_audit.build_soft_contrastive_labels --split trainval --max-scenes 500 --output-dir reports/navsim_candidate_relative_audit --mode full
python -m tools.navsim_candidate_relative_audit.run_oracle_probe --split trainval --max-scenes 500 --output-dir reports/navsim_candidate_relative_audit --mode full
python -m tools.navsim_candidate_relative_audit.audit_future_visual_anchor --split trainval --max-scenes 500 --output-dir reports/navsim_candidate_relative_audit --mode full --num-figures 12
python -m tools.navsim_candidate_relative_audit.audit_v2_extensions --split trainval --max-scenes 500 --output-dir reports/navsim_candidate_relative_audit --mode full
python -m tools.navsim_candidate_relative_audit.visualize_audit --split trainval --max-scenes 500 --output-dir reports/navsim_candidate_relative_audit --mode full
python -m tools.navsim_candidate_relative_audit.generate_final_report --split trainval --max-scenes 500 --output-dir reports/navsim_candidate_relative_audit --mode full
bash tools/navsim_candidate_relative_audit/run_audit.sh --mode full --split trainval --max-scenes 500 --num-candidates 12 --output-dir reports/navsim_candidate_relative_audit --traffic-policy non_reactive
pytest -q tests
pytest -q
python -m tools.navsim_candidate_relative_audit.run_oracle_probe --split trainval --max-scenes 500 --mode full --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.audit_v2_extensions --split trainval --max-scenes 500 --mode full --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.audit_future_visual_anchor --split trainval --max-scenes 500 --mode full --num-figures 12 --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.visualize_audit --split trainval --max-scenes 500 --mode full --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.generate_final_report --split trainval --max-scenes 500 --mode full --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.generate_final_report --split trainval --max-scenes 500 --mode full --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.visualize_audit --split trainval --max-scenes 500 --mode full --output-dir reports/navsim_candidate_relative_audit
python -m tools.navsim_candidate_relative_audit.generate_final_report --split trainval --max-scenes 500 --mode full --output-dir reports/navsim_candidate_relative_audit
