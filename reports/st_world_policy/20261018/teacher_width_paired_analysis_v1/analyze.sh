#!/usr/bin/env bash
set -euo pipefail
cd /mnt/project/VLA-Drive-st-world-policy-20261016
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
exec /root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.evaluate_campaign \
  --plan reports/st_world_policy/20261018/teacher_width_paired_analysis_v1/ANALYSIS_PLAN.json \
  --population /mnt/project/st-world-policy-20261016/audits/history_data/full_current_population_v1/dev/SELECTED_SCENES.json \
  --control-score /mnt/project/st-world-policy-20261016/formal/evaluation/jobs/SP2_step005000_dev1696_seed42_v1/official_score \
  --control-export /mnt/project/st-world-policy-20261016/formal/evaluation/jobs/SP2_step005000_dev1696_seed42_v1/export \
  --treatment-score /mnt/project/st-world-policy-20261016/formal/evaluation/jobs/SP2_PCA256_step005000_dev1696_seed42_v1/official_score \
  --treatment-export /mnt/project/st-world-policy-20261016/formal/evaluation/jobs/SP2_PCA256_step005000_dev1696_seed42_v1/export \
  --comparison SP2_PCA256_minus_SP2 \
  --stage SCREENING \
  --scope FORMAL \
  --output /mnt/project/st-world-policy-20261016/formal/teacher_channel_control_v1/analysis/PCA256_minus_384_dev1696_step5000_v1
