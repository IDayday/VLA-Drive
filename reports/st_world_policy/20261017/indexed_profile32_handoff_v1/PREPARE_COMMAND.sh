#!/usr/bin/env bash
# NOT EXECUTED. Run ONLY inside a newly admitted bounded one-shot CPU controller.
# CPU2, RSS2GiB, 180s, logical-read4GiB, root-inclusive metadata32MiB, no GPU.
# This revalidates index metadata and reads the original 1,590,864-byte PCA basis.
set -euo pipefail
cd /mnt/project/VLA-Drive-st-indexed-profile-runtime-v1-20261017
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
test "$(git rev-parse HEAD)" = 2d5e9483a8c8bc80b71348211756f3cab6aa21b1
test -z "$(git status --porcelain=v1)"
test ! -e /mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_native2_v1/profile32_v1
/root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.qualify_indexed_provider prepare-profile-migration \
  --parent-descriptor /mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_native2_v1/PROVIDER.json \
  --parent-runtime /mnt/project/VLA-Drive-st-indexed-qualified-runtime-v1-20261017 \
  --output /mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_native2_v1/profile32_v1/SOURCE_MIGRATION.json
/root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.teacher_provider \
  --request-index /mnt/project/st-world-policy-20261016/caches/future32_request_metadata_index_v1 \
  --source-index /mnt/project/st-world-policy-20261016/caches/full_current_metadata_index_v1 \
  --teacher-config /mnt/project/st-world-policy-20261016/audits/history_data/native384_train64_cost_v1/resolved_config_build_rank0.json \
  --qualification-result /mnt/project/st-world-policy-20261016/audits/indexed_provider_composition_run_v1/RESULT.json \
  --output /mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_native2_v1/profile32_v1/PROVIDER.json \
  --max-cached-bytes 14159232 --max-scene-mib 64 --max-batch-images 8 --max-metadata-cache-mib 16 \
  --training-config /mnt/project/VLA-Drive-st-indexed-composition-20261017/reports/st_world_policy/20261017/indexed_profile32_handoff_v1/INPUT_PROFILE_CONFIG.json \
  --pca-fit-cache-root /mnt/project/st-world-policy-20261016/audits/history_data/native384_train64_cost_v1 \
  --pca-fit-cache-index /mnt/project/st-world-policy-20261016/caches/original_pca_fit64_metadata_index_v1 \
  --configured-output /mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_native2_v1/profile32_v1/configured \
  --profile-parent-descriptor /mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_native2_v1/PROVIDER.json \
  --profile-source-migration /mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_native2_v1/profile32_v1/SOURCE_MIGRATION.json \
  --profile-data-manifest /mnt/project/st-world-policy-20261016/audits/indexed_composition_preparation_v1/SELECTED_SCENES.json \
  --profile-reference-config /mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_native2_v1/configured/training_config.json
PYTHONPATH=/mnt/project/VLA-Drive-st-indexed-profile-runtime-v1-20261017 \
  /root/miniconda3/envs/ddp/bin/python \
  /mnt/project/VLA-Drive-st-indexed-composition-20261017/reports/st_world_policy/20261017/indexed_profile32_handoff_v1/relocate_config.py \
  --factory-root /mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_native2_v1/profile32_v1 \
  --checkpoint-output-root /mnt/project/st-world-policy-20261016/profile/world_indexed_provider_profile32_updates64_v1
