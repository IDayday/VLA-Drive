#!/usr/bin/env bash
# Prepared command only. Reviewer did not execute it. Lead supplies real fresh evidence.
set -euo pipefail
cd /mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec timeout --signal=TERM --kill-after=30s 180s taskset -c 0,1 \
 /root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.metadata_storage migrate \
 --old-policy /mnt/project/st-world-policy-20261016/resources/CACHE_STORAGE_BUDGET_POLICY_v7_index_followup.json \
 --old-policy-sha256 b7062c4da878519c435db2e88b899bd476e5cd6a0e9f16dca191fe61522bb831 \
 --new-policy /mnt/project/st-world-policy-20261016/audits/spatial_probe_actual_SP0_5000_v1/storage_preparation_dev_v1/PROPOSED_POLICY_V8_SP0_DEV_W.json \
 --new-policy-sha256 0bf0dc467a83a378e02ab4c52e00c7f2836011985edc5bf76d4fb2135957f191 \
 --expected-old-ledger-sha256 7fcfc0c18fa69fc65a125ee36acc1204f9976e62926bede016269136f79cebbe \
 --quiescence-receipt "${ST_W_QUIESCENCE_RECEIPT:?Lead fresh exact QUIESCENT receipt; template is rejected}" \
 --quiescence-sha256 "${ST_W_QUIESCENCE_SHA256:?exact fresh receipt hash}" \
 --pilot-inventory /mnt/project/st-world-policy-20261016/audits/spatial_probe_actual_SP0_5000_v1/storage_preparation_dev_v1/PILOT_INVENTORY_NO_ADOPTIONS.json \
 --pilot-inventory-sha256 9897233f46386a9e42ee7b6f80c00d21d8a9fe4b8f1d94d5aee88cc559f2ece2 \
 --checkpoint-receipt /mnt/project/st-world-policy-20261016/audits/spatial_probe_actual_SP0_5000_v1/storage_preparation_dev_v1/CHECKPOINT_COMMITMENTS_2103G_V1.json \
 --checkpoint-sha256 60606edc286a0c34016842a03f0de08f2422936e347364c7373defe9efcc5259
