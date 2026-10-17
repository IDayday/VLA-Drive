# Next typed-W migration: preparation only, not approved

Do not change v7, restore an older ledger, or migrate before obtaining actual plan signatures and fresh exact evidence. Preparation's expected signatures are not actual cache-plan receipts. Eight planned roots total 546,104,992 bytes and SP0 train+dev total 112,340,512 bytes, but these are ceilings from the prior preparation, not eight admitted or extracted caches.

1. After the admitted dev plan succeeds, retain its `RESULT.json`, `COUNTERS.json` and actual `SIGNATURE.json`; verify identity `bccc958c07f1ff9ce1618221abe8763337cab742243975f2f7ae47180819b673`. Use `tools.st_world_policy.spatial_w_storage.claim_from_signature(root=..., claim_id=..., signature_path=..., signature_sha256=...)` to prepare its exact typed claim. This is a control validator, not extraction. Never pass an expected-signature path as an actual result. The fixed controller orders dev extraction before train plan, so the minimal honest first migration admits dev only; later train/other membership additions require explicit additive migration once their actual signatures exist. Alternatively a separately reviewed planning order is required before declaring more actual-plan claims.
2. Prepare new immutable policy from actual v7 and then-current ledger. Preserve all seven declarations, previous reservations, metadata package and baseline exactly. Add `frozen_w_package` with schema `st_frozen_spatial_w_storage_package_v1`, <=1 GiB maximum, exact old policy/ledger pins, full original baseline, and full exact members. Do not change an unadmitted member later; additions use migration. Preserve 64 MiB qualification-report and 32 MiB SP0-control administrative reservations in effective 4 GiB metadata and 768 GiB campaign totals. Do not mislabel W tensors as metadata.
3. The existing migration accepts additive legacy reservations only for inventoried pilots and requires exact MiB-rounded **observed charged bytes**. It cannot honestly turn an actively growing 32 MiB control allowance into an immutable completed-pilot inventory by fabricating physical bytes. Keep the explicit administrative allowance/effective-total receipt for ongoing controls; adopt closed report roots with genuine stat inventory when appropriate, and preserve the still-unspent allowance without double counting. A migration claiming its ledger alone accounts for all active administrative headroom would be inaccurate. Lead must make this explicit in the next exact package review, without changing the original core or opening a second ledger.
4. Acquire fresh Lead quiescence covering old declared/legacy writer roots and the exact inventoried additions, with queues disabled and no unknown writers. Read-only training remains protected. Pin fresh process evidence, new policy, old ledger, pilot inventory and checkpoint commitments. Earlier approvals/censuses are not reusable by assertion. Checkpoint commitment must cover migration plus each extraction's whole 960-second outer window and cleanup; current V1 expires around 07:47:25 UTC and may need refresh. Extractor enforces floor plus remaining commitments on every reserved write.
5. Only after independent review use the existing CLI below. Variables intentionally refer to **not-yet-produced** evidence; this is the exact interface, not a claim those files exist or execution is approved. Source entry is verified at the frozen runtime. Migration writes pending/history/snapshot/state/receipt under the one existing `CLAIMS.lock`, preserving old rows and failed windows.

```bash
cd /mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
 /root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.metadata_storage migrate \
 --old-policy /mnt/project/st-world-policy-20261016/resources/CACHE_STORAGE_BUDGET_POLICY_v7_index_followup.json \
 --old-policy-sha256 b7062c4da878519c435db2e88b899bd476e5cd6a0e9f16dca191fe61522bb831 \
 --new-policy "${ST_W_NEW_POLICY:?new independently reviewed immutable policy}" \
 --new-policy-sha256 "${ST_W_NEW_POLICY_SHA:?}" \
 --expected-old-ledger-sha256 "${ST_W_LIVE_LEDGER_SHA:?fresh exact ledger; do not restore old snapshot}" \
 --quiescence-receipt "${ST_W_QUIESCENCE:?fresh Lead evidence}" \
 --quiescence-sha256 "${ST_W_QUIESCENCE_SHA:?}" \
 --pilot-inventory "${ST_W_PILOT_INVENTORY:?exact actual retained-file inventory}" \
 --pilot-inventory-sha256 "${ST_W_PILOT_INVENTORY_SHA:?}" \
 --checkpoint-receipt "${ST_W_CHECKPOINT_COMMITMENTS:?current full-window receipt}" \
 --checkpoint-sha256 "${ST_W_CHECKPOINT_COMMITMENTS_SHA:?}"
```

There is no GPU/extract approval here. Actual extraction additionally requires its own fresh idle GPU UUID, stage admission, completed migration receipt and exact signature/claim binding. No current training runtime, policy, cache or checkpoint was modified in this review.
