# Finite indexed-provider factory preparation

**PASS_RESOURCE_PLAN, conditional on Lead-owned bounded execution; not a training approval.** The proposed one-time preparation is feasible within 2 CPUs, 2 GiB observed RSS, 180 seconds, 4 GiB logical reads and 32 MiB total new output. Exact original CLI is in `PREPARE_COMMAND.sh` at the handoff path below. It runs the existing factory, not `prepare_handoff.py` again. The reviewer did not execute either command, load the PCA tensor, rehash the large indices or start a model.

Immutable execution runtime: `/mnt/project/VLA-Drive-st-indexed-qualified-runtime-v1-20261017` at clean `31c3dca2fbc697b419bf343ce05d10be0a4fe13f`. Handoff source commit `a12f176cee7e4b6e235bde26a76c91c13541e9c6` (Lead integration b231a1f8). These source pins differ from SP0 runtime v2 intentionally; never reuse its plan approval to execute this factory.

Exact handoff directory `/mnt/project/VLA-Drive-st-indexed-composition-20261017/reports/st_world_policy/20261017/indexed_smoke_handoff_v1/`:

| File | SHA256 |
|---|---|
| PREPARE_COMMAND.sh | a50c39677a11844babd4345a90eeca2a9cfeeedc3e98abfd16b1adf1166fef20 |
| INPUT_CONFIG.json | 170299300d0090422590c7dd65b2821de8063b3ab3ffbbff475732d0e6d7f5f4 |
| HANDOFF.json | 265926a06be378b5b6e3fb2027bdc6fd04fe628bf6a39a3bcacc19d1f9f43479 |

Actual composition RESULT is 1,976,604 bytes, SHA `a4f4c86f0b5c09d18a58d02e8919dd73e659b48a53ed0d2a25c7d68e497b7a2f`. Its real qualification is inherited; this is not another teacher numerical experiment. Source review traced `teacher_provider.prepare_indexed` → `configure_smoke` → `provider_pca._register_indexed_scope`: only metadata providers with a placeholder executor, the original frozen PCA load/verification and small immutable control writes. No call to the real executor constructor, trainer, teacher forward, refit or new tensor cache. `TeacherPCA.load` does read the **existing 1,590,864-byte PCA tensor**, so do not call this zero-tensor-read preparation. No native model checkpoint or image is needed.

Current-cache index opens occur four times in the original validation path. Approximate known large-file work is `4 × (317326781 + 381902848) = 2,796,918,516 bytes`, plus small controls, fit/request indices, imports and original PCA load/hash. This fits 4 GiB with roughly 1.5 GB initial headroom; it is a source-derived budget, **not measured actual runtime IO**. Do not bypass any hash/identity/PCA check to make it fit. Exceeding budget is an incomplete finite attempt, not grounds for silently extending the allowance.

Pinned source SHA256 values:

- `tools/st_world_policy/teacher_provider.py`: `83410555d7e879407b09039f6a094fa6daac1259d0df7379cd6a220c2954b6bf`
- `tools/st_world_policy/provider_pca.py`: `06b80899a4e5559bf3ca01571d7dae33ccae2f5c9be7cbb86c54db8929ddb223`
- `tools/st_world_policy/teacher_pca.py`: `3909263223263ab42a50738b04381c42b6556eff79b510723222a8e5589d5758`
- `starVLA/dataloader/st_world_indexed_provider.py`: `95a011b022542a38009e572d149bb4eb40263c1cd283877ff27f580a9c126f85`

The new namespace `/mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_native2_v1` did not exist at review. Factory creates four immutable JSON outputs: `PROVIDER.json`, `configured/PROVIDER_CHANNEL_SCOPE.json`, `configured/training_config.json`, `configured/PROVIDER_CONFIG_LINEAGE.json`. Include atomic temporary files, a single bounded log, launch/result controls and directory blocks within **one 32 MiB namespace allowance**. It is not a tensor cache and not a new ledger. No training registration is created. Input config remains SMOKE/world1/micro1/global2/two updates, original 20k schedule and warmup1000; the generated config is still not permission to launch those updates.

Execution conditions: Lead owns the exact new process, CUDA hidden, CPU affinity of two cores and BLAS threads2; 180s finite timeout with bounded own-process cleanup; record peak RSS and logical read counters and stop on their stated bounds; enforce total output32MiB including temporary/error files and a bounded log. No automatic retry or overwrite of a partial namespace. The handoff shell itself does **not** enforce RSS/read/aggregate-disk limits. A naked `bash PREPARE_COMMAND.sh` must not be described as enforcing them. Use Lead's finite owned outer monitor; no new trainer/storage platform is needed. If a limit is reached, retain evidence and report incomplete.

Budget after reserving this independent 32 MiB: prior metadata ledger envelope1136 MiB + existing qualification-report64 MiB + SP0 controls32 MiB + factory preparation32 MiB = **1264 MiB /4096 MiB**. No reuse of another task's reservation, no changes to the 768 GiB cap/1 TiB floor/checkpoint obligation, no claim that these administrative reservations have already been adopted by the ledger. These 32 MiB do not cover the subsequent student training/checkpoints: a separate resource/registration review is required before two-update smoke.

## SP0 actual plan follow-up (small receipt only)

The separately admitted dev plan completed once. Actual `stages/dev_plan/RESULT.json` SHA `dd765726eff1c937c31a1aca0f3cc47db8515f53790f84646ae1b94218857520` was independently read: COMPLETE, 43.6879737899s, observed peak tree RSS693,870,592 bytes. Lead reported hash37,376,393,182B/proc rchar37,533,999,575B, no native loads/GPU and no remaining owned process. Its actual SIGNATURE SHA equals the expected `3de51e64aec7f555f8d5c053695656346e1549796dad08c2c5bdafa88a26cff4`. No rerun. This result still does not authorize W extraction or typed migration.
