# SP0 dev16 typed-W: concrete preparation, no migration or extraction

Status **PREPARED_VALIDATED_NOT_MIGRATED**. Exact frozen runtime remains `581399eb309677c20fe7464791b4f18685e947a6`. Private immutable materials are in `/mnt/project/st-world-policy-20261016/audits/spatial_probe_actual_SP0_5000_v1/storage_preparation_dev_v1/`, inside the already reserved32MiB SP0 control root. They occupy77,824 charged bytes. This is a resource preparation/review, not implementation of the storage module or permission to self-dispatch.

`PREPARATION.json` SHA `c40594ffaded900a5c3cfb642cc30d8e8f2404029059138a9e8870a98c25731d` binds every file. No live ledger/policy was changed and no cache directory was created. The actual dev plan already completed; it was not repeated. Its **actual** SIGNATURE SHA `3de51e64aec7f555f8d5c053695656346e1549796dad08c2c5bdafa88a26cff4` supplies the claim, not the similarly hashed expectation file.

## Exact new member and preserved envelope

New member root `/mnt/project/st-world-policy-20261016/caches/learned_w_probe_SP0_5000_v1/dev`, identity `bccc958c07f1ff9ce1618221abe8763337cab742243975f2f7ae47180819b673`, native SP0 5000 endpoint,16 fixed dev observations. Each payload has W float32[144,2048] and bool valid[144]. No train64/other arm/teacher tensor claim is added.

| W claim component | Bytes |
|---|---:|
| output | 19,007,744 |
| staging | 5,374,096 |
| metadata | 1,703,936 |
| Total | **26,085,776** |

New immutable proposed policy SHA `0bf0dc467a83a378e02ab4c52e00c7f2836011985edc5bf76d4fb2135957f191` is v7 plus this single full typed claim and a <=1GiB W package. All seven old declarations, admitted rows, legacy reservations, original metadata package and baseline,768GiB cap,1TiB floor and same ledger/lock are preserved. Live ledger SHA was still `7fcfc0c18fa69fc65a125ee36acc1204f9976e62926bede016269136f79cebbe` before and after preparation. A migration changes the core/package declaration; W's actual ledger claim is admitted later by the existing extractor `reserve`, not manufactured in advance.

Metadata core commitments remain1136MiB. Administrative reservations remain qualification reports64MiB + SP0 controls32MiB + completed factory namespace32MiB, giving **1264MiB/4096MiB**. Carry-forward SHA `1b3a22906bfebaa99bd425e855d9ec73550a4b34013490bb815814d1847aa530` binds these separately. They are **not falsely represented as already-admitted ledger claims**. Including their128MiB headroom, declared campaign allocation is694,256,684,278B <824,633,720,832B. Do not spend that administrative headroom elsewhere or double-count it if later genuinely adopted. Empty adoption inventory is intentional: no growing control root is mislabeled an immutable completed pilot.

Only the original torch-free validator was used: actual `claim_from_signature`, policy `_read_policy`, metadata/W `validate_package`, and `validate_migration`. `_read_policy` was invoked without constructor side effects or a ledger lock. The ordinary <=32MiB run-identity metadata check was retained; no bypass. Preparation took0.562s with recorded metadata-open size sum22,244,399B and peak RSS1,569,472,512B (<2GiB). A Python audit hook rejected model/image/tensor/SQLite opens and writes outside the new control directory. No native payload hashing, image open, GPU or actual migration occurred. This is bounded metadata-validation evidence, not W numerical/extraction evidence.

## Fresh checkpoint-space commitment

`CHECKPOINT_COMMITMENTS_2103G_V1.json` SHA `60606edc286a0c34016842a03f0de08f2422936e347364c7373defe9efcc5259` records fresh `statvfs` at **1792223049.4307477**, free7,491,246,944,256B. It conservatively retains **all1983GiB** original nine-run budgets plus **all120GiB** registered indexed student smoke budget:2,258,079,055,872B. None is subtracted because a checkpoint already exists. This is future commitment/headroom, **not measured consumed storage**. The separate1TiB floor remains1,099,511,627,776B. Quota is not independently verified and is not falsely marked verified.

Receipt validity ends **1792226649.4307477 (2026-10-17 08:44:09 UTC)**. Its original1983GiB evidence is a pinned historical budget basis, not an extension of the expired old lease. New validity comes from the current stat plus the conservative full budgets authorized for this task. Migration and extraction must still check this receipt freshness and current free space; if delayed beyond its window, obtain a new conservative fresh receipt. Do not edit this one.

## Lead handoff and exact migration

1. Complete **fresh** writer/queue observations on currently authorized hosts for every old declared/legacy root. The pending quiescence template lists exact roots and all new-policy/old-ledger/inventory/checkpoint pins. It deliberately has `PENDING_LEAD_LIVE_WRITER_CENSUS`, null claims of no writers and no process evidence, so the migration will reject it. Read-only ST training and the newly registered checkpoint-writing smoke are not cache-ledger writers and remain protected; their checkpoint budget is already included.
2. Lead publishes an immutable `QUIESCENT` record with actual timestamps/expiry, pinned observations, no active/unknown old cache/PCA/metadata writer, queue disabling and the exact template bindings. No source rerun, full cache scan or payload hash is needed for this census.
3. Run the included `MIGRATE_AFTER_LEAD_QUIESCENCE.sh` only after the exact evidence is approved. The only unresolved shell inputs are the real new quiescence path/SHA; every other path/hash is an actual prepared artifact. It invokes the existing same-ledger migration CLI, CPU2/CUDA hidden, bounded180s+30s cleanup. Retain stdout and original pending/snapshot/complete artifacts. No automatic retry, deletion or rollback of rows. A changed ledger safely rejects this stale transition; do not silently repin it.
4. The resulting actual `MIGRATION_<id>.complete.json` under the existing state directory must say MIGRATED and bind this policy. Only then prepare a separate `dev_extract` admission using the unchanged supervisor. This report does not supply an APPROVED GPU token.

## This dev extraction's finite resource request

| Resource | Exact finite request |
|---|---|
| Observations | fixed16 dev, actual signature above |
| GPU | one freshly idle authorized GPU with actual UUID; none selected here |
| CPU |2, fixed affinity[0,1] |
| Process-tree RSS |64GiB observed limit |
| GPU allocator peak |24GiB |
| Cooperative / outer / abnormal cleanup |900s /960s /40s |
| Dev plan+extract logical-read allowance |256GiB total |
| Already consumed by plan |37,533,999,575B (max known-file budget and rchar) |
| Remaining extraction read allowance |**237,343,907,369B** |
| W storage |26,085,776B total typed allowance above |
| Stage control/log allowance |5MiB logical inside the existing32MiB SP0 envelope |
| Extra teacher/visual cache output |0 |

Full extraction needs the original model once and real inputs; those costs are not claimed zero. It retains the existing FP32 W/normalizer/current-only source contract and creates no new teacher cache. Exact production speed/RSS remain to be measured. The stage's GPU/host/lease check is independent of this storage preparation; preserve the current GPU0 smoke and other jobs. The shared one-loader lock serializes plan/extract/fit. No automatic retry; retain any partial attempt. New train64/other arm membership later needs another additive declaration after its own actual signature, without replacing this member.

The executable extraction entry after a separately issued exact admission remains:

```bash
env PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python \
 /mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017/reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1/supervise_once.py \
 --admission "${ST_DEV_EXTRACT_ADMISSION:?fresh independently issued dev_extract admission}" \
 --admission-sha256 "${ST_DEV_EXTRACT_ADMISSION_SHA:?exact admission hash}"
```

It must bind original PREPARATION `34c174a6...`, v2 helper hashes, actual migrated policy/receipt and the2103GiB checkpoint receipt above. Lease must cover the whole outer window and cleanup; no source/runtime/cache/model identity is changed by this handoff.
