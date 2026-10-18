# Full causal-history metadata preparation

Prepared from existing readiness commit3494bd2d, reusing the completed packed64 and exact32 implementation. The actual builder runtime is the unchanged clean676e3e88 temporal tree. No new builder, model, teacher tensor cache, history mathematics, image transform, GPU job or dataset split is introduced. This directory contains unsigned registration drafts and an explicit resource proposal; it grants no launch permission.

The two fixed populations are the original full-current101,592 TRAIN scenes/1,176 logs and1,696 DEV scenes/16 logs. Paths, original COMPLETE/registration hashes and stat versions appear in PREPARATION_PLAN.json. Selection hashes come from those completed immutable controls. This preparation did not reopen the30.1MB training selection or scan the population. The admitted build must verify the actual selected content/order and each source identity. Keep the two roles separate.

For each observation the existing builder resolves actual causal−1.0s and−0.5s frames in the same log, with at most50ms tolerance, existing missing-view masks and actual camera/ego geometry. `--target-times 0` only writes the separate current-image reference metadata; it does not generate future supervision, DINO features, VLM hidden states, score labels or tensors. Existing current/future teacher and PCA identities remain untouched. Opaque ego payloads are hashed as source provenance without decoding GT values.

## Evidence and realistic costs

The existing independent packed64 result reports64 records/64 logs/64 shards,81.62s supervised invocation,1.991GB final rchar,1.669GB builder source charge,1.654GB aggregate RSS and1,156,349 logical output bytes. All384 historical views were valid in that particular population. This is prior real evidence, not a new full-population measurement. Existing packed correctness44/resource14 tests and the exact32 completed build are inherited; they are not rerun counts.

Using separate per-log and per-scene terms from that64 pilot produces an estimate of about420GB charged source bytes for TRAIN and7GB for DEV, before additional large-manifest/serialization work. Exact values and the formula are recorded in the plan. The32MiB/s pacing floor alone is approximately3.5hours for TRAIN. Thus the native4hour ceiling is tight: it is a stopping budget, not a completion guarantee. Reusing original source pins avoids an extra preliminary population/image scan, but the original builder still hashes actual history/current images and reads their headers. It has no persistent image-hash reuse path. Existing future ImageMemo is not silently inserted into a different reader. No claim of zero image I/O or already-measured full throughput is made.

Packed records preserve original dictionaries. Shards are bounded to128 records/4MiB and stop at log boundaries. Registration allows at most4,096 TRAIN shards and64 DEV shards. The count-only upper estimates are1,960 and29 respectively; byte caps can require more. The existing guard remains max20,000 entries globally; our drafts use9,000 and256, respectively. One Python builder runs at a time on two explicitly admitted CPUs, and TRAIN/DEV run sequentially. No parallel data framework is needed.

Proposal per role:

| Role | Work+cleanup | RSS | Paced source cap | Outer rchar cap | Metadata+controls |
|---|---:|---:|---:|---:|---:|
| TRAIN |14,400+40s|4GiB|448GiB at32MiB/s|512GiB|2GiB total|
| DEV |900+40s|4GiB|16GiB at32MiB/s|24GiB|128MiB total|

These are proposed independent bounds, not permissions or measured demands. TRAIN allocates2,047MiB artifacts as1,792MiB output+192MiB staging+63MiB metadata, plus1MiB sibling controls. DEV allocates127MiB artifacts as96+16+15MiB, plus1MiB controls. Expected full logical output from the prior pilot is roughly1.71GiB TRAIN/29.2MiB DEV; actual missing views, path lengths, partial files and filesystem allocation determine real use. All partial/orphan files remain charged. No checkpoint commitments are released to fund this small metadata allocation.

## Exact existing command and remaining binding

Both complete argv lists are in the corresponding STORAGE_REGISTRATION_DRAFT.json and PREPARATION_PLAN.json. They were parsed through the actual native CLI, `_resources` and packed-population guard without invoking build. For example the TRAIN builder argv is:

```bash
cd /mnt/project/VLA-Drive-st-temporal-combination-20261017
/root/miniconda3/envs/ddp/bin/python -B -m tools.st_world_policy.prepare_causal_history build \
 --student-root /mnt/project/ddp-foresight-artifacts/20260928/student_train_v1 \
 --raw-log-root /mnt/navsim/trainval_navsim_logs/trainval \
 --sensor-root /mnt/navsim/trainval_all/trainval_sensor_blobs/trainval \
 --fallback-sensor-root /mnt/navsim/trainval_sensor_blobs/trainval \
 --partition /mnt/project/VLA-Drive-st-history-population-agent-20261016/reports/ddpolicy_vehicle_from_scratch/NAVTRAIN_PARTITION.json \
 --scene-list /mnt/project/st-world-policy-20261016/audits/history_data/full_current_population_v1/train/SELECTED_SCENES.json \
 --split train --mode registered --max-scenes 101592 --seed 20261017 --tolerance 0.05 \
 --target-times 0 --record-layout packed_log_v1 \
 --registration /mnt/project/VLA-Drive-st-full-causal-history-preparation-20261018/reports/st_world_policy/20261018/full_causal_history_build_preparation_v1/TRAIN_BUILDER_BUDGET.json \
 --output /mnt/project/st-world-policy-20261016/audits/history_data/full_causal_population_v1/train
```

The argv JSON is authoritative and preserves separate flag/value arguments. Bare execution intentionally fails because the original metadata guard is absent. The real bounded child must call the existing `metadata_storage.run_registered_metadata` with the exact source registration and Lead's fresh policy/CP receipts; the original4b2f supervisor supplies ownership, wall/RSS/rchar accounting and40s cleanup. This function is the actual, already implemented guarded CLI:

```bash
python -m tools.st_world_policy.metadata_storage run \
 --policy-path "$ACTUAL_POLICY" --policy-sha256 "$ACTUAL_POLICY_SHA" \
 --registration-path "$EXACT_ROLE_REGISTRATION" --registration-sha256 "$EXACT_ROLE_REGISTRATION_SHA" \
 --checkpoint-receipt "$FRESH_CP" --checkpoint-sha256 "$FRESH_CP_SHA"
```

Do not launch either bare command outside its admitted original outer supervisor. The old64 and exact32 wrappers correctly reject this full population. The remaining controller change is a thin fixed TRAIN/DEV binding of that same runner, including the already reviewed W-source compatibility bridge needed by the current mixed SP0/SP1 storage policy. No arbitrary command support or monitor rewrite is needed. Its selection verification must handle the30.1MB TRAIN list in the admitted read budget; the existing small-control4MiB helper cannot incorrectly read it as a small control. Registration/schema/source pins remain mandatory.

Systems must account the actual current ledger declarations plus retained administrative commitments before appending these exact2,176MiB. The current metadata-package8GiB maximum is an enforced source guard. If the exact sum exceeds it, a separately reviewed explicit envelope-version extension is necessary; changing JSON alone or restoring an old policy will correctly fail. This task does not alter the package, claims, old outputs or live ledger. A fresh CP snapshot/free-floor/host/CPU/time admission must cover the selected role's full bounded window; no expired2hour receipt can authorize4hours.

On interruption the native builder retains immutable shards/partials. Existing `--resume` recomputes the original selection/log/image work and compares the shards; it is not a cheap log-skip recovery. A retry therefore requires an explicit new finite read/time allocation and failed-evidence binding. No automatic retry is proposed. `COMPLETE` is only valid once all original requested scenes appear in order with matching source hashes, the controlled process exits successfully and owned descendants are gone.

## Narrow independent handoff

Review only: exact two role bindings/population source controls; inherited builder/math hashes; current storage policy and small2,176MiB additional accounting; fresh4hour/15minute resource leases; original monitor integration and the large-selection hash boundary. Do not repeat teacher/PCA/model training audits or reread TB images for this metadata preparation. Current actual full-history data availability remains WAITING_INPUTS.

`STATIC_COMMAND_CHECK.json` records actual parser/resource/population-guard acceptance for both draft commands:3.22s,604,936KiB peak RSS and134,904,907B rchar including Python/Torch imports, CPU10/11, CUDA hidden. It read no population rows, raw logs, image bodies, checkpoint or teacher tensors and did not access the ledger. The synthetic `_ACTIVE_GUARD` used solely for native option validation is explicitly not storage admission; `writer.run` was never called. No actual builder invocation or scientific result is claimed.
