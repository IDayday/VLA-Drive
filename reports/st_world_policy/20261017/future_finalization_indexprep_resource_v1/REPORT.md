Independent future-index preparation resource review
===================================================

**PASS_CODE_RESOURCE_ONLY**, fixed source
`99ff3af1b255cf3b37b13954850e8b657c6b493f`, reviewed by
`/root/st_owned_descendants_review_v2`, independently of implementation author.
This is not an actual index preparation/build admission or full-population PASS.

Seventeen tests passed this invocation: nine author-defined controls actually
rerun and eight independently added resource/integration cases. The actual
frozen3df runtime was imported, and its unmocked finalization receipt validator
accepted a two-scene synthetic full-population receipt and rejected a modified
shard-receipt SHA. During receipt validation a guarded metadata reader rejected
any access to request bodies, source log shards, raw logs, images or tensors;
all accepted data reads were synthetic metadata. Fixture creation itself used
tiny synthetic inputs, never real campaign assets or an actual storage ledger.

Run:

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' taskset -c 8,9 python3 reports/st_world_policy/20261017/future_finalization_indexprep_resource_v1/run_bounded.py
```

The runner launches DDP Python3.10 in the fixed runtime with bytecode disabled.
Whole invocation:4.926527s, sampled aggregate RSS647,155,712B, child sampled
HWM636,383,232B, post-wait parent logical read142,660,054B (child included once).
There were243 samples and one terminal sample without RSS fields; this is
explicitly unavailable, not an observed zero. Budget:180s/1GiB RSS/256MiB total
read, synthetic read/write each32MiB. Fixture logical size82,619B; fixture-stage
read1,831,958B/write82,885B. No actual task signals, GPU, payload or ledger reads.

Import is material:3.250987s and136,088,431 logical bytes in this environment.
The two validator calls took0.012590s and436,024 bytes, while the helper's
counter advanced400,114 bytes. Thus a32MiB helper counter is not whole-process
accounting. The adapter correctly labels that boundary. The eventual actual
preparation controller must start its wall/read baseline before imports and
measure process RSS and all logical reads; it must not advertise a32MiB total
process allowance. This synthetic result establishes feasibility within our
bounded review, not an actual full-population byte guarantee.

The dynamic plan is exact: existing metadata declarations plus retained
administration/monitor/control allowances5525MiB, pending DEV index128MiB,
pending TRAIN index2048MiB, total7701MiB within8192MiB. Adding the two pending
indexes twice fails the envelope test. Policy objects and old claims remain
unchanged; proposed registrations are separate future outputs. `--max-disk-bytes`
binds output+staging and the separate metadata reservation remains counted.

TRAIN requires the new COMPLETE schema and its separate successful terminal
receipt, zero owned survivors and completed read accounting. The original
DEV COMPLETE/BUILD_COMPLETE attempt2 route and prior-failure pins are preserved.
Missing finalizer outputs return WAITING without creating preparation output.
New index callback pins use the finalizer-compatible frozen runtime; old TRAIN
PLAN source identities are still verified against the original frozen tree.
The helper and all five declared index-runtime files were hash checked; the
existing callback builder consumes its exact three callback dependencies.

Remaining: actual finalizer terminal validation, actual current ledger pins,
separate preparation admission and then index build admission are still needed.
No actual finalizer result was read in this task; the active finalizer was not
interrupted. No original failure was relabeled. Full101592-scene index cost and
downstream PROFILE compatibility are not established by this review.
