# Actual builds and recovery

| Dataset | Requested/completed | Payload bytes | State |
|---|---:|---:|---|
| Real pilot |64/64 |3,627,970,944 |COMPLETE; full shard and64-record verification PASS|
| Actual next200 batches |6400/6400 |362,797,094,400 |COMPLETE;267 shards, full shard verification and200-update read test PASS|
| Separate stage_val lifecycle |8/8 |453,496,368 |COMPLETE; not admitted to training access list|
| Full original fit |90892/0 new full-build records |5,152,398,985,032 projected |BLOCKED_SHARED_CAPACITY|

Representative index SHA:
`77bc6e3d1c07aaeb78df706838d638788ea6fbe3f9e2be5a84c12a3016b7ccd9`.
Path `/mnt/project/iqe-fp32-cache-production-runtime-3931fed/representative200`.6400 unique observations are the checkpoint sampler's
actual next200 global batches, not a repeated64-scene warm cache.
Full plan `/mnt/project/iqe-fp32-cache-production-runtime-3931fed/FULL_PLAN_40K.json` hash
`a7c81efc3cd71e4e26edd6934d0ef8fcc070005b3f9bfe6a80b251b666d4a837`.

Representative build on8 authorized idleA800s:15:52:16 ->16:12:51UTC, approximately1235s
including model/metadata loading and global verification. Per-worker build loop
597–615s. Metadata JSON totals22,063,098 bytes,267 data files. No million-file layout.

Actual pilot failure evidence is preserved: tail shard initially used a full-shard
budget; resume exposed canonical-JSON key ordering changing shard membership.
Both were fixed (tail-aware projection; plan-order restoration/content-bound shard IDs).
The conflicting orphan was retained, never silently admitted; effective64-record
payload size above excludes that orphan's extra physical allocation.

Estimated, not measured, full-build wall from1235s×90892/6400 is4.87h at unchanged
resources/storage. The paired per-update saving is1.24762s; a4.87h build amortizes
in roughly14050 updates. Remaining60000 updates at that saving would save20.8h
before subtracting build cost. Full scale IO/long-run concurrency may differ;
this estimate is not a full-run speed claim.
