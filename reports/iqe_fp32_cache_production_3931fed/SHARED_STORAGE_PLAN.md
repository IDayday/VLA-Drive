# Shared storage and protected capacity

Verified project mount: `100.97.192.3@tcp:100.97.192.4@tcp:/hbTcp001/78ebbad2386e922193d44ac249bbb12e_kl4gv5vivlcf06o2[/mnt/project]`; filesystem `lustre`,
target `/mnt/project`. Real payloads were built onvlawm-zt3, read onvlawm-zt4 andrl-zt4,
with the same shard/index checksums. Atomic rename/flock and per-record integrity are exercised.
This is project shared storage, not local `/var/tmp` payload staging.

Mounted total 432,814,591,836,160 bytes; observed free **6,000,824,569,856 bytes**.
Per-project quota is **NOT_VERIFIED**: no `lfs` client is installed. `statvfs` capacity
and successful cross-node actual writes are evidence; they are not a fabricated project quota.

| Budget item | Bytes |
|---|---:|
|90892 FP32 payloads, measured56687046 each |5,152,398,985,032|
|6400 payloads physically reused with verified hardlinks |362,797,094,400|
|Remaining new payload |4,789,601,890,632|
|5% payload/index growth allowance |257,619,949,251|
|60 remaining original1000-step checkpoints |1,368,979,984,260|
|22k parallel-work reserve180GiB |193,273,528,320|
|Safety200GiB |214,748,364,800|
|8 atomic shard temporary slots |17,179,869,184|
|Required free before launch |6,841,403,586,447|
|Additional free needed at inspection |840,579,016,591|

The full build was refused before GPU model allocation. No checkpoints, existing
cache shards, historical evidence or other tasks were deleted. `build_full.py`
rechecks real capacity before launch; builder checks before every shard and
honors pause/rate controls. Full-cache cost/coverage remain unmeasured.
`reuse_*.bin` hardlinks are excluded from newly written `shard_*.bin` accounting;
shared blocks are not counted twice when resuming.

When capacity permits, an authorized idle host may build with an explicit worker
and IO budget (default8,128MiB/s per worker). The current hold is conditional,
so no new approval is needed for this already-authorized cache build. It still
must preserve active jobs and all documented reserves.
