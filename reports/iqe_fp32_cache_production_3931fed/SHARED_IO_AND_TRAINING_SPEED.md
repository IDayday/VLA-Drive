# Matched200-update real training comparison

8×A80080GB, global32, micro4/rank, same33k complete checkpoint and original100k
schedule, original4 objectives and dropout/repeat8.200 updates; exclude10 warmup
and1 final queue-drain step ->189 steady updates. No profiler in timing window.
The table uses the **slowest rank per update**, not fastest-rank or isolated-head time.

|Path|Mean s|p50 s|p95 s|Samples/s|Peak allocatedGiB|Peak reservedGiB|Checkpoint s|
|---|---:|---:|---:|---:|---:|---:|---:|
|online|2.450300|2.437893|2.506375|13.0596|60.047|64.332|49.283|
|cached|1.202683|1.186394|1.360314|26.6072|58.663|62.055|45.970|

Cache-only speedup **2.0374×**. At the unchanged1000-update
checkpoint frequency, means are **2.499582s online /1.248653s cached**.
Actual checkpoint artifacts remain22.8GB and are complete/resumable.
Training-loop allocated GPU cost: online3941.36 GPU-seconds, cached1945.18
GPU-seconds (same6400 sample exposure); startup/verification/build costs are separate.

Actual shared reads:6400 records,362,797,094,400 bytes,0 misses, no local staging.
Every rank read800 distinct records/45,349,636,800 bytes. Throughput-equivalent
bytes divided by steady wall is1.508GB/s; it is not a physical Lustre network
counter. Own-file `POSIX_FADV_DONTNEED` was requested; physical cold cache is not
guaranteed. The64-record pilot is not presented as representative IO performance.

Each reader opens at most8 shard handles; total process FD maximum109.
Maximum observed RSS23,948,264KiB (~22.84GiB) per rank. Pinned allocator peak is
not exposed by the installed torch API; only the bounded producer/consumer
buffer count is established, and no pinned peak is fabricated. Reader inclusive
IO/checksum seconds overlap across4 threads and GPU work; do not sum them as
critical-path delay. See per-rank raw counters in `evidence/REPRESENTATIVE_IO.json`.

The original live Base still runs ~3.99s compute/update. Comparing that run with
1.20s combines earlier fused/recompute/prefetch changes with caching and different
backend determinism; only2.4503 ->1.2027 is the matched cache-only comparison.
