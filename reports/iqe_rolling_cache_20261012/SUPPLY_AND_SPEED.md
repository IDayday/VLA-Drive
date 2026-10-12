# Measured continuation and bounded supply control

The existing Base is running on training-vla-zt2 from the complete state at
step46734, after200 real updates proven exactly equal to online FP32. Its target
remains100000 optimizer steps. The complete step47000 receipt is archived here;
live progress must be read from the runtime, not inferred from this snapshot.
The two training-vlawm-tsinghua hosts each supply eight frozen-visual workers.
No new model was initialized. Locked22k is unchanged.

## Real timing, all waits included

The measurement protocol was written before steps47010–47209 were observed.
It fixes200 consecutive updates after both producer hosts reached the96MiB/s
per-worker configuration. No slow update is removed.

| Same combined FP32 recipe | Mean update | p50 | p95 | Scope |
|---|---:|---:|---:|---|
| Online visual reference |2.470295s|2.468443s|2.513398s| rank0,189 steady updates |
| Active rolling continuation |1.235289s|1.115368s|1.631981s| rank0 synchronized live events,200 consecutive updates |

Global batch is32 in both runs (microbatch4 on8 A80080GB GPUs). Rolling throughput
is25.9049 scenes/s,1.99977 times the online reference. The rolling maximum is
7.21943s; peak CUDA allocated is62,985,607,168 bytes (58.660GiB), and reserved is
66,603,450,368 bytes (62.029GiB). All four losses and gradient norms were finite.
The used-parameter identity stayed fixed. No cache miss, corrupt record or
sampler mismatch was observed in this window.

Exposed batch/cache wait is19.9771s over247.0577s of measured update time:
8.086%,0.099886s/update on average. Eight of200 steps wait over0.1s. Later rolling
monitor windows also show low water and supply pressure; the cache has not
achieved zero-wait supply. Reporting only the1.115s median would hide this.

Formal per-rank final reports are written when the long job ends. These live
numbers are rank0 synchronized wall measurements, not a directly measured
slowest-rank series. The preceding bounded200-update test contains all8 ranks:
online slowest-per-step mean2.470983s/p502.470128s/p952.514018s; rolling
mean2.005903s/p501.122865s/p951.433672s, including a real131.0456s producer encoding
failure/outage. That failure is retained separately from the healthy formal
window, with the UTF-8 launcher correction documented.

The bounded checkpoint write cost was47.6176s. At the preserved1000-step save
interval its reference amortization is0.047618s/update. This is not represented
as a direct measurement of the ongoing formal47000 write. All checkpoints and
their prior receipts remain in place.

## Supply, demand and storage control

The producer loop follows the immutable original sampler:16 updates per window,
current window plus two future windows. A complete future window is retained
until needed; once all requested windows are complete, producers idle. Reclamation
keeps the previous window and requires exclusive producer/read leases. Actual
consumption, never the speculative/prepared cursor, controls checkpoint recovery.

The rolling pool has a160GiB hard capacity and a separate1,825,361,100,800-byte
checkpoint/storage reserve. Existing permanent payloads are only referenced;
the pool cannot remove them. Snapshot storage has approximately43.6GB of owned
resident payloads and4.18TB shared free space, with zero deleted-open bytes.
These are time-stamped measurements, not quotas or promises about future users.

Sixteen producers are individually limited to96MiB/s. The1.5GiB/s sum is a
configured average write ceiling, not measured sustained throughput. The initial
64MiB/s ceiling was raised by replacing only the identified producer processes,
one host at a time at committed-shard boundaries. Base was not signaled.
There is no precision change or producer-side training. Raising the write ceiling
further is not assumed to fix compute, identity hashing or shard completion tails.

`scripts/iqe/rolling_cache/monitor.py` samples every30s and writes checked JSON
status and history. It measures completed-update consumption, committed fresh
record production, contiguous ready updates, exposed wait, resident/allocated
bytes, free-space reserve, process identity and deleted-open files. Demand uses
non-wait update time and excludes the existing permanent-cache hits. Prepared
updates are never reported as trained updates. Low water is20s; stale consumption
alerts after180s. The observer does not modify the model, sampler, rate, files or
processes. The producers themselves provide bounded demand-driven scheduling;
automatic capacity scaling or automatic trainer restarts are not implemented or
claimed.

The current buffer is frequently about6–16 updates and does not consistently
reach high water. Example fresh supply is12.35 records/s (~700MB/s) over30s;
per-window demand varies with permanent cache coverage. The completed-window
tail takes approximately19–22s versus roughly18s to consume16 ready updates.
This is a mild persistent supply limit, not just one transient alert. Waiting
preserves sample order and complete labels, rather than silently skipping data.
If supply ceases for900s, all ranks fail together and can resume from the last
complete optimizer-boundary checkpoint. The monitor is an alerting observer,
not an unattended repair service.

## Cost boundary

The2x figure describes Base training throughput, not total allocated GPU cost.
During the measured online control,8 GPUs were allocated. The current rolling
system reserves8 training GPUs plus16 remote visual GPUs. At measured rates,
allocated GPU-seconds per1000 scene exposures are approximately617.6 online
and926.5 across all24 rolling-system GPUs. Producer idle time is included in
this conservative reservation count. This deployment prioritizes completing the
same Base sooner using the user-authorized spare servers; it does not establish
a reduction in total billed GPU time. Permanent-cache reuse and actual GPU
utilization should be reported separately from reserved hardware cost.

Scientific gain remainsUNTESTED. No new navtest or deployment ACTIVE change was
made for this supply-control qualification.
