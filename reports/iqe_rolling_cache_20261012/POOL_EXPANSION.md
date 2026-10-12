# Producer buffer expansion, 2026-10-12

The user allowed a larger dynamic cache. The inspection found54.7GB resident
under the171.8GB (160GiB) budget and3.73TB shared free space. No pool-capacity or
reserve-wait error appeared in the bounded current producer log tails. Recent
training waits were caused by supply timing, not a full cache. A larger disk
limit alone would not increase producer throughput.

The producer horizon is expanded from3 to8 windows, each16 original optimizer
updates, with a384GiB storage budget. This allows up to128 updates of preparation,
including the current partial window. Worst-case all-new payloads, the retained
previous window,5% overhead and16GiB atomic-write allowance require291,454,472,551
bytes, below412,316,860,416 allocated bytes. The existing1,825,361,100,800-byte
checkpoint/storage reserve is retained. Shared free space at activation was
3,680,376,569,856 bytes. This is permission to fill a bounded buffer, not a claim
that every future window is already resident or that average production demand
can be met by storage capacity alone.

## Identity and safety

The consumer runtime remains8fe9f61, with the same running Base process,
checkpoint lineage, consumed cursor and original100k schedule. No Base signal,
restart, weight replacement,22k change, new evaluation point or precision change.
FP32 payload format, reader checksums, sampler and frozen frontend are unchanged.

The original `SESSION.json` hash remains
`e3131f5ddf53e1b24c8b0f7f53d9f2a3bdc15e064ff00cc282f0996916ddfc21`.
A separate producer policy has hash
`f3fbeb72838ef80d0175eeb325917cc4734531faec843fb29415cfb3b025572c`.
It only permits a bounded expansion of lookahead and residency, keeps the reserve,
and rejects changes to sample/window contents or foreign sessions. The consumer
source worktree is not changed by producer code. The new producer commit is
319543e on `perf/iqe-rolling-producer-index-8fe9f61`.

All16 previous frozen producers were checked by PID/start ticks/argv/cwd and
stopped before activating the new GC horizon. Partial atomic shards remain
recoverable; committed payloads are reused. This avoids an old worker deleting
future windows requested by the new policy. Both new hosts retain their original
absolute24h deadlines,16 GPUs and96MiB/s per-worker ceiling. The Base only waits
for complete windows during producer handover; it consumes no fabricated data.

## Tests and measurement

44 producer/core tests passed, including12 expansion checks: identical window
plans and sampler/RNG under expanded lookahead; original consumer unchanged;
current/future/previous-window GC protection; rewind; end-of-schedule bounds;
policy hash/session/reserve/capacity/unknown-field rejection; explicit activation.
21 monitor tests passed, including separate original/effective capacity reporting
without modifying SESSION or counting unbuilt future windows as ready.
These are targeted storage/scheduling tests, not a new model-training experiment.

The last200 completed updates before the operation were50207–50406:
mean1.76436s, p501.14389s, nearest-rank p954.17264s;124.180s of352.872s was
exposed batch wait. This is a current observational baseline, not a replacement
for the earlier predeclared1.23529s window. Shared filesystem load varies.

The new timing window was fixed before observation:32 warmup updates after50486,
then50519–50718 inclusive, all200 events with waits retained. Producer startup
and handover waits are recorded separately, not silently removed from operational
cost. Results are in `POOL_EXPANSION_AFTER.json` once complete. Do not infer a
speedup merely from the larger budget.

The declared window50519–50718 completed on the same Base PID2495728:
mean1.305857s, p501.120311s, nearest-rank p951.567522s, max8.735344s;
24.50498 scenes/s. Exposed batch wait was37.40095s out of261.17142s (14.32046%).
All200 updates kept global batch32, and losses/gradient norms were finite.
Compared with the preceding observational baseline, wait decreased from35.19%
to14.32%; filesystem load was not controlled, so this is not a randomized causal
estimate of capacity's contribution. The128-update horizon has not been filled:
production rate and window completion remain the limitation, with low-buffer
alerts still observed. Capacity must not be confused with ready data.

The producer handover plus32 warmup updates took187.71791 update-seconds, with
151.72921s exposed wait; the longest update waited through restart and took
130.80456s. No Base process was stopped, no optimizer step was replayed/skipped,
and no numerical path was changed. The startup cost is real and separate from
the steady window, not an excluded hidden outlier. A larger capacity alone does
not guarantee zero long-run cache starvation.

Runtime evidence is under `/mnt/project/iqe-remote-rolling-cache-20261012/`:
`POOL_EXPANSION_BEFORE.json`, `POOL_EXPANSION_ACTIVATION.json`,
`POOL_EXPANSION_TIMING_PROTOCOL.json`, both `PRODUCERS_*_EXPAND_*` receipts,
`MONITOR_EXPANDED_LAUNCH.json`, and append-only `monitor/history.jsonl`.
