# Independent resources review: fixed TRAIN road-failure diagnostic

**APPROVE_ROAD_DIRECTION_EXECUTION**, resources scope only, for registration
`c9c76ffd513fcf3cf59e1753358bd552b0635ae6394399e509d15c09dd9cda0a`
at source `864619b0130fb3b220b577dd34b993ffeac97e27`.
This reviewer did not implement the runner or dispatch proxy/official scoring.
Lead admission and a distinct correctness reviewer remain mandatory.

The one executable placement is local `training-vla-zt-worker-0`, CPU affinity
0–1, one numerical thread and one sequential scoring worker, all CUDA devices
hidden. At 2026-10-18 19:07:45 UTC, the host had 128 available CPUs, CPU0/1
short-interval busy fractions 0.04/0.12, 901,240,044 KiB available RAM and
3,519,337,844,736 B free shared storage. The snapshot is an observation, not a
future reservation. Lead must recheck the current policy and host before launch.
There is no need to occupy GPUs or compete with the active zt5 formal work.
No real/unknown job or pressure process was signalled.

The real, already completed asset-binding receipt reports 32/32 bound scenes,
31 original ledgers, 32 rank payloads, 130,602,424 B logical reads, 4.476 s wall,
2.054 CPU seconds and peak RSS 2,095,808,512 B. This is close to the separate
2-GiB *binding* cap but below it; it is not an execution throughput benchmark.
It performed no proxy, official call, model load or CUDA initialization.
The diagnostic execution retains its original 8-GiB aggregate RSS limit.

I independently matched all eight registered source hashes and the clean
execution source SHA. I checked sizes of 202 unique immutable source assets
and 32 compressed contexts using stat only. The worker plan is exactly:
130,660,606 source B + 8,690,956 context B + 275,245 selection B =
139,626,807 B, below the unchanged 268,435,456-B logical asset cap.
No actual metric-cache content, image, model or raw trajectory tensor was read
by this review. Small source/control files and previously written JSON receipts
were read; the recorded fixed plan remains subject to runtime checksum checks.

The narrow new mode retains one worker, threads=1, 1,800-s wall, 32-MiB/s
logical read throttling, 64-MiB expanded-cache maximum, 8-GiB aggregate
parent/worker/tracker RSS plus worker address-space cap and 1-GiB output cap.
These are sampled/process controls, not a storage quota: the parent samples
RSS and output every 0.2 s and may overshoot between observations. Each
ordinary atomic write reserves temporary-file and 2-MiB terminal space.
The parent retains the run lock through TERM / 15-s join / owned-child KILL /
3-s join cleanup; it never targets an unrelated PID or process group.

The original `bounded_unpack`, output scanner, verified-context binding,
child entry, geometry/objective and tracking functions are AST-identical to
the actually exercised 0d0d19dd version. `Budget` adds CPU observation and
an optional CPU cap, and `supervise` explicitly resolves the versioned limits,
rejects schema mismatch and preserves frame-derived denominators. These
deltas were inspected rather than incorrectly described as AST-identical.
An initial overbroad static assertion is documented in REVIEW_HARNESS_NOTE.

New calls are durably charged **before** backend invocation. The persistent
output-wide ledger counts failures and uncertain attempts. Failure mode allows
only negative/positive/fixed-random variants, at most three per scene and 96
total; it forbids a new original call or repeated variant. Resume requires new
Lead admission and continues only untouched scenes. Started uncertain scenes
are retained, not retried or substituted. Originals and safe siblings are
reused only after input/context/source checks; the sibling is not a scored
new candidate. The reviewed direct engine does not create a process pool.

**Current independent fixtures: 14 passed**, 8.47 s pytest / 9.810 s subprocess
wall / 8.930 waited-child CPU seconds / sampled tree RSS 1,283,792,896 B.
The fixture runner bounds its own process tree to 2 GiB and wall to 600 s.
Coverage includes aggregate RSS/output stops with owned-worker exit, native
exit, monitor exception, TERM, INT, deadline, hash/read growth, decompression,
exact new-mode admission, durable failed-call charge, 3/case and 96 total.
These are synthetic engineering tests, not real road-repair evidence.

The prior actual-v3 record (read-only inherited evidence) reports 21 new calls
plus 12 preserved prior-version calls, 61.161 s wall and aggregate peak RSS
1,520,934,912 B. Its proxy/backend functions are unchanged here, but its
DAC-compliant population differs. Those costs support this bounded single-CPU
topology; they do not predict exactly how many new directions will be defined
or prove a future 96-call runtime. No enlarged worker topology is warranted.

Approval authorizes only this bounded diagnostic's resource plan. It does not
approve mathematics, DFQL/formal proxy training, held-out benefit, a new GPU
job, feature-side work, retry-budget expansion or any new scene selection.

