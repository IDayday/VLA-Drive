# First formal Flow pair started with no new feature-cache allocation

C0 and C1 have completed real optimizer updates from the same bound original
S0, on the independently reviewed immutable runtime
`6714b1e9fb2e0c8059b90b8b4d48cd40796fe345`. This supersedes only the pending-launch
status in the earlier snapshots; their evidence and limits are retained.

- C0 launched at **2026-10-17 02:44:54 UTC**, GPUs0–3 on tsinghua-zt4.
- C1 launched at **2026-10-17 02:45:06 UTC**, GPUs4–7 on the same host.
- Both actual registrations received separate Evidence, Correctness and
  Efficiency PASS records. The real approval checker returned VERIFIED before
  launch; the launcher also checked fresh GPU UUIDs/availability, locks, the
  authorization, lease and disk reserves.
- The first update in each arm has finite loss and gradient norm and verifies
  an FP32 master update. C0 made zero official feedback calls. C1 generated and
  scored128 actual Euler10 rollouts for32 scenes. These are training observations,
  not held-out scientific results.

The independent startup observer subsequently recorded C0 step25 and C1 step3,
with matching source/binding/approval identities and all four ranks' exact FP32
source-master initialization. Its immutable report hash is
`a6ecc1cfde5d8907934e08e828d235ade530eca610a658656fa04b0e46b8035f`.
This is a bounded startup check, not a full-run frozen-parameter audit.

Exact artifact paths, hashes, initial losses, resource identity and first-update
counts are in [FLOW_FORMAL_STARTED_v1.json](FLOW_FORMAL_STARTED_v1.json). The
evaluation registration now binds these actual launches in
`../20261016/evaluation/PREREGISTRATION_EXECUTION_v4.json`; it adds no evaluation
execution authorization. No diagnostic weight is a formal initializer.

## Fixed execution and storage

Both arms retain world4, micro4, accumulation2 and global batch32, the complete
101,592-scene/1,176-log training population, original BC repeat8, and the
4,096-update schedule with warmup100. C1 adds the previously verified official
PDMS/FPO objective. The original frozen/shared encoder and ten-step direct
single-trajectory deployment remain intact. BF16 execution, FP32 master/Adam
state and ZeRO2 follow the already verified official-runtime adaptation.

The runs reuse existing metric contexts and current images. They allocate
**zero new teacher, condition or metric tensor cache bytes**. Their native
checkpoint and feedback/log files have separate budgets: C0 187GiB and each
policy arm 209GiB as conservative upper reservations, including pause/partial
writes. Those reservations are not current occupied bytes.

The cache decision remains768GiB total including indexes/staging, with existing
current train tensors586.123482GiB and a separate1TiB filesystem free floor.
No multi-TB future cache, cache eviction, precision conversion, or active-run
restart was performed. The bounded on-demand future-label provider remains a
conditional implementation task; it is not claimed to exist or to be qualified.

## Measured efficiency and its limits

[FLOW_RUNTIME_MEASUREMENTS_v1.csv](FLOW_RUNTIME_MEASUREMENTS_v1.csv) contains the
six completed100-update windows from the8-GPU reference and the two concurrent
4-GPU profiles. Each profile completed240 updates; the underlying immutable
review retains startup, checkpoint, CPU-tree and storage evidence.
Each second window includes the final checkpoint: its normal-update statistics
use99 updates, while complete wall throughput includes all100 updates and the
checkpoint. CSV columns distinguish these populations and checkpoint time.

Actual concurrent aggregate throughput was **3.47817 and3.58506 scenes/s** in
the two joint windows. These values use the measured joint spans, not twice a
single-run throughput. The8-GPU reference was on a different host, so the
comparison is a resource-allocation observation, not a pure topology experiment.

Profiles reused64 real scenes. They do not establish sustained cold I/O over
the full population or the formal endpoint completion time. C0 has no feedback
pipeline, so its speed cannot be inferred from the FPO profiles. First-update
times are deliberately not reported as steady-state speed.

The independent correctness review also recorded21 focused tests passing in
5.20seconds and an exact four-rank24-scene tail reduction check. This launch
did not rerun unrelated tests or change the reviewed trainer. Existing W jobs
continue on their original runtime.

## Continuation and remaining work

Use the exact commands in
[FLOW_FORMAL_FIRST_PAIR_RUNBOOK.md](FLOW_FORMAL_FIRST_PAIR_RUNBOOK.md).
Each invocation has an8h wall allowance and bounded save/cleanup grace inside
the original48h lease. Checkpoints preserve optimizer/master, scheduler, RNG,
consumed cursor and feedback identity. Resume only the same registered run after
the previous invocation exits; there is no automatic followup job.

The planned512/1024 diagnostics and4096 endpoint, C2–C4 registration/launch,
formal planning results, and spatial-dependent future/history work remain
unfinished. No navtest, production ACTIVE update, push, cache deletion, or
restart of held historical experiments occurred.
