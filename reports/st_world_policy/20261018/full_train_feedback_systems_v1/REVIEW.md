Independent Systems verdict: **PASS_BOUNDED_READ_ONLY_ANALYSIS**. There is no
remaining blocking finding for the registered C1 TRAIN ledger analysis. This
does not approve training, new scoring, model loading, feature-side work or DEV
evaluation, and does not assert a full-analysis result that has not been run.

The executable runtime is
`/mnt/project/VLA-Drive-st-full-feedback-runtime-20261018`, clean detached commit
`dd413aea5c848f5b92e0fb6225e2a8e20a9e5741`. Its analyzer bytes exactly match
the implementation at `c830a8932d60a792a6489ecd3504db3abda85dbf`:
`30bb3237019a463e68bb5c0686d902521d814ee99bec228a94de7b537f92f4ca`.
The immutable plan SHA256 is
`2d50faf1c690bd8ebc487504a32ecae9637432e4742c561a9ea5f6442a995b06`;
plan ID is
`4f52f393a2be7ebb69128e8bad401007500b7ae5a6ba0f23e0e58b176506b50d`.
Current dispatch policy SHA256 is
`f494747d8c24439a2a914b65b6cb4a469ed7f7ff866c9caa67de581b1304a4f8`.
Documentation-only integration does not alter these executable identities.

I independently recomputed the pinned metadata partition without opening any
RESULTS payload. The eight shards each contain exactly 512 distinct ledgers;
their total payload is 14,088,142,976 bytes. A payload is read once for hash and
BytesIO decoding, while the budget conservatively charges both actual bytes and
the nominal decoding size. Eight fixed worker quotas plus the 256 MiB control
reserve equal 32 GiB. Requests, admissions, completion records and small rank
receipts are included. Merge checks the final worker budget, not only the
pre-completion snapshot. The 128 MiB output allocation is 8 × 12 MiB workers,
16 MiB merge and 16 MiB controls. It means retained/reserved artifact capacity,
not cumulative filesystem write traffic from atomic budget updates.

Lead's permitted execution is two single-CPU workers on each of Tsinghua zt3,
zt4, zt5 and zt6, using CPUs 112 and 113. Each worker has one BLAS/Torch thread
and empty CUDA visibility. No worker opens a GPU context. The Lead-reported
16:40 resource snapshot found 128 CPUs per host, 78–96% idle CPU, over 800 GB
available RAM per host, and about 3.6 TiB free on the same shared project mount.
CPU PSI and cgroup-v2 telemetry were unavailable; they are not marked verified.
Real workloads must remain untouched. Policy must be refreshed at dispatch.

The rate limiter reserves actual-byte tokens before reads. Each worker permits
4 MiB/s sustained traffic and a bounded 8 MiB burst. Across eight workers this
is 32 MiB/s sustained with at most 64 MiB burst, **not** a claim that every
one-second window is below 32 MiB. Idle time cannot build unlimited credit;
interrupted reservations survive resume. Nominal BytesIO decoding costs count
against the logical-read quota, not physical bandwidth. The initial plan/source
read is bounded and charged as startup overhead after admission.

The workers share one immutable deadline, at most 1200 seconds. Python's alarm
provides normal interruption. Lead must also use the stated GNU timeout wrapper:
TERM ten seconds before that deadline and KILL five seconds later, restricted
to its owned subprocess group. Merge uses the same deadline. No automatic
retry or deadline reset is approved. The 2 GiB RSS check is enforced at IO and
record boundaries through `ru_maxrss`; it is not an instantaneous kernel memory
ceiling. Bounded payload size, one-ledger-at-a-time decoding and the measured
prior single-ledger footprint support this finite scope. Stalls are additionally
bounded by the external owned-process timeout.

Atomic immutable ledger outputs, per-shard flock, persistent consumed budgets
and an immutable shared execution identity protect interruption and duplicate
execution. A failed attempt remains recorded; only explicit resume with the
same plan/deadline can continue unfinished work. Merge requires all 4096 policy
boundaries and the exact 131,064 scene / 524,256 candidate presentations. It
cannot convert partial output into COMPLETE. Source, plan, input and result
hash mismatches fail closed.

Cross-host lock behavior was exercised by Lead and its small receipts were
independently read and hashed here: zt3 held the shared lock, zt4's concurrent
nonblocking attempt was rejected, then zt4 acquired it after owned release.
Receipt paths and hashes are in REVIEW.json. I did not dispatch remote workers.

Validation performed on the final analyzer bytes:

- Existing targeted suite: **10 passed**, 1.99 seconds; one existing pynvml
  deprecation warning. Full captured output is in SOURCE_REGRESSION_TEST.log.
- Independent control checks: **8 passed**, 0.167 seconds. They cover global
  quota arithmetic and partition, token rate/burst/resume debt, read admission
  before opening, output admission, actual duplicate-process lock exclusion,
  real alarm interruption, stale code/limit rejection, and CPU/thread/CUDA
  boundaries. They use only synthetic files plus the pinned plan.
- Review training payload reads, remote dispatches, scoring calls, model loads
  and optimizer updates: **all zero**.

The author's real one-ledger receipt was produced before the final rate-limit
and bookkeeping changes, at analyzer hash `77d1bfab…`. It remains earlier
mathematical/capacity evidence, not final-source full throughput. This review
does not fabricate a new real benchmark. Actual full-run resource consumption
and completeness must be reported from the dispatched run. Scientific
interpretation remains separate from this execution review.
