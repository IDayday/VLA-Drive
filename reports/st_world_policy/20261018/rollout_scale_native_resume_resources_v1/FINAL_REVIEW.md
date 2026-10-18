# Independent Systems review: native resume SMOKE

**PASS for the exact source and actual metadata binding below.** This approves
only the finite native continuous4 versus pause2+resume2 SMOKE resource contract.
It does not assert replay correctness, approve a new formal run, start anything,
or permit diagnostic weights to initialize formal training. Lead alone dispatches
after the separate Correctness review and fresh admission.

Controller commit: `19a5a244cca2a07600339ba9be7c3a150ef61285`, clean.
Controller SHA256: `693d9307033efeca90da059ce4b6a75a35537f99ddf90dac092f96e356cf618e`.
Frozen trainer remains `751fcfa2b41549ae16f7b3204aef8dd18b912016`.
Binding file SHA256: `cf901802b08c5289b26fe366f19d471caee803ca9fd9ff844b3cc31f470b4661`.
Binding identity: `0a6e0eb53db85a946ccae9c9bed86ccce89eb90f5e466c90aa700df2fad7177c`.

## Scope and layout

The actual binding pins the previously observed eight zt6 GPU UUIDs, matching
leases, current policy/authorization and exact formal software environment.
World4 A uses GPU0–3/CPU0–15; world4 B uses GPU4–7/CPU16–31. Each group keeps
micro4, accumulation2 and effective global32. This is two independent arms,
not an altered global64 training recipe. Four sequential BLAS1 feedback workers
per arm give at most eight globally. The exact32 fixed TRAIN manifest is pinned.

Only A4, B2 and one B resume2 are allowed: eight total optimizer updates,
512 primary official calls per arm,1024 globally, zero retry. Durable admissions
remain charged after partial/failed launch. There is no automatic retry.
The new external controller does not modify runtime751 or any active formal job.

Heavy reconstruction is staggered by whole world4 group: B waits for actual
A initialization/frozen record, all four engine receipts and native0 COMPLETE
bound to the current PID/start/session. B resume requires the initial phase
finished, all owned processes exited, exact native2, and an owned exact STOP
marker/acknowledgement. It cannot blindly acknowledge a later operator STOP.
No two-rank loading claim is made for source751's collective-first initialization.

Fresh RAM minimum640GiB; group RSS256GiB and aggregate512GiB; GPU driver48GiB/card.
These are fresh admission plus sampled guards, not memory reservations or hard
allocation quotas. Fresh RAM, actual UUID availability, policy, software, mount,
free space and lease remaining time still need to pass actual launch checks.

## Storage and IO

The104GiB new-output allowance covers at most eight native directories including
partials, each12GiB, plus8GiB for all feedback/control/hash metadata/temp files.
Normal six0/2/4 payloads occupy51.90GiB. Launch requires1634GiB shared free;
runtime protects1530GiB, including the1TiB floor,292GiB SourceFlow future writes,
209GiB active ST future writes and the previously accepted smaller commitments.
This is charged once on the shared filesystem, without multiplying by host or
double-counting already occupied checkpoints. No deletion is assumed.

I independently stat-checked the bound original S0's17 payloads:37,597,351,782B.
Twelve rank reconstruction passes plus three rank0 byte-audit passes nominally
read563,960,276,730B (525.229GiB), before framework and restore reads. Hence the
earlier512GiB draft is replaced, with Lead consent, by a cumulative1TiB sampled
rchar guard. These input reads are not new output space. rchar/read_bytes are
separate, retained over samples and PID reuse, and accumulated across phases.
Mmap and short-lived exited children can escape sampling: these remain observed
lower bounds, not exact physical IO accounting or bandwidth guarantees.
Output checks run every10s and GPU queries every5s; these are sampled bounds.

Full live module/master/Adam/RNG digest hooks copy/hash data and synchronize.
Their per-rank labelled audit_seconds are included separately; this diagnostic
must never be reported as ordinary training throughput. Hooks emit digests and
bounded metadata rather than duplicate GB tensor dumps. CPU comparison reads
bounded observer/rollout artifacts, not GB native payloads.

The full envelope is3600s trainer/controller +120s separate comparison
(2CPU threads,2GiB RSS,64MiB inputs,16MiB output) +180s owned cleanup =3900s.
The comparison command needs its explicit outer timeout; the trainer's3600s
clock alone does not cover this separate CPU stage. No extra training is added.

## Findings and affected verification

Two resource findings from the dirty draft were resolved in the frozen source:

1. Unconfirmed cleanup previously could coexist with a success status. The final
   controller treats any cleanup_complete=false as failure.
2. Sampled IO counters could be overwritten/reset. The final source preserves
   counters, retires PID-reuse values, and inherits the initial phase on resume.

The mature helper SHA `a2e24537b9749258417e390be705d4d5d4e6f775986ca479a1c67c4330759a3e`
retains the independently verified deferred SIGTERM/INT ownership mechanism.
I ran four narrow final-source controller cases with real short-lived CPU
children: SIGTERM before Popen returns, SIGINT in that same window, LAUNCH
receipt-write failure, and explicit negative cleanup confirmation. All passed;
five spawned children exited, all admissions were charged, no child remained,
and negative cleanup returned PARTIAL_OR_FAILED. No GPU/model/scoring command
was executed. Production route/score completion and receipt IO were explicit
fixtures; this verifies lifecycle behavior, not actual model resume.

Final harness:0.158s wall; see CPU_LIFECYCLE_RESULTS for actual CPU and memory.
Post-exec current RSS was23,805,952B; child maximum25,911,296B. getrusage reports
an inherited pre-exec high-water2,095,808,512B, so that value must not be treated
as this tiny harness's synchronized aggregate resident footprint.

The first harness attempt passed the same four behavior cases but failed its
own no-Torch-import assertion: mature immutable lazily imported base_binding.
Its original result is retained. Mocking only receipt persistence removed that
unneeded dependency; no production code was changed. A first metadata-check
assertion likewise assumed numeric CUDA indices; sealed source and real leases
both consistently use exact UUID strings. The corrected binding check passed.
Neither harness correction masks a production failure.

The implementer's38 CPU tests are separately attributed inherited evidence;
I did not rerun that suite or old endpoint tests. Real native GPU replay remains
unexecuted by this reviewer and must be reported separately from this approval.

## Review cost and remaining boundary

Three original metadata collectors each read170,039B; preliminary source/control
inspection was conservatively budgeted2MiB. Additional bounded source, binding,
receipt and plan inspection remains below the8MiB selected-control allowance.
Each final binding check read201,140B of small controls. No image, metric cache,
model/native payload or rollout payload was read or hashed by this reviewer.
Python/software dependency loading in the first harness is separately disclosed;
it is not model/input payload inspection. Total review CPU is far below60s.

The actual binding is metadata-only and was checked before preparation. It is
not a resource reservation. Lead must not dispatch if actual fresh guards fail,
lease expires, source/pins change or an unrelated job is using the GPUs. Protect
the existing zt5 formal job and all other genuine jobs. This resource PASS does
not replace the independent Correctness review or claim real exact resume.
