# Native resume SMOKE: independent resource plan

Status: **PLAN_READY_WAITING_FROZEN_CONTROLLER_AND_LEAD_ADMISSION**.
No training, model, GPU, image/cache or native payload was loaded by this review.
This is only the continuous4 versus pause2+resume2 diagnostic. It does not approve
new formal training, change runtime751 or authorize use of SMOKE weights later.

At2026-10-18 19:50:21UTC, a fresh read-only SSH probe of the authorized zt6 alias
found all8A80080GB at0MiB/0%, no GPU process, 840.87GiB MemAvailable,128 CPUs and
the same project Lustre source with global flock. Free space was
3,466,060,107,776B (3228.12GiB). This is not a reservation or quota guarantee;
Lead must repeat fresh admission. FRESH_ZT6_RESOURCES records the exact8 UUIDs.

Use groups A=GPU0–3/CPU0–15 and B=GPU4–7/CPU16–31. Each is world4, micro4,
accum2, effectiveglobal32, Torch2/rank and BLAS1. Each group owns four sequential
single-thread scorers, at most eight globally. Same registered32TRAIN scenes,
original4096 scheduler, G4/MC50/Euler10/repeat8/standardized-advantage semantics
remain unchanged. Only A4 and B2+resume2 are admitted: eight total updates.

Final call budget is **512 per arm,1024 total, retry_attempts=0**. A failed
score/launch remains failed evidence; no automatic completion retry or additional
attempt is authorized. The earlier1056-call suggestion is retained explicitly
as a superseded unexecuted draft. A new phase/controller cannot reset B's ledger.

## Memory, loading and shared IO

The actual matching4rank PROFILE measured118.10GiB summed process RSS and
17.68GiB GPU driver memory/card. Two observed group peaks sum236.20GiB; this
is a planning basis, not an8GPU measurement or guarantee. Adopt256GiB/group,
512GiB aggregate RSS including known ranks/scorers/observers,48GiB driver/card
and **640GiB fresh MemAvailable** so the aggregate ceiling leaves128GiB margin.
The fresh840.87GiB observation supports this layout.

Do not launch both heavy reconstructions together. Start B only after A's actual
four initial-engine receipts, initialization/frozen-state record and complete
native0 agree with the current invocation and source. B's native resume similarly
waits for its pause2 subtree to exit and for no other group reconstruction.
Compute may overlap a second group's loading; only one world4 group reconstructs
at once. Runtime751 calls NCCL init before every rank's original load_student;
it cannot honestly promise a two-*rank* loader limit through an external wrapper.
The independent-exporter's2-loader rule is not transferred to training.

This bounds loading concurrency, not physical bytes/s. No filesystem bandwidth
measurement exists. Measure own load/read/checksum/save/hashhook wall and actual
child CPU; do not stop the protected zt5 formal job or SourceFlow job. If real
contention appears, only this diagnostic may wait/stop at its recoverable bound.
Do not claim the diagnostic is a throughput benchmark: read-only full FP32
master/Adam/RNG hashhook cost must be included in the resource receipt and
reported separately from ordinary update wall. Hooks may emit only digests and
bounded metadata, not another full model/optimizer tensor dump.

## Single shared-filesystem write budget

I read six small COMPLETE receipts and stat-checked every named native shard.
Original SMOKE0, PROFILE0 and current formal0 all have4,917,269,512B payload;
mature SMOKE2/4 and PROFILE240 each have11,473,306,504B. No payload checksum or
deserialization was performed here; the real run still uses strict native checks.

Each new arm ordinarily stores0/2/4, so six native payloads total
55,727,765,040B (51.90GiB). Two additional simultaneous mature atomic partials
would add22,946,613,008B (21.37GiB). Use **104GiB total new output**, including
at most8 native directories counting partials at12GiB each plus8GiB for all
feedback, controller, digest hooks, observer logs and temporary writes. This is
an execution cap/allowance, not a prediction that104GiB will be written.

Launch free-space threshold is a single **1634GiB**:

|Future protected term|GiB|
|---|---:|
|Shared minimum free floor|1024|
|Protected SourceFlow remaining-write upper bound|292|
|Active ST formal remaining-write upper bound|209|
|Previously unresolved metadata headroom|4|
|Already sealed endpoint evaluation allowance|1|
|This diagnostic's complete new output allowance|104|

The observed headroom after those bounds is1,711,565,967,360B (1594.00GiB).
The diagnostic's live monitor preserves1530GiB for floor+other future writes.
These inherited remaining-write terms are charged once as conservative upper
bounds; occupied files are already excluded by df.free and are not separately
added. Do not add old SMOKE/PROFILE, completed/paused feature caches or finished
C0–C4 future budgets again. Do not multiply capacity or reservations by host/GPU.
No deletion/retention change, checkpoint copy or assumed disk reclamation occurs.

## Pending source-bound review

RESOURCE_LAYOUT and SHARED_STORAGE_PLAN are the concrete resource contract,
not a launch approval. The new external controller and read-only hashhook must
be frozen before I review only their lifecycle delta: bounded launch attempts,
deferred SIGTERM/INT through Popen ownership transfer, full owned-session RSS
and exit, correct group reconstruction gate, persistent1024-call accounting,
shared output bounds and explicit native2 resume. The formal job continues.

The collector read170,039B of small controls per execution; it was run three
times while tightening the draft. Prior source/receipt inspection is bounded
conservatively at2MiB; no large native/model/image/context payload was read.
That keeps this preparation below8MiB control inspection and60CPU seconds,
with room for the narrow final controller review. Raw private process inventories
are not committed. This plan neither claims the real resume succeeded nor
replaces an independent correctness review.
