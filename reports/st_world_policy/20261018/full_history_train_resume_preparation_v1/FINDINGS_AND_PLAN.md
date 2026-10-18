# Actual train history failure and one bounded continuation

This is preparation, not independent approval or an executed resume. Runtime
`/mnt/project/VLA-Drive-st-full-history-v3-runtime-20261018` remains
`eae8922d72b3881c0c162e96b8dd67e7c15a41c7`; its original builder, writer,
source pins, input population, storage registration and outputs are unchanged.

## What actually failed

`train_controls_attempt2/RESULT.json` says `FAILED_NO_AUTO_RETRY`, phase
`RESULT_PUBLICATION`, `StorageBudgetError`, and “remaining reservations plus
separate free-space reserve do not fit”. It records zero live owned children.
Its512GiB read charge is conservative on exception, **not a measurement**.

The original `run_pilot_once.py::run_parent` had already received a monitor
return when it entered this phase. It then reacquired the native storage guard
to publish `CONTROL_HISTORY_RESULT.json`. The exception handler retained only
cleanup and discarded the monitor return code, failure, child error and read
accounting. Therefore it is not possible to recover the child's actual first
failure from this receipt. “Result publication failed” does not mean that the
builder had successfully completed. A child-side shared-space guard failure is
plausible because its `MetadataWriteContext.check` uses the same checkpoint
reserve at every bounded read/write. A source/time budget failure cannot be
ruled out from the surviving artifact. We do not invent a terminal read count.

The original builder commits `HISTORY_MANIFEST`, `TEACHER_REQUESTS`, `COMPLETE`
only after visiting all selected scenes. None exists here. The final committed
shard timestamp precedes the nominal4h invocation deadline; this is consistent
with an early resource guard failure, but not independent proof of the cause.

## Actual committed work and IO estimate

Two separately bounded read-only metadata scans used CPU12/13,32MiB/s pacing,
120s caps each. Neither opened source images, raw logs, model payloads or GT.
They read2,301,241,770B total in68.6464s. Maximum recorded RSS was271,196,160B.

| Quantity | Actual metadata evidence |
|---|---:|
| Contiguous committed shards |944|
| Covered scenes |78,657 /101,592|
| Covered logs |619|
| Remaining scenes |22,935|
| Committed shard bytes |1,150,620,885|
| Historical image references in committed records |471,603|
| Unique historical image paths |276,642|
| Original registered repeated historical-image read bound |196,034,310,460B|
| Strict one-SHA-per-unique-image bytes |57,450,849,394B|
| Avoided registered read allowance for this prefix |138,583,461,066B (129.07GiB)|

The original image reference charges one full-file header bound plus a full
SHA read on each occurrence. The proposed path needs no repeated image header
decode because an identical SHA proves the previously recorded dimensions.
It rehashes every unique original image at least once on this invocation.
At the unchanged32MiB/s pacing, the prefix reduction alone saves approximately
4,130s of registered pacing. This is a cost estimate, not observed builder
speed. Current t0 teacher images, raw logs, current/ego pins and22,935 uncovered
scenes remain unmeasured here and continue through the original implementation.
The unchanged448GiB inner and512GiB outer caps remain hard stop conditions;
completion within4h is not predeclared.

## Why plain native resume is insufficient

The native parser has `--resume`; config identity deliberately removes that
operational flag. `HistoryShards.flush` derives identical names and
`write_chunks` verifies existing content rather than replacing it. However,
`prepare_history.run` still rereads/rebuilds every scene before this check.
There is no stored scene cursor that would jump directly to shard944.

The historical typed registration and old controller also require `resume=False`.
Changing their JSON in place would break immutable claim/admission provenance.
The single proposal instead adds an explicit independently signed
`resume_overlay` in a new attempt control namespace. Its narrow parser bridge
first validates the exact original registration argv, then enables the existing
native resume flag. No input/scene/geometry/target option is changed. Original
registration/source validation, ledger, builder, finalization and monitor still
execute. This execution overlay is explicit and must be independently reviewed;
it is not presented as an unchanged historical invocation.

## Strict reuse, not a stat-only shortcut

Original committed image references contain path,SHA256,bytes,mtime_ns and
original dimensions. They do **not** contain original dev/inode/ctime. The
implementation never fabricates these old versions and does not trust only
size/mtime or a self-signed record.

The resume overlay:

1. Checks pinned shard hashes, complete canonical records, native schema/self
   identities, unique tokens, contiguous shard order and the exact original
   grouped scene prefix from the retained selection. Partial/orphan/extra or
   conflicting shards fail.
2. Retains the original fresh hashing of raw logs and current/ego records. For
   every reused scene, actual raw SHA and current-record identity must equal
   those recorded in the committed history; causal offsets/tolerance/log match.
3. Rehashes each distinct historical image against its old committed SHA on
   first use, with before/after dev/inode/size/mtime/ctime checks. It checks the
   same full version for subsequent repeated references. Same-path mutations,
   even with original size/mtime restored, fail. Missing/mismatched inputs do
   not fall back to stale records. Runtime HistoryReader still hashes image
   payloads at actual training reads.
4. Returns byte-identical old records to the original packed writer; its own
   immutable shard checksum comparisons remain active. Uncovered scenes call
   the original history builder, and current teacher references are unchanged.
5. Uses the original native manifests/COMPLETE; no completion marker is written
   by the bridge. Summary counts must be78,657 reused and22,935 new.

`resume_records.py` is scoped to this invocation, restored on exit. No model,
trainer or original runtime file is modified. Original sources remain byte
identical; `SOURCE_AND_FAILURE_PINS.json` binds the new execution overlay.

## Failure evidence and next attempt

`run_resume_once.py` loads the pinned original attempt2 adapter and owned
monitor. It preserves the original start marker instead of conflicting with
its old admission, records a new invocation in `train_controls_attempt3`, and
adds child/monitor/read-accounting fields to the separately reserved failure
receipt. Old failure receipts and their two512GiB conservative charges remain.
No original allocation is released and no completed shard is deleted.

One new invocation only: two CPU cores10/11, CUDA hidden,4GiB aggregate RSS,
14,400s+40s cleanup,448GiB inner source/512GiB outer logical reads,32MiB/s,
same2GiB combined artifact allowance, separate new1MiB control reservation.
The typed artifact claim is reused, not reserved twice. Resource owner must
bind a fresh reconciled checkpoint obligation receipt that fits for the entire
attempt, current free space, writer quiescence and the original nonblocking
global/ledger locks. No outdated3.701TB remaining number is hardcoded.

Author CPU tests:14 passed. They include same-size/restored-mtime corruption,
post-verification inode replacement, source/causal mismatch, shard checksum/
self-hash/prefix corruption, original uncovered-scene equivalence, scoped
parser and budget restoration, and preservation of first child failure evidence.
These are generated-file fixtures, not a new real-image resume or independent
PASS. Actual78,657-scene reuse/22,935-scene completion remains NOT_EXECUTED.
