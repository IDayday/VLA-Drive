# Small-disk execution update — 2026-10-17 08:51 UTC

This supersedes the live-status portions of v9, without changing old receipts.
The active ST world/consequence campaign uses the user's small-disk constraint:
reuse existing frozen current-teacher tensors, index their metadata, compute
missing future targets with the same FP32 teacher, and keep a bounded RAM LRU.
No new full-population teacher tensor bank is authorized by this decision.
This is the ST DINO target path; it is not a new Query Base visual-cache run.

## Actual completed student execution

The 64-update diagnostic completed with exit 0 and paused at its registered
budget. Evidence: `indexed_profile64_actual_v1/ACTUAL_RESULTS.md` and its pinned
`RESULT.json`, integrated at `3d31843e`. Actual runtime source was
`2d5e9483a8c8bc80b71348211756f3cab6aa21b1`.

- One A800, global batch 2, 32 fixed real scenes, each presented four times.
- All 64 updates had finite losses/gradient norms and sampled FP32-master
  changes. Current/future/ego-flow objectives and original 20k learning-rate
  timeline were retained.
- Provider failures: 0. New teacher tensor disk bytes: **0**. RAM tensor LRU
  peak: **14,159,232 bytes (13.5 MiB)**. Metadata reader budget: 32 MiB.
- Complete update wall, all 64 intervals: mean **2.9380 s**, p50 **2.7192 s**,
  p95 **4.5229 s**. Excluding the explicitly reported first eight updates:
  mean **2.7476 s**, p50 **2.6777 s**, p95 **3.7735 s** (56 intervals).
- Rank RSS peak 7.9768 GiB; first/last eight-update median RSS was unchanged.
  The per-step CUDA allocated/reserved peaks were 52.4187/56.0020 GiB; their
  reset excludes preceding data/teacher preparation, so these are not whole
  invocation GPU peaks.
- Exactly one full native checkpoint, 37,365,928,667 bytes, was saved in
  73.1697 s. This recovery state is separate from the teacher cache. At this
  diagnostic's save-every-64 cadence, update plus save averages 4.0813 s/update,
  excluding the 455.6922 s cold startup.

These are real short-window results, not an eight-GPU/global32 speedup claim,
not full-population supply evidence, and not formal scientific results. The
diagnostic weights remain forbidden as formal initialization. The recorded
training session had no remaining processes at the final read-only inspection.

## Disk decision and remaining dependency

The existing current bank (539,427,538,104 bytes) is reused, not regenerated.
The existing full-current metadata index is about 364 MiB. Actual32 future
preparation plus index occupy 879,160 bytes. Linear scaling gives approximately
2.643 GiB for full train/dev future metadata and indexes; this is an estimate,
not completed coverage or a reservation. SQLite journals, partial files and
controls must be counted in the peak disk allowance as well.

The next preparation is metadata-only for the existing train101592/dev1696
population. An explicit combined few-GiB allowance is under independent review;
the old 4 GiB registration is not silently enlarged. No future teacher tensor
cache, precision change, data reduction or checkpoint deletion is involved.
Full metadata coverage and the intended topology/resume qualification remain
uncompleted; the 32-scene test does not stand in for them.

## Small W diagnostic cache: artifacts complete, supervisor failed cleanup

The independently admitted SP0 dev retry computed all 16 W records. The
original runtime validator independently verified the full payload hashes,
FP32 shapes, masks, source identity and completeness. Payload bytes total
18,879,232. Evidence: `correctness/sp0_dev_artifact_reconciliation_v1/`, integrated
at `0fcc5a79`.

The original supervisor RESULT remains `STOPPED_NO_AUTO_RETRY`, with the sole
reason `OWNED_DESCENDANTS_AFTER_LEADER_EXIT`; it reports exit 0 from extraction,
eight owned descendants cleaned up, and zero remaining. Do not replace it with
a fabricated successful launch. A narrow additive reconciliation consumer is
being implemented so valid committed artifacts need not trigger another large
model reload. No extraction rerun is scheduled.

## Independent experiments

Fixed full-dev scoring has completed for C3@512, C1@1024 and SP1/SP2@5000.
All use 1696 scenes / 16 logs, one original Euler-10 trajectory, and the same
official reference protocol. Paired analyses are in the evaluation reports.
C4@512 and SP3@5000 exports continue. No new Navtest, held IQE/MOPD run, new
Base, formal ACTIVE change or push was performed for this update.

The current teacher stays FP32. Student execution follows the already verified
official BF16 execution/FP32 master and Adam-state path; this storage decision
does not introduce a new precision experiment or remove any supervision.
