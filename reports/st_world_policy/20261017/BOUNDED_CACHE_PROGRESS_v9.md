# Small-disk execution update — 2026-10-17 08:02 UTC

This supersedes the live-status portions of v8; v8 and all original receipts
remain available. The user's storage limit remains the decision boundary:
reuse the existing FP32 current-teacher bank through small indexes, compute
missing future targets with the same frozen FP32 function, and bound the RAM
LRU. No new full-population future tensor bank is being built. Student official
BF16 execution, FP32 master/Adam, losses, data and schedules are unchanged.

## Actual student result

The registered two-update diagnostic finished with exit 0. Its full evidence is
in `indexed_student_smoke_actual_v1/`, integrated at `abd51fd0`.

- Exactly 2 optimizer updates and 4 sample presentations, world1/global2.
- Current/future/trajectory losses and gradient norms were finite. Sampled
  FP32-master elements changed on both updates (2694 and 2679 elements).
- Provider: 48 immutable hits, 12 computed images, 4 fixed-eight teacher calls,
  no reader failure, RAM LRU peak 14,159,232 bytes, **0 new teacher tensor disk
  bytes**. The metadata reader did not materialize the full population.
- The two native checkpoint tensor payloads total 74,731,854,932 bytes. These
  are full model/optimizer/recovery states, accounted separately from caches.
- Allocation wall 529.271105 s includes cold loading and two checkpoint writes
  (76.910 s and 72.888 s). Two updates do **not** establish sustained throughput.
- This remains disposable SMOKE evidence; its weights cannot initialize a
  formal run. No new multi-rank/exact-resume claim is made from this run.

The next implementation is an explicit, bounded PROFILE path using the same
trainer and fixed teacher/PCA. Its new configuration and source-compatibility
checks require affected-scope independent review. It does not remove old
SMOKE limits or open FORMAL admission by changing a constant.

## Storage ledger

The actual additive migration completed under the existing shared claim lock:
`10df383020d32441a7c3cfe617c89e3b8a651f2af720b7dd8fe8918cee221b34`.
All seven old claim rows were preserved. Only one new dev16 frozen-W diagnostic
member was declared, with 26,085,776 bytes for output, staging and metadata.
The 768 GiB cache envelope includes existing caches; it is not new allocation.
Effective metadata reservation remains 1264 MiB / 4096 MiB.

The fresh separate checkpoint commitment is 2103 GiB (1983 GiB registered-run
budgets plus 120 GiB diagnostic allowance), conservatively without deducting
existing files. Another 1 TiB free-space floor is retained. The capacity receipt
expires at 08:44:09 UTC and must not be reused after expiry. This is allocation
accounting, not a verified storage quota. No cache or checkpoint was deleted.

## Failed bounded W extraction is retained

The independently admitted SP0 dev extraction was attempted once at 07:56 UTC
on GPU5. It stopped before any model/native load, with `MONITOR_UNREADABLE`
and exit -15 after 35.150432 s. Observed RSS peak was 711,897,088 bytes;
native loads 0, teacher/W tensor writes 0, remaining owned processes 0.

Private evidence:
`audits/spatial_probe_actual_SP0_5000_v1/stages/dev_extract/RESULT.json` under
the campaign root. The original dev plan and failed extraction evidence remain
immutable. There is no automatic retry. The resource monitor is under a narrow
failure investigation; no mathematical/numerical tolerance is being relaxed.
This is not a cache-computation or student-update failure.

## Existing work continues independently

SP1_5000 finished exporting all 1696 fixed-dev trajectories. Official scoring
started with the existing two-worker/1728-call bound (PID452984); no result is
claimed yet. C3_512, SP2_5000 and C1_1024 exports remain active. No Navtest,
held IQE expert/Scorer/MOPD job, new full Base, or ACTIVE update was launched.

Current work uses isolated commits and namespaces. No active training source
was hot-edited. Nothing was pushed.
