Independent control/resource review of 7d698b6d
============================================

Status: **PASS_CONTROL_ADAPTER_AND_RESOURCE_PLAN_ONLY**. Reviewer is
`/root/st_owned_descendants_review_v2`, independent of the finalizer/controller
implementer. This is not an observed full-scale completion or scientific PASS.

The isolated review branch fixes author commit
`7d698b6d6f04661833f03f73ce4f0e92f23820cb`. Actual writer runtime remains
`/mnt/project/VLA-Drive-st-future-finalize-runtime-v1-20261017` at
`3df052ea422c4ba042ef3be33dea19438c5702dc`. No live code was changed.

Twenty independently written control tests passed, not inherited author counts.
Run: `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' taskset -c 8,9 python3
reports/st_world_policy/20261017/future_finalization_control_independent_v1/run_bounded.py`.
The invocation took 0.521359 s; 25 samples observed aggregate RSS38,969,344B,
child HWM28,418,048B, and post-wait parent logical read3,589,093B. The latter
includes the child once and is not added to the child again. No terminal RSS
sample was missing. Limits were120s/512MiB/32MiB read/8MiB fixture output.
Tests created only tiny control fixtures in this isolated report directory,
removed them after use, and retained small logs. Source/control inspection was
limited to small Python/JSON files; no shard/raw/image/teacher payload, live
ledger, real task signal, GPU or dispatch was used.

The actual finalizer AST uses `FINALIZE.lock`, now allowed by the fixed draft.
The author’s pre-fix failure is preserved in the proposal directory. The exact
five monitor/controller pins and nine writer dependency pins match. Draft
admission, implementation-author review, expired/nonfinite lease, changed
runtime/limits, wrong hash, symlink controls, wrong final artifact and terminal
read overrun all fail closed in our tests. Immutable failed results cannot be
overwritten; both fixture control-lock and actual ledger-flock wrapper are
nonblocking. A monitor exception retains UNKNOWN cleanup, and a busy ledger
before spawn yields one attempt and NOT_STARTED, without an automatic retry.

All11 old policy claims and legacy records are unchanged. The new claim is
2GiB with1008MiB output+1008MiB staging+32MiB metadata. Lead separately adopted
1MiB for terminal controls. Its inclusive plan is7701MiB within8192MiB, with
the original admin and monitor reservations retained. The existing2303GiB
checkpoint obligation and1TiB free floor remain. Lead’s completed same-ledger
migration receipt is pinned in REVIEW.json; this reviewer did not mutate or
read the live ledger. A separate actual admission must bind correctness and
resource reviews plus this completed migration before dispatch.

The actual resource envelope is CPU0/1,3600s work+40s cleanup,16GiB total
logical read,4GiB aggregate RSS,2GiB artifact and1MiB separate controls. The
entry clock/read baseline precede helper imports and admission reads. The
real wait-boundary arithmetic was exercised directly: baseline100, pre-wait120,
post-wait1125 and child1000 charge1025; later parent1130 charges1030.
The obsolete double-charge result is not rewritten.

Current core monitoring uses **OwnedSession**, not a new cross-SID torchrun
monitor: fresh unreaped Popen anchor, PID/start/SID verification and pidfds,
bounded TERM/KILL cleanup and no historical PID authority. That scope fits this
one-reader CPU finalizer. This round does not requalify multi-rank/cross-SID
jobs. Existing helper code is source-reviewed and pinned; our controller
fixtures do not signal real tasks or claim new live process-tree evidence.

Remaining limits: full1176-shard memory, compression, output capacity and
completion time are not measured by these tests. RSS limits are sampled with
writer prechecks, not hard cgroup guarantees. Actual regular pinned control
files and frozen runtime are required; this is not a hostile-filesystem general
security certification. `FINALIZATION_COMPLETE.json` alone cannot turn the
wrapper result into success: separate terminal RESULT must establish final
accounting and zero owned survivors. Old source failure stays
FAILED_NO_AUTO_RETRY. No GPU, model training, teacher extraction, automatic
retry or scientific coefficient acceptance is authorized.
