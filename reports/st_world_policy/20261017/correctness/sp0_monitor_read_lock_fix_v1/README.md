# Independent narrow supervisor counter-lock review

**FAIL: final successful counter-read lock wait is measured but not enforced
against the wall limit.**

Tested integrated source `6e32bf9363d7c8c10fca437c69304671529290f1`.
The only reviewed helper is
`reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1/supervise_once.py`,
SHA256 `e70909ef323cc6ebfe8e4430212d47ce30be90194c52216c5ac20f22c6f5372b`.
No production implementation was modified by this reviewer.

## Actual checks

- Implementer's nine new counter-lock tests independently rerun: **9 passed in
  2.69 seconds**, complete output in `OWNER_TESTS.log`.
- Three additional reviewer cases: **1 failed, 2 passed in 0.91 seconds**,
  complete output in `INDEPENDENT.log`.
- CPU affinity 0,1; two library threads; empty CUDA visibility; external
  90-second test timeouts. Only tiny temporary JSON, fixture locks/threads and
  fresh owned CPU child processes were used. No old process was signalled.
- No GPU, images, weights, tensor cache, real stage, runtime mutation, ledger,
  retry, dispatch or production-result edit occurred. The implementation's
  historical 59-test result is not claimed as this review's run.

## What passed

The actual writer and reader share `prepare_bindings.control_lock` across
existence check, the unchanged strict read, and freshness stat. The controlled
atomic-replacement race confirms the writer cannot publish while the reader is
inside its strict fstat/freshness sequence. The two-second lock timeout remains
fail-closed. Invalid JSON, oversize content and symlink cases remain rejected.

The first observed error is bounded and retained. The independent additional
test injects a freshness error followed by a final lock timeout and verifies the
first error survives. A valid final counter reporting `IO_LIMIT` remains a
failure, rather than being lost. Original fixed-artifact hashes and monitoring
threshold constants are preserved. Periodic successful lock waiting is included
in the refreshed wall check.

## F1: final read can cross the wall limit and still succeed

Location: `_supervise`, final `read_counter_snapshot` block around lines
470–480. The periodic branch refreshes `now` and checks the wall limit; the final
branch does neither before returning `limit_reason`.

Minimal reproduction:

1. Create a valid tiny `COUNTERS.json` using the original writer.
2. Supervise a newly spawned CPU child that sleeps 0.02 seconds and exits 0,
   with a 0.5-second fixture wall budget.
3. Immediately after actual owned-session cleanup completes, have a fixture
   writer thread hold the **real shared control lock** for 0.65 seconds.
4. The final counter read waits, succeeds and returns the valid counter.

Observed result: child exit 0, no owned descendants, `wall_seconds > 0.5`, but
`limit_reason is None` and `first_monitor_error is None`. The independent test
expects `WALL_LIMIT` and fails. The real main's success predicate uses exit code
and absence of limit reason; it has no independent final wall check. Thus the
new wait can be admitted as success even when it crosses the invocation limit.

The short fixture budget only exposes the boundary quickly; the same omitted
check exists for an actual stage ending near its fixed 600/960/1260-second
limit. This is **not** evidence that the old actual extraction crossed its wall
limit, and no actual production counter was altered or re-read for this test.

Required narrow correction: after the final snapshot, refresh elapsed wall time
and enforce the original wall threshold before returning successful status,
preserving any earlier reason/error. Do not increase the threshold or add retry.
The actual child is already cleaned up, so this correction needs no new signal.
Then rerun this failure and the neighboring final-error/first-error tests.

## Original actual failure remains unattributed

The earlier `dev_extract` returned `MONITOR_UNREADABLE` before model/tensor
activity and did not record the underlying exception. That unique failure's
cause remains **UNKNOWN_NOT_RECORDED**. The reproduced atomic-replacement race
establishes a real defect mechanism, not a retrospective diagnosis. Its spent
attempt and original failure artifact remain unchanged.

This review does not approve retry, a new admission, runtime freezing or learned
W extraction. Lead must close F1 and independently satisfy existing execution
and storage/resource conditions before deciding on any separate new attempt.

## Reproduce CPU evidence

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  timeout --kill-after=5s 90s taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python -m pytest -vv -p no:cacheprovider \
  reports/st_world_policy/20261017/sp0_monitor_read_lock_fix_v1/test_counter_lock.py \
  reports/st_world_policy/20261017/correctness/sp0_monitor_read_lock_fix_v1/test_independent.py
```

The evidence was collected as two separate runs; the combined command above is
the equivalent reproduction, not a claimed third execution.
