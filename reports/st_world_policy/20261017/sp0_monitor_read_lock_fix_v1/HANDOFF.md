# Bounded mutable-counter monitor repair

Base: `eea55d43`. The only production/helper change is
`spatial_probe_actual_preparation_v1/supervise_once.py`.
No model, mathematical recipe, reader/writer implementation, admission policy,
ledger, real stage, runtime checkout, or failed result was changed.

`read_counter_snapshot` holds the original
`prepare_bindings.control_lock(counter.parent)` across existence check, strict
64 KiB read, and freshness stat. Both periodic and final counter reads use this
helper. Fixed artifacts still use the original strict reader unchanged.
Atomic publication by the cooperating counter writer therefore cannot switch
the pathname between the strict reader's inode checks and freshness lookup.
Uncoordinated changes, invalid JSON, wrong hashes, oversize/symlink data and
lock timeout still fail closed. No retry loop was added.

The first monitor exception is retained in `resources.first_monitor_error`:
operation (at most 64 characters), exception class (128), message (1024), Unix
time and elapsed seconds. Operations distinguish RSS, counter lock/existence,
strict read, freshness and status processing. Subsequent/final-counter failures
do not replace that first error. A malformed final snapshot returns
`MONITOR_UNREADABLE` with the error record rather than dropping the diagnostic.
The existing result writer bounds the complete serialized result.

The shared control lock retains its **two-second timeout**. A counter sample
can add up to two seconds of lock waiting, plus filesystem I/O and scheduling;
the output monitoring description now states this. Wall checks refresh their
clock after sampling, so lock waiting counts against the existing deadline.
No RSS, GPU, IO, wall, log, stage-storage, freshness, or cleanup-grace threshold
was increased. This is an observed monitor, not a hard cgroup deadline.

## CPU tests

**59 passed in 6.17 seconds**, including all 50 existing controller/lifecycle
checks and nine new cases:

- writer reaches the shared lock during strict fstat checks, remains blocked
  through the read and freshness capture, and publishes after release;
- real original lock timeout remains approximately two seconds and fails closed;
- malformed, oversize and symlink counter rejection;
- actual JSON exception operation/class/time and owned-child cleanup;
- first RSS exception is bounded and is not overwritten by a later final error;
- final read goes through the same locked helper and preserves its failure;
- missing-counter and fixed-artifact hash semantics, original constants.

Tests ran with CPU affinity `{0,1}`, RLIMIT_AS 2 GiB, empty CUDA visibility,
two-thread environment limits and a 90-second suite timeout. Only tiny fixture
JSON and newly spawned CPU sleep processes were used. The prior actual failed
stage was neither read for a rerun nor modified. No GPU, images, weights, tensor
cache, real fit/plan/extract, ledger or production-task signal was used.

```bash
/root/miniconda3/envs/ddp/bin/python -m pytest -q \
  reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1/test_preparation.py \
  reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1/test_lifetime_fix.py \
  reports/st_world_policy/20261017/sp0_monitor_read_lock_fix_v1/test_counter_lock.py
```

## Remaining execution boundary

The original 35-second failure's unique cause remains
**UNKNOWN_NOT_RECORDED**. This change repairs the independently reproduced
atomic-counter/strict-reader race and adds evidence for any later failure; it
does not retroactively establish that race as the actual cause.

The old `dev_extract` attempt remains spent. No automatic retry, deletion,
resume, re-admission or next stage occurs. Lead alone may prepare a distinct
explicit attempt with new helper/runtime pins and independent review. Existing
storage/resource conditions remain applicable. This implementation report
does not grant actual dispatch or claim successful learned-W extraction.
