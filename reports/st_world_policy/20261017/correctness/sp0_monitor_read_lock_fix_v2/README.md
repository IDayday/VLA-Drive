# Independent bounded retest of final counter wall enforcement

**PASS for the reviewed monitor repair**, including closure of F1 from the
independent v1 report. No retry or production dispatch is approved here.

- Tested fix: `bdd750a4489f4a5019cd4d33d3bde9571da344d2`.
- Fixed detached runtime:
  `/mnt/project/VLA-Drive-st-monitor-lock-review-runtime-bdd750a4`, clean before
  and after the test.
- Helper: `reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1/supervise_once.py`.
- Actual helper SHA256:
  `607fa859a6b814accba2b40cf3568d9611f96d940f89f7dbdce2dfcd7bc61b30`.
- Original failure evidence remains at commit
  `bcc4718146bc960d5ef99ba6e74da8331e9b22aa`, in
  `correctness/sp0_monitor_read_lock_fix_v1`; neither its tests nor logs changed.

The narrow fix computes elapsed time after the final locked counter read,
applies the original wall threshold while preserving an earlier failure reason,
and returns that same elapsed value. It changes no stage budget, retry policy,
ownership/signalling logic, mathematical code, storage policy or error truncation.

The independent test with an actual fixture writer holding the shared lock
after owned-child cleanup now returns `WALL_LIMIT` instead of admitting success.
All nine implementer counter-lock tests and all three independent cases passed:
**12 passed in 4.12 seconds**, no failures, skips or warnings. This includes:

- Writer blocked during strict inode/read/freshness observation.
- Original two-second lock timeout and fail-closed behavior.
- Malformed/oversized/symlink snapshot rejection.
- Bounded first RSS/read/freshness error, not overwritten by final failure.
- Final read using the same locked helper.
- Final valid `IO_LIMIT` retained.
- Final successful lock wait crossing wall now rejected.
- Missing counter and immutable artifact hash semantics unchanged.

Only temporary small metadata and freshly owned CPU child processes/threads
were used. CPU affinity was 0,1; CUDA visibility was empty; the test had an
external 90-second timeout and two-thread library limits. No real failed stage,
runtime artifact, GPU, image, tensor, checkpoint, ledger or production process
was read/changed/signalled. This is a finite code regression, not actual
extraction evidence or a resource qualification.

Reproduction from the fixed detached runtime:

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  timeout --kill-after=5s 90s taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python -m pytest -vv -p no:cacheprovider \
  reports/st_world_policy/20261017/sp0_monitor_read_lock_fix_v1/test_counter_lock.py \
  /mnt/project/VLA-Drive-st-monitor-lock-correctness-20261017/reports/st_world_policy/20261017/correctness/sp0_monitor_read_lock_fix_v2/test_independent.py
```

The unique cause of the original real `MONITOR_UNREADABLE` failure remains
**UNKNOWN_NOT_RECORDED**. Reproducing and fixing the atomic-replace race does not
establish that it caused that old failure. The original attempt remains spent,
its RESULT remains immutable, and no automatic retry, re-admission or frozen
runtime update follows from this review. Any distinct attempt remains Lead's
decision under the unchanged source/storage/resource gates.
