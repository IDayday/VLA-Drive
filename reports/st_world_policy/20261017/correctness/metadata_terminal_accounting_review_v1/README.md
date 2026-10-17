# Independent review: terminal process-tree read accounting

2026-10-17. **PASS for the exact source change and CPU tests below.**

Source commit: `5b956b05df21e0b3cf194b5d0ac7f6218a596de3`.
Controller: `reports/st_world_policy/20261017/full_future_metadata_preparation_v1/run_build_once.py`.
SHA256: `a775a87532dab70a376cb64a7c0c19403e21e2f3a3b8040152fbefb8cebf10b9`.
Review tree: `/mnt/project/VLA-Drive-st-metadata-terminal-accounting-review-20261017`.

This review does not amend the original double-count evidence at `c14f0aa2`,
historical charges/results, current ledger budgets or the active old controller.
No live task was signalled, restarted, hot-patched or dispatched.

## Verified calculation

While the child remains unreaped, monitoring retains the original calculation:
parent counter minus invocation baseline, plus the owned child's absolute
counter. Process identity, session and pidfd ownership checks are unchanged.

Immediately before and after `owner.finish()` the controller records parent
counters. It requires exact nonnegative integers, a monotonic parent counter,
available terminal child evidence, and:

`post_reap_parent >= pre_reap_parent + terminal_child`.

The terminal charge is then `post_reap_parent - baseline`; it does not add the
already absorbed child again. Parent activity during the wait remains included
and is separately reported as `post - pre - child`. The final parent tail is
`current_parent - post_reap_parent`, added once by a pure calculation.

Missing/invalid baseline, before/after sample or child evidence, or inconsistent
absorption produces `INCOMPLETE_CONSERVATIVE`, charges the entire remaining
allowance and sets a monitor failure. A missing/nonmonotonic final parent sample
raises; the existing outer failure path publishes a failed result with the
remaining allowance charged, rather than substituting last-active progress.
This cannot become BUILD_COMPLETE. A successful reap still sets `finished`
before receipt processing, so later failures do not clean the same PID twice.

## Independent execution

**75 passed in 3.02s**: 61 focused existing cases independently rerun plus
14 new checks in `test_independent.py`. Full output is `TEST.log`.

- Two real local owned-child tests read a 4 MiB file per child; the parent reads
  its own 2 MiB file before reap and again afterwards. Exit 0 and exit 3 both
  confirm parent/child reads are retained once and no owned process remains.
  Total newly created payload across these cases is 12 MiB.
- Existing lifecycle tests cover ordinary completion, nonzero exit, invalid
  child accounting, pipe overflow and read/wall/RSS-triggered signal cleanup.
- Independent arithmetic covers nonzero baselines, zero child reads, activity
  during wait, subsequent parent reads and repeated observation without mutation.
- Independent negative cases cover absent, boolean, fractional and negative
  baseline/child values and missing post-reap evidence. Invalid evidence cannot
  discount the full conservative charge.
- AST comparison confirms all original functions except `monitor` and `run`
  remain identical. Only the two explicit accounting helpers are added. Limits,
  controls, source pinning, child reader and publication functions are unchanged.

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 timeout 180s taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python -c \
  'import resource,runpy; resource.setrlimit(resource.RLIMIT_AS,(2<<30,2<<30)); runpy.run_module("pytest",run_name="__main__")' \
  -q reports/st_world_policy/20261017/full_future_metadata_preparation_v1/test_run_build_once.py \
  reports/st_world_policy/20261017/full_future_metadata_preparation_v1/test_prepare_controls.py \
  reports/st_world_policy/20261017/full_future_metadata_preparation_v1/test_terminal_accounting.py \
  reports/st_world_policy/20261017/correctness/metadata_terminal_accounting_review_v1/test_independent.py
```

CPU affinity 0–1, inherited 2 GiB per-process address-space ceiling and
180-second outer bound; no visible CUDA. No actual builder, raw log, image,
teacher tensor, model checkpoint or campaign ledger was accessed.

## Decision boundary

The duplicate terminal child addend is removed without dropping parent activity
or weakening missing-evidence handling. No correctness blocker was found in
this narrow patch. It is suitable for a future separately admitted invocation;
it is not permission to modify/restart an active controller or recalculate
previous authoritative charges. This PASS does not certify actual full-build
completion, new resource capacity, teacher numerics or training throughput.
