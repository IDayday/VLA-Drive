# Independent request-index execution review

2026-10-17. **FAIL: two narrowly reproducible execution-boundary defects.**
The index entry's 53 unit cases passing does not make the composed execution
path ready to dispatch.

Reviewed commit: `15967d1de0615ed104a2c326a09b4ac7dd63d2ea`.
Adapter SHA256: `e20b3f9e0317bd7b122b0fa9d74f99f8b6bc11cc12a598db70524bcb83bdbc0f`.
Pinned shared monitor SHA256:
`a775a87532dab70a376cb64a7c0c19403e21e2f3a3b8040152fbefb8cebf10b9`.
Isolated tree: `/mnt/project/VLA-Drive-st-full-future-index-review-20261017`.

## F1: stale leader snapshot misclassified as surviving descendants

The shared monitor calls `active=owner.discover()` and then
`done=owner.leader_done()`. Discovery includes the live leader itself. If the
leader exits between those two observations, `done and active` becomes true
even though no descendant exists. It sets `OWNED_DESCENDANTS_AFTER_LEADER_EXIT`
and enters termination handling for a completed child.

The owner's retained real-child failure is independently confirmed with a
deterministic scheduling fixture in
`test_exit_between_discover_and_done_is_not_a_descendant[False]`. The first
snapshot has only the live leader, the exit check is true and every subsequent
scan is empty. Nevertheless the monitor returns that failure. The matching
real-descendant fixture passes: a surviving child still triggers rejection and
cleanup. Both retain exactly one reap, no owned residue and the corrected
150-byte terminal charge. Tests use local pipes and fake ownership records;
they send no signals to real processes.

Minimal fix boundary: when the exit check reports done, refresh discovery
before using active membership for classification/loop termination. Keep the
unreaped anchor, identity checks, pidfds, read accounting and cleanup. Do not
ignore all `done and active` states or suppress genuine descendants. This is
one shared-monitor lifecycle fix, not a new controller. Pin the resulting exact
core hash in the adapter, then rerun the deterministic race, genuine-descendant
case and existing normal/nonzero/signal cleanup tests. Do not hot-edit an active
controller or erase the prior real failure.

## F2: small COMPLETE reader has an unbounded growth window

`run_index_once.py::read_small_actual` checks an initial lstat against 1 MiB,
then calls `Path.read_bytes()` without a read bound. If a file grows after that
stat, the entire grown file is read before version mismatch rejection.

`test_complete_growth_is_bounded_before_rejection` starts with a two-byte
private JSON file and grows it to 2 MiB just after the initial stat snapshot.
The actual function reads **2,097,152 bytes**, exceeding its intended
1,048,577-byte bounded-read ceiling, then raises. Rejection after consuming the
data does not enforce the read/memory boundary. Only this small temporary file
is used; no real receipt or SQLite body is read.

Minimal fix: no-follow/nonblocking descriptor open, regular/private descriptor
checks and at most `MiB+1` bytes; retain file/descriptor version consistency,
length, JSON and receipt identity checks. Nonblocking open avoids introducing
the already-known FIFO-before-fstat problem. No new I/O framework is required.

## Confirmed correct scope

Code inspection of the original `index_teacher_metadata.run` and
`build_index` confirms the actual CLI returns a ten-field projection without
`artifact.status`. The adapter correctly validates that projection against
`st_teacher_metadata_index_v1`, COMPLETE and canonical_json_zlib_level1_v1,
the role request identity/hash, exact source/contract, scene count and
metadata-only fields. It does not substitute the future-builder schema.

The additive-policy check preserves every non-claim field and each existing
claim exactly. Only the two exact index claims are appended. Independent
negatives additionally reject duplicate old/new claims, changed legacy
reservations and a different state directory. Existing index role budgets,
same-ledger writing, immutable start/result markers and failed-stage full-charge
behavior remain in place. No fictitious migration receipt is introduced for an
unchanged core. No actual policy migration or registration was performed here.

## Actual test evidence and limits

**2 failed, 58 passed in 0.14s**: all 53 adapter tests independently rerun,
plus seven independent cases. The exact final log is `TEST_FAIL.log`. Owner's
earlier `113 passed / 1 failed` combined-run log remains unchanged; this review
does not replace it with the 53 entry tests or claim the full chain passed.

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 timeout 180s taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python -c \
  'import resource,runpy; resource.setrlimit(resource.RLIMIT_AS,(2<<30,2<<30)); runpy.run_module("pytest",run_name="__main__")' \
  -q reports/st_world_policy/20261017/full_future_index_execution_v1/test_run_index_once.py \
  reports/st_world_policy/20261017/correctness/full_future_index_execution_review_v1/test_independent.py
```

CPU2, inherited 2 GiB per-process address-space ceiling, 180-second outer bound,
no visible CUDA. Temporary fixtures are at most 2 MiB; no actual indexer, raw
logs, images, weights, tensor payload or live ledger is accessed. No dispatch,
live source change or resource approval occurs. Exact fixes and narrow
independent regression are required before execution admission.
