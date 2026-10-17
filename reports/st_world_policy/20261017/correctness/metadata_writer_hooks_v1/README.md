# Independent metadata writer-hook review: changes requested

Reviewed fixed Lead integration `b3b7d79f65ade003cae116e7c4b3d0f935b4799e`
on 2026-10-17 UTC. Reviewer is separate from implementation owners. This is a
CPU synthetic-fixture review, not an actual-data or storage qualification.

**CHANGES_REQUESTED:** the real `python -m` CLI does not share its active guard
with imported writer callbacks. The two subprocess tests fail although all
same-process tests pass. No real qualification may be inferred from those passes.

## Finding H1: CLI module identity splits the guard

At `tools/st_world_policy/metadata_storage.py:730`, `python -m
tools.st_world_policy.metadata_storage run ...` calls `main()` in `__main__`.
`metadata_writing()` sets that module's `_ACTIVE_GUARD`. The writers import
`check_active_metadata_storage` from the canonical module name, creating a
different module with `_ACTIVE_GUARD = None`. Their callbacks silently return.

Both real subprocess commands execute their fixture build, publish
`COMPLETE.json`, and only then return exit 2 with:

> metadata writer never called required resource/publication hook

This is a publication safety defect, not merely an overly strict final receipt.
It bypasses the guard's runtime checks until the builder has returned. The tiny
synthetic artifacts and their SHA256 are recorded in `FAILED_CLI_ARTIFACTS.json`.
No real assets were used or written.

Minimal reproduction is
`test_review_metadata_writer_hooks.py::test_real_subprocess_cli_completes_and_reuses_synthetic_only`,
parameterized for `current_index` and `future_preparation`. It constructs a
registered fixture, then invokes the actual subprocess entrypoint with exact
policy, registration and checkpoint-receipt hashes. It does not monkeypatch the
child or call `main()` directly.

The correction must make the CLI and writer imports use the same canonical
guard instance, retaining all guards and source-pin checks. Retest real
subprocess success/resume and rejection before COMPLETE publication. Do not
remove the missing-hook assertion or treat final rejection as sufficient.

## Actual evidence

Fixed detached implementation worktree:
`/mnt/project/VLA-Drive-st-metadata-hooks-review-runtime-b3b7d79f`.
Exact HEAD, clean status, imported paths and SHA256 were asserted before testing.
The source identities are in `independent_hooks_tests.log`.

Environment: Python 3.10, `/root/miniconda3/envs/ddp/bin/python`, CPU affinity
0,1, OMP/MKL/OpenBLAS threads 2, CUDA hidden, bytecode writes disabled.

| Execution | Result | Evidence |
|---|---|---|
| New independent hooks suite | 14 passed, 2 failed, 9.67 s | `independent_hooks_tests.log` |
| Previous 35 independent boundary/ledger cases + 84 owner regressions + 8 owner hook cases | 127 passed, 9.78 s | `regression_tests.log` |

No skip or xfail. Existing pynvml deprecation warning only.

The passing cases cover source-pin rejection before parser/build/admission;
same-process rejection before SQLite and future artifact publication; optional
source-population callback before metadata open and at 1 MiB read boundaries;
SQLite VM callbacks and abort; 1-second census throttling; worker callbacks using
the already-held guard without reentering the campaign flock. The old symlink,
hardlink, COMPLETE-validation and bounded-ledger fixes have no detected regression.

The 1-second throttle applies only to ordinary census calls. Explicit path,
creation and `before_commit` calls bypass it. This is preserved in the tests;
publication must never rely on a later periodic census.

Code review found no target-time, tie-break, missing-view, scene/log-order or
label-input changes: the new optional callback wraps the existing reads and
checks. The default `source_population` route is preserved; the callback route
returns the same binding on synthetic data. Real source/target equivalence has
not been rerun here.

## Permission and next qualification boundary

No real raw log, image, teacher/tensor or GPU was accessed. No actual PLAN/build,
new reservation or storage admission was executed. The fixture builds only use
temporary synthetic metadata. The separate storage reviewer owns resource
approval; this review does not replace it.

After this CLI defect is fixed and independently retested, Lead can consider
registering the fixed 32-scene/two-log comparison. A fresh immutable PLAN must
pin final code while retaining the old scene/log order, one worker and 32 MiB
qualification metadata. The old v4 PLAN remains historical evidence.

Reference and new executions need separate explicit read/time allocations whose
**sum is at most 8 GiB and 900 s**, not 8 GiB/900 s each. The new preparation
budget currently starts fresh per invocation; the legacy reference also needs
metered hash/unpickle/image-header reads. Preserve aggregate <=64 MiB/s and
<=4 GiB RSS, zero teacher/GPU work, and the existing combined preparation/index
storage reservation. These are concrete required runtime hookups, not resource
PASS claims from this CPU test.
