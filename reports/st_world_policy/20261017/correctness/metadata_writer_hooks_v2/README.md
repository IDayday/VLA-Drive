# Independent writer-hook and CLI re-review

**PASS_HOOK_INTEGRATION_LIMITED** at fixed Lead commit
`bac46066be1de9e0248f321d30412e8819b6467f`.
Reviewer `/root/st_future_metadata_correctness` is separate from the authors.
The previous failing review and logs remain in `../metadata_writer_hooks_v1/`.
This approval covers code and synthetic CPU execution only.

## What changed and what was retested

The previous real subprocess review found that the `python -m` entrypoint held
the guard in `__main__`, while writers imported a second canonical module. The
CLI reported missing callbacks only after publishing COMPLETE. Lead's minimal
fix delegates the entrypoint to the canonical module's `main()`. Guard and writer
callbacks now share the same state; no guard or publication check was removed.

The exact fixed implementation was tested in a separate clean detached worktree:

`/mnt/project/VLA-Drive-st-metadata-hooks-review-runtime-bac46066`

Exact HEAD, clean status and actual imported module paths were asserted before
testing. `independent_hooks_tests.log` records each module SHA256. The only
implementation difference from the initial reviewed integration `b3b7d79f` is
the six-line entrypoint change; intervening changes are tests and reports.

| Independent execution on the fixed source | Result | Full log |
|---|---|---|
| Expanded reviewer hook/real-CLI tests | **22 passed, 33.67 s** | `independent_hooks_tests.log` |
| Prior 35 reviewer boundary/ledger cases plus 8 owner hook cases | **43 passed, 5.52 s** | `regression_tests.log` |

No skipped or xfailed cases. The existing pynvml deprecation warning appeared.
The earlier 127 passing regression cases remain evidence for `b3b7d79f`, not
silently relabelled as executions on the fixed commit.

Environment: Python 3.10, `/root/miniconda3/envs/ddp/bin/python`, CPU affinity
0,1; OMP/MKL/OpenBLAS threads 2; CUDA hidden; bytecode writes disabled. Subprocess
tests inherit the same limits. Every input was synthetic temporary fixture data.
No actual raw log, image, ego/teacher tensor, teacher model, GPU, production PLAN
or real-data build was used. No campaign storage policy/ledger was changed.

## Behavior covered

- Both actual `python -m tools.st_world_policy.metadata_storage run` routes:
  current index and future preparation, initial completion and same-ledger reuse.
  Successful receipts contain positive writer and publication callback counts.
- Real child processes reject a forbidden COMPLETE destination before creating
  its temporary or final file. Prior legal artifacts are retained as evidence.
  A 1 MiB RSS limit and wrong source SHA also reject without COMPLETE. Wrong
  source pins reject before metadata admission; same-process sentinels also prove
  the parser and builder are not called.
- Injected same-process failures before SQLite/future-shard manifest publication
  leave no published target or COMPLETE and retain the charged claim.
- Optional `source_population` callback stops before the first metadata open;
  a multi-MiB synthetic document is read in 1 MiB chunks and returns the same
  validated source/index/binding as the original no-callback path.
- SQLite's real VM progress handler runs at its registered 1000-operation
  interval; injected failure prevents COMPLETE publication.
- Two worker threads use the admitted process-global guard and its RLock.
  An explicit sentinel proves callbacks do not reacquire the original ledger
  flock. Production qualification remains one worker; this test only checks
  thread-safety of the API.
- The 1-second throttle applies to ordinary census only. Every explicit path,
  creation or `before_commit` call checks immediately, including repeated calls
  within one second. It is not a one-second publication grace period.
- Prior symlink/hardlink namespace rejection, semantic COMPLETE validation and
  bounded same-byte ledger verification remain passing.

`SEMANTIC_SCOPE.json` compares the fixed implementation with the earlier boundary
fix. Original frame selection, missing-view handling, source/log grouping and
row ordering are unchanged after removing the new check-only expressions.
The original target/history helpers and provenance validator are byte-identical.
The lookup constructor changes a dictionary comprehension to its original-order
loop so it can check resources; duplicate-token rejection remains. No GT enters
feature/target selection. These are code/fixture findings, not a claim that the
real 32 scenes have already been compared.

## Reproduction

From the fixed detached implementation, with its directory first in PYTHONPATH:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python -m pytest \
  --import-mode=importlib \
  /mnt/project/VLA-Drive-st-future-metadata-correctness-20261017/tests/st_world/test_review_metadata_writer_hooks.py -q

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python -m pytest \
  --import-mode=importlib \
  /mnt/project/VLA-Drive-st-future-metadata-correctness-20261017/tests/st_world/test_review_future_metadata.py \
  tests/st_world/test_metadata_writer_hooks.py -q
```

The recorded executions used an equivalent `pytest.main` harness with mandatory
HEAD/path/hash assertions before pytest. No implementation source is changed by
these tests.

## Permission boundary and remaining wiring

This review removes the writer-hook correctness blocker to **Lead subsequently
registering** the fixed 32-scene/two-log qualification. It does not itself
approve storage resources or execute that qualification. The independent storage
review, valid source pins and a finite registered invocation are still required.

For that invocation:

1. Preserve the exact scene/log order, one worker and 32 MiB qualification
   metadata. Create a new immutable PLAN pinning the final implementation; keep
   the old v4 PLAN as evidence, since its old implementation pins are stale.
2. Declare reference and new-builder `max_read_bytes` and `max_seconds` separately
   with **sums <=8 GiB and <=900 s**. Do not give both paths a fresh 8 GiB/900 s.
   Meter all raw-log hashing/unpickling and image hash/header reads in both paths,
   including repeated qualification reads. A failed/retried leg must not silently
   reset this joint allowance. Source/control metadata allowances must also be
   explicit. Sequential execution keeps the single-worker and aggregate
   <=64 MiB/s requirement straightforward.
3. Bind the new-builder's limits into its PLAN/registration and the reference
   limits into its recorded command/receipt. Check their sum before either is
   started; record consumed bytes by category and elapsed time after each leg.
   `PreparationBudget` is per invocation and cannot alone enforce a joint budget.
4. Keep <=4 GiB process-tree RSS and a finite outer monitor for blocked native
   operations, while retaining all synchronous pre-publication callbacks. The
   monitor does not substitute for callback correctness.
5. Charge qualification plus future preparation/index work against the same
   aggregate <=4 GiB metadata package. The 32 MiB qualification is part of that
   package, not an extra allocation. No teacher/tensor/GPU work is allowed here.

Actual original-versus-new metadata equivalence, storage throughput/capacity,
full-population preparation and training integration remain **NOT_EXECUTED** in
this report. Resource approval remains the independent storage reviewer's scope.
