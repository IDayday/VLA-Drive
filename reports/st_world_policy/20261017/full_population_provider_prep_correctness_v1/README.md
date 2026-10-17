# Independent narrow provider-preparation correctness review

Verdict: **PASS_NARROW_WRAPPER_CORRECTNESS_ONLY** for source
`06aae622c49d18f98d366895d3287dbbd120fd19` and the exact PLAN recorded in
`REVIEW.json`. This does not assert a successful actual factory run, approve
resources, authorize training, or supersede retained failed upstream evidence.

The review ran 16 CPU tests independently: the author's 9 narrow fixtures and
7 additional tests. No implementation file was changed. Full output is in
`TEST.log`; the independently added tests are in `verify_wrapper.py`.

The wrapper's `original_cli` (lines 109–119) accepts normal return and
`SystemExit(0/None)`, restores the original argv object, and raises on a nonzero
CLI exit. Real `runpy` execution of a tiny synthetic module verified success,
failure, argv restoration, and multiple printed JSON values. Both original
CLIs' stdout is redirected to the existing bounded stderr stream. The child
retains one final JSON accounting envelope on stdout.

The actual wrapper child branch (lines 155–179), with synthetic CLI calls and
only temporary JSON outputs, executed these cases:

- Both stages exit successfully: source migration precedes provider creation,
  stage two sees its required source-migration file, and all five output hashes
  appear in the sole final envelope.
- Stage one fails: stage two is never called and no success result is emitted.
- Stage two fails: stage one's artifact remains, no retry occurs, and the child
  returns nonzero with `cli_result=null`.
- A required configured output is absent: the result remains incomplete and
  exits nonzero even though both synthetic CLIs returned success.

These child tests stub only external imports, execution environment checks and
resource observation. They exercise the wrapper's real ordering, exception
handling, output collection and envelope code. They do **not** run the original
provider against real indices or validate actual cleanup.

`validate` (lines 68–106) and the existing pinned bounded `Reads` implementation
were tested together using small fixture files. Changed TRAIN/current index
receipts, reference config, teacher config, finalized future receipt, PCA index
receipt and provider source are rejected by checksum. The initial PLAN is
itself hash-pinned; the runtime commit, exact controller paths/pins, independent
admission status, fit300 prerequisite, limits and output root remain checked.
The real two CLI parser definitions were compiled in isolation from their AST
and accepted the wrapper's exact argument lists. No module-level imports or
real data access from those CLIs were executed.

Static source checks confirm `ROOT.exists()` rejects a second parent invocation
before monitor/lock launch (line 180); there is no auto-retry loop. The original
`4b2f0f…` monitor, lock, deadline, post-reap accounting and cleanup predicates
remain in the call chain. Actual resource behavior belongs to the separate
resource review. The pinned source files match the already-reviewed frozen
`a8b0036f33554fd2a00ea1af8c9b5f20fb6b965a` runtime. The underlying factory still
checks original full-population order and finalized TRAIN lineage; this review
did not repeat that algorithm's tests or traverse 101592 rows.

Compared with the reviewed draft recipe, only previously unassigned execution
values are filled (`max_seconds=1500`, `save_every=64`, isolated output root)
and the descriptive adaptation reason is expanded. Model, loss, precision,
population, scheduler, global batch and cumulative 64-update/128-exposure
diagnostic budget are unchanged. Neither this wrapper nor these tests begin
training. The provider's existing configuration function changes only its
declared data execution source/PCA scope and isolated configured output.

Execution:

```bash
taskset -c 6,7 timeout --signal=TERM --kill-after=5s 120s \
  /root/miniconda3/envs/ddp/bin/python -B \
  reports/st_world_policy/20261017/full_population_provider_prep_correctness_v1/verify_wrapper.py \
  > reports/st_world_policy/20261017/full_population_provider_prep_correctness_v1/TEST.log 2>&1
```

Measured test cost: 0.1369 seconds, 30,855,168 bytes peak RSS, 2,370,211 logical
read bytes, 15,006 logical written bytes, CPU affinity 6/7. The test imposes
1 GiB address-space and 110-second in-process wall limits. Source/control
inspection and repository metadata operations are separate from this measured
test process; no broad data inventory was performed. All task reads remain
below the 64 MiB authorization. No Torch/GPU, actual DB/PCA/model/image/tensor,
ledger or dispatch was used.

Next actual evidence is the original factory's single bounded invocation:
its final controls, actual configured descriptor/PCA scope/lineage hashes,
complete population/order validation and resource accounting. A runtime failure
must retain its evidence and must not inherit this fixture PASS as an actual
success. Resource admission and the Lead's current-input checks are separate.
