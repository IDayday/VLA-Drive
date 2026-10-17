# Preserve the real failed build; remove observer writes from its scanned tree

Lead's first real dev build failed after 48.58698667772114 seconds. Its unchanged receipt is:

`/mnt/project/st-world-policy-20261016/caches/full_future_metadata_dev_v1/CONTROL_BUILD_RESULT.json`

SHA256 `60b7ba6177b51081fdb323b36a0821f3848e6ddf2c2bb625b97e9a8c850aefda`.

The exact cause was `prepare_future_metadata.metadata_inventory`: scandir enumerated `CONTROL_BUILD_COUNTERS.json.4a1o_q1m.tmp`, the observer atomically renamed it, then entry.stat raised FileNotFoundError. This was interference from the controller's background metadata writer. It was not a failed input or a resource overrun. The outer controller retained the error, cleaned its one owned process and reported zero live owned processes. Actual charged parent+child rchar was **2,202,810,181 bytes**. Peak aggregate RSS was 772,366,336 bytes. No model or teacher tensor was generated.

This follow-up changes only the outer controller. The immutable dev/train runtimes, PLAN files, registrations, numerical code and existing failed/partial outputs remain unchanged. There is no blanket ENOENT handling, ignored temporary file or source integrity relaxation.

## Observation without filesystem mutation

During the builder execution, the controller creates no counter file or observer temporary file. The parent reads `/proc/<owned leader>/io` only after checking its pinned in-memory session/start identity, checks that identity again after the bounded read, and samples aggregate tree RSS. The process leader remains unreaped until the terminal IO sample is taken. Existing OwnedSession/pidfd cleanup remains unchanged, including the prior finished flag that prevents a second cleanup.

The original Counters implementation is reused once in the child at completion, with its publication sink temporarily directed to RAM. Its final stdout envelope contains the original CLI result unchanged, any actual failure type/message, the pre-import baseline, the final absolute counters and the exact nonnegative delta. No monitor thread runs. Traceback printing occurs before final accounting so diagnostic source reads are included. Parent stdout/stderr still have independent 1 MiB ceilings and actual escaped JSON/atomic staging is precharged before the final receipt.

For active enforcement, child whole-process absolute rchar is charged (baseline zero, conservatively including taskset/Python startup), plus the parent's own pre-import-baseline delta. Final charge uses the maximum of the verified unreaped terminal sample, final child snapshot and active sample. If both final sources are unavailable, it charges the whole remaining allowance rather than treating an early sample or missing data as zero. Malformed final child accounting always fails completion. The fixed child writer uses one thread worker, not additional IO worker processes.

## Explicit second attempt, no automatic retry

Dev attempt 2 is recognized only with a new independent admission containing:

- `attempt: 2`, role dev, status `PASS_BOUNDED_DEV_BUILD_ATTEMPT2_ONLY`;
- the exact immutable failed receipt path/hash above and verified zero owned residue;
- an actual invocation read ceiling **14,977,059,003 bytes = 16 GiB - 2,202,810,181 bytes**;
- unchanged CPU 0/1, 4 GiB RSS, no GPU, 1800-second per-invocation wall and 40-second cleanup;
- the newly reviewed controller hash and fresh checkpoint/lease bindings.

The new START/RESULT names are `CONTROL_BUILD_ATTEMPT2_STARTED.json` and `CONTROL_BUILD_ATTEMPT2_RESULT.json`. Both are one-shot. No generic third attempt or train retry is added. The original START/RESULT/counter/lock and all valid log shards or partials are retained. The existing builder alone decides strict reuse of its immutable outputs. Creating a corrected controller is not authorization to execute attempt 2.

## Validation

CPU fixtures cover strict child accounting (reject bool, negative, fractional and inconsistent counts), exact prior receipt/remaining budget, bounded pipes, no repeated cleanup, missing terminal evidence charged conservatively, and unchanged preparation guards. Seven actual tiny owned CPU subprocesses exercise successful completion, genuine exit failure, malformed stdout accounting, pipe overflow, read, wall and RSS rejection. All assert zero residual owned processes; monitoring produces no `CONTROL*` files in the test writer root. These are controller tests, not a rerun of the real metadata build.

Historical failed RESULT and previous 31/32-test evidence remain intact. This owner did not dispatch a real build, delete partials or touch the ledger. An independent source/resource review and explicit Lead attempt-2 dispatch are still required.
