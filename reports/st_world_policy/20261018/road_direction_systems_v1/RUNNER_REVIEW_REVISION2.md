# Exact v2 runner resource review

**APPROVE_ROAD_DIRECTION_EXECUTION**, resources only, for registration
`d6ad347867b77a274de976d62e07a0ab9c142eb46336c7e7de9e0198185c4152`
and frozen integrated commit `83679937fe9702be39f0184154fdf9ebdc4eb98a`.
The Lead must separately bind this review and independent correctness approval
into a time/host-specific admission. This is not formal-training approval.

The previous monitor-error orphan failure is preserved in
`RUNNER_FAULTS_ATTEMPT1.json`. Narrow fix `17c3e92c` now performs owned worker
cleanup in finally, keeps the run lock until exit confirmation, handles parent
TERM/INT, restores signal handlers and emits all32scene statuses. A disappearing
atomic temporary file is skipped via one lstat; other monitor errors are failures.
Bounded reads reserve a maximum before IO and charge actual bytes, preventing
post-stat file growth from exceeding the read allowance. The known spawn resource
tracker is included in aggregate RSS.

Independent re-execution of the original two-process fault witness confirmed:

- Native child exit23: nonzero result, complete32-row unresolved denominator,
  child exited, no residual fixture workers.
- Parent monitor FileNotFoundError: nonzero result, complete32-row denominator,
  child exited, no residual fixture workers. The prior orphan is fixed.

An independent fixture suite run passed **20 tests in6.83s** (wrapper7.533s),
including actual spawned-child native exit, monitor error, parent TERM/INT and
shortened fixture deadline. The test suite consumed about7.14CPU seconds across
parent and child, peak parent640900KiB and child640588KiB; the two-scenario replay
used a further3.20CPU seconds. No metric-cache, rollout or model asset was loaded,
no GPU initialized, no official score/update performed. An audit hook refused
actual campaign payload reads. Fixture admissions are explicitly fake controls
for systems behavior, not real launch evidence.

The frozen runtime is clean and all six registered source hashes exactly match
the independently tested source. New evidence is kept in REVISION2 files; the
old registration/reviews/failure evidence were not overwritten.

The actual single-worker run is bounded to1800s,8GiB aggregate RSS/address space,
256MiB logical asset reads,64MiB LZMA expansion per cache,1GiB outputs and128calls
including failures/uncertain invocations. No automatic retry; originals are
reused only with exact full identity. Calls are durably charged before backend
entry. A native failure produces a32-row incomplete result rather than success.
The parent handles worker native failure and bounded TERM/KILL escalation; this
review does not claim host-power-loss durability or authorize killing the parent.

Fresh read-only host/memory/disk observations are in the JSON review. Local host
is permitted and has capacity for one CPU worker and the small bounded output;
no GPU or pressure-worker release is needed. No unrelated workload should be
stopped. Runtime is not yet a throughput measurement: actual map geometry can
consume the budget, and undefined/saturated directions remain in all32denominators.
All originalDAC=1; there is no violation-repair denominator or policy-learning
claim. Feature-side research remains paused.
