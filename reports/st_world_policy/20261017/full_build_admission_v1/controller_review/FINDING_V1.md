# Independent bounded build controller check: initial version

Reviewed owner commit `01886c1afee5d96c8ae2d234ca905d2b9638c55e`,
`run_build_once.py` SHA
`10ece34548c88069e765de95c5053688ac104de5862e45fc01c342882bb6fa67`.
Scope is the thin metadata build controller only. No real builder, ledger
migration, image/log payload, tensor, model or GPU was invoked.

Thirty targeted CPU tests ran: 20 owner cases and 10 independent cases, including
seven fresh tiny CPU children. Initial result: **29 passed, 1 failed**.
Success, RSS limit, source-read limit, wall limit, stale counter, pipe overflow,
control tampering/symlink and escaped-result headroom checks behaved as intended.

## FBC-01: a corrupt final counter masks the original error after successful cleanup

A bounded fixture writes malformed `CONTROL_BUILD_COUNTERS.json`, then launches
only its own tiny Python sleeper. The poll detects `MONITOR_UNREADABLE`, correctly
terminates the child and `owner.finish()` reaps it. The final `read_counter()`
raises `JSONDecodeError`. The broad exception handler invokes `owner.finish()` a
second time; the already reaped PID no longer exists in `/proc`, so
`FileNotFoundError` replaces the original counter error.

This is a failure-diagnostic/cleanup-state bug, not evidence of foreign process
signalling, surviving children or a resource-limit bypass. Request a minimal
finished/reaped flag to skip duplicate cleanup after successful `finish`, and
retain the monitor exception type/message. Do not change the original owned
session helper or build runtime. The captured traceback and original test output
are retained in `TEST_OUTPUT.txt`; the independent fixture must pass after the
small owner correction. No actual build admission is issued for this version.
