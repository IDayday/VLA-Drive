# Independent systems findings: narrow corrective delta

The independent review reproduced an orphan on parent monitor exception in
55721af7: a temporary-file stat race could make the parent exit, release its run
lock and omit summary while its spawned worker kept running. No real diagnostic
was authorized or executed with that version.

Fixes preserve geometry, all scientific constants and actual resource limits:

- Parent signal/monitor paths now always enter bounded owned-worker cleanup.
  TERM and join up to15s, then KILL/join up to3s if needed. Only the Process object
  created by this invocation is targeted. Exit confirmation and cleanup errors
  are explicit; the lock covers cleanup and terminal summary. SIGTERM/SIGINT
  become controlled interruption; repeat signals cannot interrupt cleanup, and
  original handlers are restored afterward.
- Native crash, monitor exception, operator signal and deadline all produce a
  terminal attempt summary retaining the full fixed32 scene denominator. Summary
  parsing failure itself produces32 unresolved rows, never a scientific PASS.
- Output scanning performs one lstat per entry, skips a vanished temporary file,
  and propagates other IO errors through cleanup. No is_file/stat race remains.
- Spawn's known CPython resource_tracker PID is included in aggregate RSS and
  reported, without broad process discovery or signalling that helper.
- Asset reads use a bounded `read(min(size+1,remaining_budget))`. They reserve that
  bound first, then charge actual returned bytes; IO exceptions retain the bounded
  conservative charge. Growth after stat therefore cannot cause an unbounded
  read before the existing hash/stat rejection. Control JSON reads are also
  bounded. The original256MiB asset budget is unchanged.

`SUPERVISOR_FIX_TESTS_TIMEOUT.txt`: **50 passed**,11.729s wall,
823448KiB peak parentRSS /642312KiB peak childRSS; no CUDA, real asset read or
real official call. Actual spawned-process fixtures cover native exit23,
monitor FileNotFoundError, SIGTERM, SIGINT and deadline; every worker exited,
all32 summary rows remained, and signal handlers restored. Separate fixtures
cover stat-then-file-growth and temporary unlink versus real permission errors.
The first attempt ran from stdin, which cannot be reopened by multiprocessing
spawn: one native-exit fixture failed for that harness reason. Its output is
retained; the corrected committed `run_cpu_checks.py` has a real guarded main
path. The49-test and final50-test runs are separate evidence, not replications.

The shared atomic writer fsyncs each file before linking its immutable name.
This supports the tested process-crash/no-retry accounting boundary. No claim of
host-power-loss durability is made: that would additionally require validated
filesystem semantics and parent-directory persistence. No shared writer was
modified in this narrow fix. Independent delta review remains required before
real execution.
