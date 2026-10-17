# Narrow controller lifecycle repair

Base: `31c3dca2` (contains integrated candidate and independent `b8e36e67`
counterexample review). Worktree:
`/mnt/project/VLA-Drive-st-spatial-probe-supervisor-fix-20261017`.
Only the owned preparation directory changed. The original independent review,
its four defect reproductions, old candidate hashes/33-test record and old
PENDING admission draft remain historical evidence; they were not rewritten.
Use the new `HELPER_HASHES.json` for this repaired candidate's independent review.

SYS-01: scoped SIGTERM/SIGINT handlers request termination through the same
cleanup path as a timeout/exception. They do not interrupt cleanup. The child
is still a separate newly spawned session. A supervisor signal produces a
stopped result and preserves the consumed attempt; no retry/resume was added.

SYS-02: the supervisor retains the actual Popen leader **unreaped** using
`waitid(WNOWAIT)` until its session is closed. This reserves its PID/SID even
when the leader exits. Only matching live `/proc` session members with a
non-earlier start time are accepted; identities are checked again after
`pidfd_open`. Signals use these pidfds, never a raw/recycled PID or PGID and
never a historical receipt. A scoped Linux subreaper lets the supervisor reap
its verified orphaned descendants. A successful leader leaving a live tail is
reported `OWNED_DESCENDANTS_AFTER_LEADER_EXIT` and that tail is stopped; it is
not a successful stage. Normal child completion, normal waited descendants,
and spontaneous worker signal exit all reach an empty owned session before
return. Existing unrelated processes are not members and are never signalled.

TERM grace remains 30 seconds, followed by up to 10 seconds for KILL/closure;
cleanup uses one deadline rather than restarting its grace on an exception.
If discovery fails, known owned pidfds can still be stopped; no unknown PID is
guessed and no success is issued. SIGKILL of the supervisor and uninterruptible
kernel operations cannot be handled as an OS hard limit; existing qualifications
about sampled resource limits remain. This controller assumes the fixed CLI
does not deliberately daemonize into unrelated sessions; no such behavior was
added to the runtime. There is no new production process inventory artifact.

SYS-03/04: the same bounded control writer now rejects symlink ancestors before
creating output directories, and checks the existing directory envelope even
when reusing identical immutable bytes. Existing serialization/staging/reserve
and lock budgets are unchanged.

`test_lifetime_fix.py` adapts all four independent counterexamples into required
post-fix behavior and adds SIGINT, normal return, naturally waited children,
worker signal exit, tail retaining stdout, and changed spawn identity checks.
The original 33 tests are also run. All tests create only tiny CPU fixtures,
use two CPUs with empty CUDA, and signal only their own freshly recorded tasks.
No real admission, GPU, model, native/image/label payload, ledger, plan,
extraction, fit, export or scoring task was run.

The observer/API, training/source distinction, stage sequence, single-use
markers, input populations, exact checkpoint, runtime/helper pin checks, read
and storage budgets remain unchanged. Lead must integrate this fix, obtain the
original independent reviewer's re-review, freeze the final v2 runtime, and
regenerate controls/admissions with its exact SHA and repaired helper hashes.
This handoff supplies no self-issued resource or execution PASS.
