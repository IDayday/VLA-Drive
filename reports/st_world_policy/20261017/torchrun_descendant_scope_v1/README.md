This change repairs ownership discovery for **future reviewed invocations**. It
does not change the frozen T1 runtime or adapter used for the completed/live
native 2 → 4 qualification, and it does not retroactively qualify its resource
monitor. The historical `FAIL_LIVE_DESCENDANT_SCOPE` evidence remains intact.

The old helper accepted only the launcher's SID. Installed torchrun starts ranks
with `start_new_session=True`; ranks and their loaders can therefore sit outside
that SID. `owned_descendants.py` supplies an isolated facade around the unchanged
pidfd/subreaper helper. Membership is established through actual PPID edges from
the current unreaped Popen, with parent and child PID/start-tick checks bracketing
pidfd acquisition. Discovery follows the ancestry transitively across sessions.
Later `setsid()` and reparenting do not revoke a witnessed child's identity.
Signals still target exact pidfds, never numeric process groups, command lines,
GPU ownership, saved PIDs or all children of the supervisor.

Previously witnessed identities remain in evidence after reaping. This avoids
mistaking a stale snapshot of an owned, already reaped zombie for an unknown
child. Unwitnessed newly adopted children cause failure without claiming or
signalling them. Existing unrelated supervisor children are excluded using a
snapshot taken before the one task Popen. Do not start unrelated subprocesses
under the same supervisor during the controlled interval.

This remains a sampled process monitor, not a cgroup hard ceiling. A very short
lived descendant may escape live sampling. If its parent waits for it, Linux
transfers its cumulative IO to the observed parent. An orphan whose ancestry was
never witnessed cannot safely be cleaned by inference; it is an unresolved scope
failure. The implementation makes no guarantee of reconstructing missed peaks.
RSS sums can count shared pages more than once; a concurrent reap can make a
live IO sample conservative. Terminal charging reuses the unchanged post-wait
accounting core and does not add already absorbed child IO again.

The future T1 adapter now explicitly pins this helper and requires both review
receipts to bind it. Old admissions do not become valid automatically. Normal
and error cleanup receive the existing absolute deadline; no new grace period
is created after expiry. The scientific recipe, original trainer, native load
and RNG observer functions are unchanged (AST checks in `SOURCE_AUDIT.json`).

Actual CPU verification used cores 10 and 11 and empty CUDA visibility. The final
recorded run has **40 passed, one existing pynvml deprecation warning**, in 4.70s
wall time. Tests include a real launcher → new-session rank → new-session loader,
live memory/reads, abnormal leader exit and retained orphans, TERM cleanup,
unrelated process protection, PID reuse before/during pidfd acquisition, strict
deadlines, normal reap accounting and adapter integration. The real adapter
fixture observes all three processes and exceeds its own RSS by at least 16MiB;
all fixture processes are reaped. This is CPU process correctness, not a new GPU
qualification or a training result.

An initial run had four failures: the already-reaped witness/snapshot race and
three existing control tests that expected nonfinite time rejection before any
runtime read. The race was repaired; nonfinite time validation remains first.
The intermediate helper-only rerun passed 10 tests; the next full run passed 40.
The final recorded full run also passed 40. Tests did not read any model, image
or training checkpoint, dispatch SSH/GPU work or signal any real task.

The final harness measured 461,111,296 bytes maximum child RSS and 77,683,687
post-wait rchar bytes including runtime imports and fixture reads. The latter is
above 64MiB if the assigned code/control read cap is interpreted to include all
test imports and fixture IO. Code/control reads were not independently metered,
so no resource PASS is claimed. The Lead was informed and further tests stopped.
`TEST_RESULT.json` and `PYTEST.txt` retain the actual measurements and output.
The implementation author does not provide an independent acceptance verdict.

No historical T1 peak has been recalculated, no prior receipt has been replaced,
and no additional 2 → 4 training is requested. Independent review and a new
explicitly budgeted invocation are prerequisites for any future deployment.
