# Spawn setup ownership follow-up

Parent: `a217c98e135f7c8f7fd54cfeae9b5c1256924eb3`. Worktree:
`/mnt/project/VLA-Drive-st-spatial-probe-supervisor-fix-20261017`.
Owned paths are confined to this preparation directory. Historical candidate,
independent counterexamples, v1 repair report, 43-test record and PENDING
admission remain unchanged.

The independent reviewer reproduced an orphan if `OwnedSession.__init__` or
selector registration raised after Popen returned. The initial leader pidfd,
OwnedSession registration, selector construction and selector registration now
all execute in the protected cleanup region. A partially registered actual
owner is recovered only from this same Popen object. Selector/owner/fd cleanup
handles absent initialization. The normal monitoring and completion path is
unchanged.

A setup failure still fails the consumed stage. Its fallback is restricted to
the freshly returned `start_new_session=True` Popen child: `waitid(WNOWAIT)`
first proves that it is our unreaped child; kernel SID and PGID must equal its
PID before a group signal. The unreaped leader reserves that identity, including
when it is a zombie. This exceptional cleanup uses SIGKILL on that anchored
fresh group and on the already opened leader pidfd, then scans the same session
with double identity checks and pidfds to stop/reap verified remaining members,
including a descendant in another process group. The leader is reaped only
after the cleanup signals. It never reconstructs authority from a receipt.
If session verification fails, only the actual Popen child is stopped; a foreign
group is not signalled. If `/proc` scanning itself fails, the final anchored
fresh-group signal and actual-child wait still run and the stage stays failed;
no complete-session success is claimed from unreadable process metadata. The
fixed CLI must not deliberately escape its session, as in the prior repair.

The exceptional setup path has at most a ten-second closure wait and respects
an already-started 40-second cleanup deadline (plus at most 0.1 seconds for a
final direct-child reap attempt at an expired deadline). Normal TERM/KILL grace
and all five stage limits, pins, single-use admission, helper hashes, locks,
serialization envelopes and storage/read limits are unchanged. The fallback
adds no retry or execution approval.

50 small CPU tests passed in 3.37 seconds, retaining all 43 prior checks. Seven
new cases inject ownership construction, selector construction, selector
registration, process metadata and initial child-pidfd failures; two cases make
the leader exit before registration fails, with its descendant either retaining
the group or moving to another group within the same session. Tests verify
actual child exit, descendant closure, inherited lock release, and an unrelated
new-session fixture remaining alive. Fixtures alone receive signals. No real
stage, GPU, model/native/image/label payload, ledger, export or scoring was used.

The original reviewer must independently re-review this exact candidate. Lead
alone integrates, freezes the final runtime and creates actual admissions with
its helper hashes. This is no formal PASS or permission to dispatch.
