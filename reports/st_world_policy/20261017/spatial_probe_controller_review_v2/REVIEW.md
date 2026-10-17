# Learned-W controller lifecycle re-review

Candidate: `a217c98e135f7c8f7fd54cfeae9b5c1256924eb3`.
Lead integration `122a7f14` contains identical `supervise_once.py`,
`prepare_bindings.py`, and `observed_cli.py` bytes.

**Result: original SYS-01/02/03/04 reproductions are repaired, but complete
lifecycle qualification remains FAIL due to a post-spawn setup exception
window. No real W stage is authorized by this review.**

This is a narrow re-review, not a model/protocol or historical experiment audit.
No real plan, extraction, fit, GPU operation, checkpoint/image read, live ledger
mutation, production-process signal, or dispatch occurred. CPU tests ran on
affinity `{0,1}` with an inherited 2 GiB address-space ceiling, empty CUDA
visibility and two-thread environment limits. This is stronger than a 2 GiB
RSS allowance for these tiny tests, not a measurement of real extraction memory.

## Passed repairs

- **SYS-01:** supervised TERM and INT now enter bounded owned-session cleanup.
  An independent harness checked the worker and grandchild disappeared from
  `/proc`, inherited fixture lock became acquirable, and a simultaneously
  running foreign-session control remained alive.
- **SYS-02:** early successful leader exit no longer hides its running child.
  WNOWAIT retains the leader's PID/SID identity until cleanup; start ticks and
  pidfds identify observed members, and the subreaper reaps adopted descendants.
  Both inherited-pipe and closed-pipe cases are covered by the owner and
  independent suites. The result records a failure reason, not normal completion.
- **SYS-03:** ancestor symlinks are rejected before the fixture output directory
  is created. Read/write/lock paths call the ancestry guard.
- **SYS-04:** even exact immutable reuse checks the existing directory budget;
  the retained oversized sibling fixture is rejected without deletion.
- Owner regression: **43 passed in 3.07 s**. Independent regression:
  **7 passed in 2.51 s**. Five independent cases validate desired behavior;
  two deliberately assert the remaining defect described below. Seven passes
  therefore must not be presented as all boundaries qualified.

## Remaining SYS-01 setup window

In `_supervise`, these steps remain outside its cleanup `try`:

```python
child = popen(..., start_new_session=True, ...)
owner = OwnedSession(child)
selector = selectors.DefaultSelector()
selector.register(child.stdout, selectors.EVENT_READ)
```

If either ownership registration or selector registration raises after Popen,
the function leaves its newly created worker alive. The worker may also retain
inherited loader/GPU lock descriptors. The new TERM/INT handler alone does not
repair this exception path.

Two CPU fault injections reproduce this precisely:

1. `OwnedSession.__init__` raises `OSError` after Popen returns.
2. `DefaultSelector.register` raises `OSError` after ownership was established.

In each case `supervise()` raises while its recorded newly created child is
still alive. The independent fixture then stops and waits for **only that
child**, using its pidfd. The simpler first reproduction is retained in
`SESSION_INITIALIZATION_FAULT.json`.

Required repair is local: put post-spawn ownership and selector setup under a
cleanup guard, retain the unreaped direct-child identity, and ensure a failed
setup either completes owned cleanup or explicitly retains a bounded owning
monitor. Preserve the pidfd/start-time/foreign-session checks; do not replace
them with unconditional numeric-group signalling. Add both setup faults to the
owner regression. No changes to model, scoring, native hashing or live ledger
are requested.

## Non-regression and actual admission boundary

The repair preserves source constants and original negative tests for:

- fixed dev plan/extract → train plan/extract → CPU fit ordering;
- one-shot consumed attempts, no automatic retry/resume, exact admitted GPU;
- two CPU slots, 64 GiB extraction RSS and 24 GiB allocated GPU observation;
- 900 s cooperative / 960 s outer extraction and 1200 / 1260 s CPU fit;
- 256 GiB per-role known logical read envelope, 2 GiB fit read envelope;
- 4 MiB raw log, 5 MiB stage control envelope including staged serialized JSON,
  counter and final-receipt reserves;
- exact runtime/helper pins and fresh checkpoint reserve requirements.

These remain observed limits rather than OS/cgroup hard limits. SIGKILL cannot
be handled by a Python cleanup handler. Successful CPU cases do not establish
actual extraction throughput, memory, or compatibility with the final runtime.

After the final narrow repair passes, Lead still must bind the immutable source
v2, regenerate metadata expectations with new pins, admit each actual fixed-role
plan, and approve its corresponding typed W storage admission before extraction.
The PENDING draft is not an authorization.

Future typed W migration must retain v7's **all seven existing claims** and the
new teacher-combination qualification's **≤64 MiB report reservation** within
the same **4 GiB administrative package / 1200 MiB effective budget**, as
reported by Lead. That report reservation is not a new ledger claim or tensor
cache. W tensors remain separately typed **≤1 GiB** within the same **768 GiB**
ledger; no older policy rollback is allowed. No live ledger/policy was read or
changed here, and no actual admission is issued.

Real W timing, GPU memory, native IO, and fit costs: **UNKNOWN / NOT_EXECUTED**.
Do not turn the previous roughly 1.86 TB repeated logical-read estimate into
new disk allocation or a measured IO result.

## Commands and evidence

Both suites were launched via a tiny Python parent setting affinity, RLIMIT_AS,
thread limits and an empty CUDA environment before `subprocess.run`:

```bash
/root/miniconda3/envs/ddp/bin/python -m pytest -q \
  reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1/test_preparation.py \
  reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1/test_lifetime_fix.py
/root/miniconda3/envs/ddp/bin/python -m pytest -q \
  reports/st_world_policy/20261017/spatial_probe_controller_review_v2/test_independent_lifetime.py
```

The full output is preserved in `OWNER_REGRESSION.txt` and
`INDEPENDENT_REGRESSION.txt`. The exact helper hashes and machine-readable
scope/verdict are recorded in `VERDICT.json`. No production script was edited.
