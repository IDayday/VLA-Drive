# Independent systems review: SP0 learned-W five-stage controller

Verdict: **FAIL — do not admit a real stage with candidate a785985.**

Reviewed candidate `a785985dc7bd780e97b029219caec2587ae8b299`, parent
`7c2bfe07a11f1ddbfd62be60a6aac579cd4f18e6`. This review concerns only the
three controller/helper scripts under `spatial_probe_actual_preparation_v1/`.
It does not reopen the model, target, split, or learned-W protocol review.

The owner's existing 33 CPU tests pass here in 1.01 s. Four independent tiny
CPU reproductions also pass; **their assertions demonstrate defects**, not
qualification. The two process-lifetime defects below block dispatch. The two
control-path observations are separate defensive limitations.

No native weights, images, real W cache, live ledger, GPU, real plan/extract/fit,
or real task process was read, modified, dispatched, or signalled. Only the
test-created CPU processes were signalled through pidfds. A first fixture run
found that this Python build omits optional pidfd wrappers; the preserved log
records two fixture API failures. The fixture now invokes Linux pidfd syscalls
and waits for its own signalled children to exit. No production script changed.

## Blocking findings

### SYS-01 — supervisor termination does not terminate its own worker

Location: `supervise_once.py:134` (`supervise`), `:172` (limit-triggered signal),
`:182` (Python exception handler), and `:125` (`own_group_signal`).

`Popen(..., start_new_session=True)` deliberately places the worker in a new
session. The supervisor has no SIGTERM/SIGINT forwarding/cleanup handler.
Default SIGTERM terminates the supervisor without entering `except BaseException`.
Its independently sessioned worker remains alive. A shell or outer deadline
terminating the supervisor therefore does not enforce the stage's resource
deadline; inherited locks can remain held by the worker.

Minimal reproduction: run `supervise` around a newly created 15-second CPU
sleep; wait for its `CHILD.json`; send SIGTERM only to the new supervisor using
its pidfd; observe supervisor exit `-SIGTERM` while its exact recorded child
remains alive. The fixture then terminates only that child via its pidfd.

Required thin fix: route TERM/INT into a bounded cleanup of this spawn's verified
session/children, preserve the consumed attempt and failure diagnostics, and
release inherited locks only after those owned processes have terminated. Do
not signal an arbitrary PID or group obtained from an old receipt. SIGKILL is
not catchable; document that boundary and avoid claiming an OS hard limit.

### SYS-02 — an exited leader hides a still-running owned descendant

Location: `supervise_once.py:127` (`child.poll() is None` signal guard),
`:150` (loop exit), `:162` (RSS monitored only while leader runs), and `:177`
(pipe timeout branch).

Minimal reproduction: the new worker forks once and exits successfully; its
descendant closes stdout/stderr, retains the inherited global loader lock, and
sleeps for 15 seconds. `supervise` returns `exit_code=0, limit_reason=None` while
the descendant remains in the original worker's session and holds the lock.
A separate attempt to take the same fixture lock fails. The fixture terminates
only that descendant via pidfd and waits for exit.

When a descendant instead retains the pipe, the current `wall + 40` branch can
break without stopping it because `own_group_signal` ignores a dead leader.
The normal leader-exit reproduction is sufficient to establish the defect.

Required thin fix: retain verified spawn identities and safely account for the
owned session after the leader exits. An exited/reaped leader cannot be the
only ownership test. Before success or release, verify that no owned running
descendant retains the lease/locks. Handle PID reuse and foreign groups
explicitly; do not fix this by unconditionally signalling a recycled numeric
PGID. Add both normal-leader-exit and interrupted-supervisor regressions.

## Additional bounded-control limitations

- **SYS-03:** `prepare_bindings.py:71–75,95–106` rejects a symlink leaf/directory,
  but a symlink ancestor is followed. The fixture writes
  `alias/stage/RESULT.json` through `alias -> actual`; the file appears under
  `actual/stage`. Actual admissions must verify the complete output/lock path
  ancestry and trusted namespace. This review did not inspect or mutate live
  paths. This is not evidence that the actual fixed campaign root is a symlink.
- **SYS-04:** `prepare_bindings.py:107–110` reuses an exact immutable control
  without checking the directory envelope. A fixture with an existing valid
  signature and a sparse sibling exceeding 5 MiB returns the original pin.
  No additional large bytes are written, so this is a missing reuse-time audit,
  not demonstrated disk expansion. Fresh stage directories are one-shot;
  final real admission still needs an inspected clean bounded namespace.

The new-write paths correctly measure serialized JSON (including escaping),
conservatively reserve temporary publication space, bound raw logs, and retain
a failure-receipt reserve in the tested normal namespace. These mechanisms
should be retained in the narrow repair.

## Other reviewed controls

| Requirement | Code/control result | Actual execution status |
|---|---|---|
| dev16 plan/extract, then train64 plan/extract, then CPU fit | Fixed five stages and prior-stage COMPLETE dependency present | NOT_EXECUTED |
| One explicit plan per role, no auto retry/resume | Fixed CLI call; durable `mkdir(exist_ok=False)` attempt; no resume passed | PASS_SOURCE; extract internally revalidates its plan and hashes |
| Whole-campaign one-loader | Shared fixed Lustre/global-flock check; inherited descriptor | FAIL_LIFETIME until SYS-01/02 fixed |
| Two CPU affinity slots | Exact two current available CPUs; taskset plus thread limits | PASS_CPU_FIXTURE |
| Extraction 64 GiB RSS / 24 GiB GPU | Observed RSS and allocated GPU peak; no claim of cgroup hard cap | SOURCE_ONLY; no real GPU/RSS measurement |
| 900 s cooperative / 960 s extraction outer | Binding and outer constants correct | FAIL_OUTER_LIFETIME until SYS-01/02 fixed |
| Role 256 GiB logical read envelope | Plan used amount deducted from extract; file hashes and nominal torch.load pre-admitted; `/proc/self/io` separately observed | PASS_SOURCE_SCOPE; mmap/decode not individually metered; not physical-byte/bandwidth measurement |
| CPU fit 300 exposure cycles / 2 GiB RSS / 1260 s outer | Original fit API/registration, 1200 s cooperative | SOURCE_ONLY; empty cycles remain exposure cycles, not claimed optimizer updates |
| Fresh checkpoint commitments | Pinned CURRENT receipt required through outer wall + 30 s grace | PASS_NEGATIVE_FIXTURE; no live receipt admitted |
| W typed package ≤1 GiB, same 768 GiB ledger | Exact typed claim/signature and migration checks; original StorageBudget still required | WAITING_ACTUAL_PLAN_AND_MIGRATION |
| Final runtime, helpers, controls pinned | Exact reviewed hashes and clean runtime required | WAITING_FINAL_RUNTIME_V2 |
| No new GPU discovery / self-approval | One explicit admitted UUID/index; PENDING draft fails; no allocation/search | PASS_SOURCE; Lead remains sole dispatch authority |

Read accounting does not sum independent physical streams: the controller uses
the maximum of known hash/load logical bytes and process `rchar`, with observed
`read_bytes` reported separately. mmap reads and CUDA allocator reservations
have explicit scope limitations. Keep those qualifications in any real result.
The earlier roughly **1.86 TB** four-arm lifecycle estimate is repeated logical
checkpoint/input reading, **not new cache disk usage**. It is not a timing result.

## Remaining real admission conditions, after narrow repair

1. Independent CPU re-review passes the two process cleanup regressions. Lead
   freezes one clean final runtime v2 containing the dedicated 19 MiB run-control
   reader fix (`8fdc8be5`, review `6e12fc2a`) and repaired helpers. Regenerate all
   runtime-dependent bindings/pins in a new immutable control directory. The
   old cf580501 expectations and PENDING draft are not authorization.
2. Lead independently issues each exact stage admission, binding final helper
   hashes, preparation SHA, host, two CPU IDs, lease expiry, and extraction GPU
   UUID. No free GPU search or automatic next-stage dispatch.
3. Admit exactly one actual dev plan; only its verified signature may enter a
   typed W claim. After successful dev extraction, repeat for the fixed train
   role. Expected signatures are not successful plan receipts.
4. Typed W migration extends the **v7** policy and preserves all **7** claims,
   including the two newly completed metadata indexes. Latest Lead-reported
   ledger SHA is `7fcfc0c18fa69fc65a125ee36acc1204f9976e62926bede016269136f79cebbe`;
   this review did not reread or authorize that live ledger. Source policy:
   `/mnt/project/st-world-policy-20261016/resources/CACHE_STORAGE_BUDGET_POLICY_v7_index_followup.json`.
   Use the same state directory and 768 GiB cap; W tensor allocation stays
   separately typed, ≤1 GiB. Do not turn W tensors into metadata claims or
   replace the ledger with a new budget namespace.
5. Fresh checkpoint reserve, migration receipt, storage capacity, authorized
   resource lease, clean bounded stage namespace and mount/flock checks must
   all hold at actual extraction time. A completed earlier migration does not
   make a later stale reserve valid.
6. Report actual plan/load/hash/extract costs, allocated/reserved memory,
   read scopes and fit completion only after execution. CPU fit stays on CPU;
   it does not retain a GPU lease. All real times and efficiency are UNKNOWN.

No actual execution PASS is issued by this review. The owner is to repair only
the controller lifecycle; no model/native hash/storage framework rewrite is
requested. Lead decides any subsequent run.

## Reproduction commands

Run from this isolated review worktree; both commands operate only on small
CPU fixtures and metadata:

```bash
timeout 90s /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1/test_preparation.py
timeout 45s /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  reports/st_world_policy/20261017/spatial_probe_controller_review_v1/test_system_boundaries.py
```

The second command is an observed-counterexample suite: its four passing tests
mean the candidate defects remain reproducible. After a fix, replace these
counterexample assertions with the required bounded cleanup/rejection behavior
and retain the old evidence. Full local results accompany this report.
