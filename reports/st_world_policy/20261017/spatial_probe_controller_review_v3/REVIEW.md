# Final narrow systems re-review

**SOURCE_CPU_PASS_BOUND_REAL_PLAN_PENDING**

Reviewed owner commit `7056a949b45b61712e40170f0a226245fd70d89b`.
The three helpers in Lead integration `16c6e902` have identical contents.
Exact SHA256 values are in `VERDICT.json`; these are the helper bytes to pin,
not a substitute for freezing/rebinding the final immutable runtime v2.

This closes the two setup-window counterexamples in independent review
`ef6f6bdcd16d839d7e8f22ef59db74853997a5f2`. The original FAIL evidence and the
earlier `b8e36e67` review remain intact. No production code was modified here.

The owner/selector initialization and registration steps after Popen now execute
inside the cleanup guard. The normal path retains its unreaped leader, verified
start ticks and pidfds. Failed setup uses only the just-created, still-unreaped
Popen child: `waitid(WNOWAIT)` establishes parent ownership, actual SID/PGID are
checked, and that anchored fresh group can be stopped before reaping. This
exceptional group signal does not acquire authority from an old receipt.
Same-session tails are checked twice around pidfd creation. The ten-second
setup cleanup bound and original failure are preserved; no retry is added.

## Actual CPU evidence

- Owner tests: **50 passed in 3.42 s**. Includes all prior limits/negative tests,
  ownership/selector constructor and registration failures, pidfd/proc setup
  failure, and already-exited leader with same-session tail in another group.
- Independent tests: **7 passed in 2.27 s**. Both prior setup failures now
  require the spawned child to have exited and been reaped. Independent nested
  session tests cover TERM, INT, early leader exit, inherited lock release,
  actual worker/grandchild disappearance, and an unrelated process group
  remaining alive. Foreign-group signalling is rejected. Ancestor symlinks and
  over-budget immutable reuse remain rejected.
- Both suites run under CPU affinity `{0,1}`, inherited `RLIMIT_AS=2 GiB`, empty
  CUDA visibility, and two-thread environment limits. Parent timeout was 90 s
  for owner tests and 60 s for independent tests. All signals targeted test
  processes created during these suites. No real task was stopped.

These independent tests now assert the corrected behavior; unlike the earlier
counterexample suites, their pass status means the scoped cleanup checks pass.

## Inherited controls retained

The final diff is confined to setup cleanup and owner tests/reports. The source
constants, admission rules and observer are unchanged from the prior review:

- fixed dev16 plan/extract → train64 plan/extract → CPU fit300;
- exactly one explicit plan per role, one global loader lock, no auto retry,
  auto resume, or GPU discovery;
- two CPU slots; extraction 64 GiB RSS / 24 GiB allocated-GPU observation;
- extraction 900 s cooperative / 960 s outer; fit 1200 / 1260 s and 2 GiB RSS;
- 256 GiB role logical read envelope, including plan hashing, and 2 GiB fit
  read envelope; known hash/load counters and `/proc/self/io` have their stated
  scope rather than a claim to meter all physical reads;
- 4 MiB raw log, 5 MiB stage envelope with bounded JSON/temp publication,
  counter and final-failure reserves;
- exact runtime/helper/input pins, fresh checkpoint commitment coverage,
  typed W ≤1 GiB in the same 768 GiB ledger.

SIGKILL of the supervisor itself cannot execute Python cleanup. The fixed worker
is not permitted to deliberately escape its session. RSS/GPU/wall monitors are
observed limits with reaction/cleanup grace, not OS cgroup hard limits. These
boundaries remain explicit and do not affect the scoped successful CPU verdict.

## Actual execution remains pending

No actual W plan/extraction/fit, weights/images/native checkpoint read, GPU,
live ledger access/mutation, dispatch, or actual efficiency measurement occurred.

Lead must next freeze the final runtime v2, regenerate its immutable metadata
bindings, and issue exact stage/resource admissions. Expected signatures and a
PENDING draft do not authorize real reads. Actual plans must verify inputs
before their signatures can support typed W claims. Storage migration must
preserve **all seven v7 claims**, the teacher-combination qualification's
**≤64 MiB report-only administrative reservation** within the same **4 GiB /
1200 MiB effective** package, and the same 768 GiB ledger. The report reservation
is not a new claim or tensor cache. Fresh checkpoint reserve and authorized
leased resources must be checked at execution; this review grants neither.

Actual IO, latency, GPU memory and fit performance are **UNKNOWN**. Prior
roughly 1.86 TB lifecycle arithmetic describes repeated logical reads, not new
disk allocation or measured storage traffic.

## Commands

The recorded commands run with the bounded parent described above:

```bash
/root/miniconda3/envs/ddp/bin/python -m pytest -q \
  reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1/test_preparation.py \
  reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1/test_lifetime_fix.py
/root/miniconda3/envs/ddp/bin/python -m pytest -q \
  reports/st_world_policy/20261017/spatial_probe_controller_review_v3/test_independent_lifetime.py
```

Full outputs accompany this report. No broader feature or model-method review
was reopened.
