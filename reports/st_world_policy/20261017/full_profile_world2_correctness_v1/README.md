# Independent world2 PROFILE controller review

**PASS_NARROW_DIFFERENTIAL_CORRECTNESS_ONLY** for owner commit
`a6cd73920816816cbaa13544cf898002445b9981`, controller SHA256
`bd802d47f187e494c2dde72ab8c04c670cb529d2b32b187ca06dfad3f3cc6bd1`.
The runtime remains `a8b0036f33554fd2a00ea1af8c9b5f20fb6b965a`.
No implementation change, GPU query/launch, real checkpoint/database/model read
or ledger mutation occurred in this review.

20 narrow CPU tests passed independently: all 14 author fixtures were executed
again, plus six additional tests. Full stdout/stderr and measured cost are in
`TEST.log`; `verify_control.py` contains the independent fixtures. These tests
do not replace the separate resource review or a real world2 run.

The following affected behavior was verified:

- Recipe requires world2, micro1, global2, sample population 101592, cumulative
  stop64, original max_updates20000 and warmup1000. Population receipts must
  declare 64 total updates / 128 total exposures, including resume. No command
  supplies a replacement short scheduler or extends the step budget.
- The original frozen trainer restores native model/optimizer/scheduler/client
  state strictly. Its loop compares restored `state.completed` with stop_after,
  and learning_rate uses completed/max_updates, not per-invocation progress.
  This was a narrow source check, not a repeated resume qualification.
- `registered_config` is a deep copy changing only `runtime.output_root` into
  this run's independent namespace. Nested teacher/data/optimizer fields stay
  unchanged and cannot alias the parent. The registration explicitly sets
  `weights_usable_for_formal_init=false`.
- Both physical GPU identities and both locks are used. A fixture exercising
  the real `execute` function verified the two lock descriptors passed to the
  mocked launcher. No actual GPU lock or process was used.
- The real execute branch rejects a completed64 resume before launcher or
  monitor invocation. All 1–63 interior boundaries with twice-as-many scene
  exposures are accepted by the boundary predicate; mismatches are rejected.
  A partial extra checkpoint directory counts against the slot budget.
- Original native save already reuses a just-saved periodic boundary on the
  immediately following pause; this avoids a duplicate final64 slot. This
  existing source path was confirmed, not modified or numerically retested.
- Post-leader-exit rediscovery distinguishes an exit between snapshots from a
  genuinely remaining descendant. The latter still records failure.
- A monitored RSS failure remains FAIL even when the mocked native process
  returns0 and its result says completed64/exposure128. The real execute tail
  preserves monitor/exit evidence, then raises. A wrong exposure count also
  raises rather than claiming completion.
- Cleanup uses the same absolute invocation end and at most the existing180s
  remaining window. Tests verified a shorter absolute end wins, cleanup is
  attempted once, and resource/ownership failures are retained. Real OS cleanup
  and resource performance are outside this correctness-only review.

The initial CPU plan mentioned eight cores, while this exact implementation
accepts two or four. Lead explicitly corrected planned admission to CPU0–3
(four cores, two library threads); no source modification was needed. An
eight-core receipt remains rejected by this source. Test affinity was CPU6/7.

The read-only default uses current host/source/input checks and does not query
GPUs or create an output. The actual controller's metadata checks and the
native original provider/registration validators remain enabled. We did not
read the full scene manifest or repeat the original31 provider qualifications.
Actual input pin values were taken from the committed INPUTS.json, not claimed
as freshly revalidated live dependencies in this review.

Command executed:

```bash
taskset -c 6,7 timeout --signal=TERM --kill-after=5s 120s \
  /root/miniconda3/envs/ddp/bin/python -B \
  reports/st_world_policy/20261017/full_profile_world2_correctness_v1/verify_control.py \
  > reports/st_world_policy/20261017/full_profile_world2_correctness_v1/TEST.log 2>&1
```

Measured fixture cost: 0.0714s, 26,316,800 bytes peak RSS, 1,768,350 logical read
bytes and 2,698 logical written bytes. No Torch import. A1GiB address-space
limit and110s in-process timer were applied. Separate known-source inspection
and repository administration are outside those test-process counters; total
source/control reads and report outputs remain below the64MiB scope.

Next actual evidence must establish world2 startup, real full-population input
and native state integrity, finite updates, actual128 exposures if64 completes,
teacher service rate, peak resource/checkpoint cost and zero live owned
processes. This review grants none of those actual claims or formal training
permission. The independent resource admission and Lead's live checks remain
separate; prior failed evidence is not overwritten.
