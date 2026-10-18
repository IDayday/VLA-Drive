# Epoch timestamp parity fix, 2026-10-18

The original v2 run remains unchanged at
`/mnt/project/st-world-policy-20261016/audits/consequence/road_direction_fixed32_v2`.
It reported 9 geometry parity failures among the fixed32. This patch does not
convert those failures to successful samples or claim a scientific gain.

## Reproduced implementation error

The proxy used `initial_time_us / 1e6`. Official nuPlan `TimePoint.time_s` uses
`initial_time_us * 1e-6` (unlike `TimeDuration.time_s`, which uses division).
Official `transform_trajectory` calls `_get_fixed_timesteps`, adding that value
to the eight future times before `int(t * 1e6)`. Official
`get_trajectory_as_array` similarly uses it for all41 interpolation queries.
At epoch-scale timestamps the two arithmetic expressions can differ by one
float64 ULP; truncation then changes a knot/query by one microsecond. The
initial actual EgoState timestamp remains the original integer in both paths.

Synthetic counterexample, no real scene/cache/score: `1630000000000003`us,
rear axle `(500000, 4200000, 0.37)`, existing deterministic trajectory fixture.
Division gives `1630000000.000003`, multiplication `1630000000.0000029`.
Before correction: maxXY error `1.864507794380188e-6`m; corner error
`1.8849968910217285e-6`m. After using the official multiplication: both zero.
The same occurred for timestamps ending `...0004` and `...0008`.
The existing timestamps did not cover this rounding boundary.

The correction is confined to integer-time-grid construction. Raw decode,
anchoring, rotation, unwrap, interpolation, footprint, union, cost, FD steps,
gradient cutoff and direction/safety budgets are unchanged. The absolute
geometry tolerance stays **1e-9**. This establishes a real implementation bug,
not yet the cause of all9 real failures.

## Failure evidence now retained

`command_geometry_audit` records initial rear axle/time, both exact integer
time grids, max and worst-index XY/periodic-yaw/corner errors, and all41
proxy/official poses and four corners. `GeometryParityError` carries that audit
before the unchanged gate raises. A new attempted run writes
`NUMERICAL_PARITY_FAILURE.json` and embeds it in its failed scene receipt.
No old receipt is rewritten. This helper executes no simulator or scorer.

## Bounded geometry-only localization

`tools/st_world_policy/diagnose_road_time_parity.py` prepares the exact9 original
failures from existing STARTED/SCENE_RESULT receipts. The original32 denominator
remains explicit. It never reads rollout tensors or checkpoints. A prepared
plan binds receipt/content/source hashes and does not authorize execution.

The Lead-only execute path requires a checksum-bound admission with status
`APPROVED_GEOMETRY_REPLAY_ONLY`, issuer `lead`, exact plan hash, host and window.
It reads only the9 original hash-matched caches with bounded LZMA expansion,
validates actual vehicle/schema, then compares old-division and corrected
multiplication paths against official geometry. The only patched function in
the old diagnostic is the time-grid helper. Both versions receive identical
physical commands, initial EgoState, vehicle and official interpolation.

Limits: 32MiB logical asset reads, 64MiB maximum expanded cache,120s wall,
8GiB address space/RSS,4MiB output, CPU threads1,0GPU,0score,0training. There
are no automatic retries or replacement scenes. A fresh output directory is
required; ATTEMPT_STARTED preserves all9 pending rows even on abrupt failure.
The program catches normal deadline/error termination and writes all9 statuses.
Use the Lead's existing external supervisor for native hard-kill containment.

**Actual real-cache localization status: NOT_RUN by implementation agent.**
Old complete/undefined scenes are not rescored by this script. Whether all9
failures are explained, and whether further roundoff exists, remains unknown.

## Verification

`TIME_PARITY_TESTS.txt`:63 passed in11.74s; wrapper12.62s; parent843132KiB,
child833012KiB peakRSS; CUDA uninitialized; real score calls0/real asset reads0.
Includes epoch rollover cases at three headings, periodic yaw, actual official
geometry with nonzero initial velocity/acceleration, failure-error retention,
no-admission refusal, and all existing geometry/runner fault tests.
`TIME_PARITY_CLI_HELP.txt` records actual executable help.

```bash
cd /mnt/project/VLA-Drive-st-consequence-road-proxy-20261018
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /root/miniconda3/envs/ddp/bin/python \
 tools/st_world_policy/diagnose_road_time_parity.py --help
```

After Lead integration, run `--prepare --plan <new-plan.json> --output-root
<fresh-geometry-only-output>` on that frozen worktree. After exact independent
delta review and actual Lead admission, use `--execute --plan <same-plan.json>
--admission <lead-admission.json> --admission-sha256 <actual-sha256>`.
No admission or real localization result is fabricated in this report.
