# Thin real runner delta (not executed)

`tools/st_world_policy/run_road_direction.py` now supplies the actual CPU path;
the earlier geometry source and its review identity are unchanged. This is not
an additional trainer, reward, model, or automatic research controller.

## Admission boundary

`--prepare` writes a fixed registration with exact source hashes, git commit,
32-scene selection, one CPU worker/one numerical thread,1800s wall,8GiB aggregate
RSS/address-space cap,1GiB output,256MiB logical asset reads,32MiB/s read throttle,
64MiB maximum decompressed cache and128 durable backend invocations. Existing
original scores are reused; normally only96 new invocations are needed. No
retry is automatic. The32 selected scenes and all unresolved/undefined states
remain in the result denominator.

`--execute` requires the registration plus explicit Lead admission path/hash.
That admission binds exact source/registration, permitted actual host and window,
current dispatch-policy checksum, and two distinct independent reviews. Each
review explicitly uses `APPROVE_ROAD_DIRECTION_EXECUTION`, its role, registration
checksum and source hashes. A prepare-only PASS is deliberately insufficient.
The actual Lead creates these artifacts only after the independent reviews. This
implementation neither generates nor fakes them.

The parent launches one spawned CPU worker, observes aggregate parent+child RSS,
output size and wall time; it requests termination20s before the1800s bound,
then kills only its child if necessary. The worker has its own alarm/address-space
limit and checks budgets before file IO, geometry and canonical calls. The parent
writes an attempt summary even after native child failure/timeout. There is no
CLI worker mode bypassing the supervisor.

## Predeclared decoder discrepancy gate

Before geometry or scoring, original raw must be the actual saved FP32 tensor.
The CPU derivative uses its exact float64 widening. The XY origin correction is
bounded component-wise by original FP32 constant-rounding errors plus half the
maximum adjacent FP32 spacing of product and sum, and a conservative float64
roundoff term. This covers the original eager multiply/add rounding; the bound
is recorded at every point. Nonfinite spacing/overflow refuses execution.

For atan2, periodic difference is bounded by **8×FP32 epsilon =
9.5367431640625e-7 radians**. This is explicitly a predeclared numerical admission
tolerance, **not** a theorem about CUDA atan2 error. CPU FP32 and CPU FP64 decodes,
original raw hash and exact observed corrections are recorded. No tolerance is
chosen after reading actual trajectories. These conditions follow the independent
correctness reviewer's pre-read recommendation; runner approval is still pending.

## Actual execution wiring

1. Verify all four original boundary4095 rollout receipts/payload checksums,
   ten Euler steps, exact selected raw/observation/context identities, complete
   original official request/result receipts and fixed TRAIN-index membership.
   The176MiB enriched manifest and model checkpoints are not read.
2. Charge compressed reads against the new fixed budget. Bound LZMA output
   before unpickling and reject trailing data. Validate actual cache schema,
   maps, exact simulator/scorer vehicle dimensions and original official
   interpolation/footprint parity for every scene (predeclared1e-9 absolute
   pose/corner tolerance, periodic yaw). Bind the verified file stat to reject
   mutation between checked read and reuse.
3. Reuse the original canonical `ConsequenceEngine.score`. Its native simulator,
   reference normalization and scorer are unchanged. Only a preverified bounded
   context is made available; the native loader still verifies the immutable
   file stamp before each use. No unbounded fallback deserialize is allowed.
4. Persist all four raw/physical variants BEFORE scoring. Each new invocation
   receives a durable fsynced immutable intent before reaching the backend.
   Failures preserve payload, identity and canonical status, never become zero.
5. Persist official metrics, reference/executed arrays, custom command and
   executed road cost, geometry/parity, tracking and kinematic diagnostics.
   Paired negative-minus-positive/random contrasts are labelled valid-subset
   only; full requested32 denominator remains visible. Zero/high-to-zero and
   new NC/DAC/TTC failures are reported even if margin improves.
6. `--resume` requires a new explicit admitted attempt. Complete or failed scenes
   are not repeated. A scene interrupted after STARTED is labelled uncertain and
   not replayed, because a native call may have completed without its receipt.
   Only never-started scenes can continue; charged intents are global to the run.

## Verification actually performed

`RUNNER_TESTS_FINAL.txt`: **43 passed** (30 geometry +13 runner),7.628s wall,
824008KiB peakRSS, CUDA uninitialized. Fresh real asset reads0; real official
calls0. Checks cover the decoder gate/periodic seam, bounded decompression,
hash/read limits, immutable outputs, exact approval and stale-source rejection,
prepare-only refusal for real execution,32-row partial denominator, cache
TOCTOU refusal and explicit undefined stationary curvature. Preparation and
`--help` are real code paths; no test fixture is presented as real scoring.

Native-process fault injection and real context/runtime throughput remain part
of the independent runner/resource review and later admitted diagnostic. The
30-minute cap is a hard limit, not a measured promise that all32 scenes finish;
per-point polygon tests remain a possible CPU bottleneck. No tolerances, scene
selection, epsilon or scientific definition may be relaxed to force completion.

## Exact CLI

```bash
cd /mnt/project/VLA-Drive-st-consequence-road-proxy-20261018
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /root/miniconda3/envs/ddp/bin/python tools/st_world_policy/run_road_direction.py --help
```

The Lead must run `--prepare --registration <new registration JSON> --output-root
<admitted new diagnostic directory>` on the final frozen execution worktree.
Then `--execute --registration <that file> --admission <actual Lead receipt>
--admission-sha256 <actual checksum>` performs the diagnostic. These actual
admission paths do not yet exist, so this document does not invent executable
permission or a dispatched command. Optional `--resume` has the no-retry behavior
above. No real run was started by this implementation task.
