# Independent review of the actual road-direction runner

**APPROVE_ROAD_DIRECTION_EXECUTION**, correctness scope only, for the exact
registration SHA256 `9bdcced47aa522eeefc8900854a9fd2ee5c75c922ec09fdd635e3b9ce35730d8`
and frozen integrated source `a7d0a01583027dca3b04a84b5ab51d3fb2989984`.
The actual diagnostic remains **NOT_RUN** at review time. This is not a formal
training or scientific-effect approval; separate resource review and Lead
admission are required.

The original geometry module is unchanged from the independent review of
`ec9fbb32`. This review covers the thin real runner delta `55721af7`, its 13
tests, actual call chain and five added independent fault/identity tests.
All **18 runner tests passed**, no skips; one dependency deprecation warning.
Measured CPU 4.347 seconds, wrapper wall 4.663 seconds, peak RSS 695,734,272 bytes.
No model, real rollout payload, metric cache, GPU or official scoring call was
used. The 412-byte original COMPLETE receipt was read to confirm the native
receipt schema; its tensor payload was not read.

The previously required anchoring guard is now present before geometry/scoring:
actual saved FP32 raw; explicit XY constant-rounding and multiply/add half-ULP
envelope; fixed periodic yaw limit `8*float32_eps`. The latter is an admission
tolerance, not a GPU math-library error theorem. Offset evidence is persisted;
out-of-envelope inputs fail without scene replacement. Existing saved raw and
physical values are not modified. Actual offsets remain unmeasured.

The loader joins exact rollout receipt/payload, policy, scene, raw, observation,
condition, current TRAIN membership, request and result identities. The added
fixture exercises all 32 selected positions across four rank payloads and proves
that a changed condition is rejected. This uses synthetic in-memory payloads and
mocks the already-reviewed native reference verifier; it is not real integration
evidence. Native `validated_reference_input` is called in production, unchanged.

Before real scoring, each bounded cache is checked against its hash and file
version, actual schema, simulator/scorer vehicle dimensions and original native
interpolation/four-corner conversion. A private canonical engine consumes only
the verified context. Each new call is charged through an immutable persisted
intent first; existing originals retain their exact request and score. Known
scoring failures remain failures. The worker retains terminal failures, and an
interrupted STARTED scene is never retried automatically. Five independent tests
also cover these error and non-mutation paths without running a score backend.

The approval retains the fixed limits: 32 TRAIN scenes, four variants, maximum
128 official invocations including any admitted retry, one CPU worker/thread,
1800 seconds, 8GiB combined RSS cap, 256MiB logical asset reads, 64MiB decompressed
per-cache cap, 32MiB/s read throttle and 1GiB output. The resource reviewer owns
native-process timeout and capacity evidence; correctness approval alone does
not dispatch anything.

Real numerical/parity failures, undefined directions and incomplete scoring must
remain in the full requested denominator. The reported paired contrasts and
risks explicitly identify their common valid subset, with individual scene
records retained. Do not interpret all-DAC1 originals as a road-repair cohort.
No policy, reward, model, deployment or feature-side work is changed.

Reproduce the bounded runner-only tests:

```bash
cd /mnt/project/VLA-Drive-st-road-runner-correctness-20261018
/root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261018/road_direction_correctness_v1/run_runner_review.py
```

The authoritative machine-readable approval is `RUNNER_REVIEW.json`, which binds
all six registered source hashes and the exact integrated commit. This report
does not supersede the registration or permit automatic budget enlargement.
