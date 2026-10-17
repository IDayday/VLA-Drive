# Narrow independent review of the cuBLAS startup repair

**PASS_CODE_CPU_ONLY**, for source
`c1168f97efb051b1418b01004668426211d68d42` and proposed calibration config SHA256
`d5358c65585a4852bc959e5795a0f0ae647b9bdf6dd99c64220e361b38eed560`.
This is not a successful real calibration measurement or dispatch approval.

Reviewer: `/root/st_calibration_independent_correctness_v2`; implementation
author: `/root/st_temporal_scale_calibration`. The review worktree is isolated
from the author and actual runtime. The full reviewed source hashes, config
pins, checks and limitations are in `FIX_REVIEW.json`.

## Exact implementation scope

The child launcher now binds `CUBLAS_WORKSPACE_CONFIG=:4096:8` before starting
Python. It refuses a conflicting inherited value. CLI `measure` and direct
`execute` both check the exact configuration, actual environment and
uninitialized CUDA before payload loading; `measure` checks before its
admission/device inspection and result-directory creation. It never repairs an
already initialized CUDA context in-process. The equally supported `:16:8` is
rejected as a different declared execution setting, not alleged to be invalid
for PyTorch.

This fix does not disable deterministic algorithms, relax TF32, change precision
or change the actual training/loss computation. An independent AST test removes
only the new preflight call and runtime-receipt assignment and compares the rest
of `execute` with the failed runtime: it is identical. The underlying measurement,
model and loss source bytes are unchanged. No new optimizer is introduced.
Existing tests again cover original global reductions, autograd gradients,
shared draws, no model mutation and zero-signal rejection.

The new configuration preserves the same common S0 binding, initialization,
four scene IDs, microbatch, dropout/noise/time/horizon/patch seeds, target
identities and declared shared parameter subset. The training-config differences
are only the unused trainer output namespace and wall cap; the calibration
configuration additionally records the workspace environment and new output.

## Actual frozen-runtime inheritance

The prepared runtime
`/mnt/project/VLA-Drive-st-loss-scale-real4-v3-20261017` was read-only checked at
clean HEAD `f212c96cbb74af9b3c59d91e80241675dc55fec0`. Its calibration CLI,
measurement implementation, scoped launcher, ST model and loss source files are
byte-equal to the reviewed source. This permits inheriting this **code-only**
review for those exact hashes. It does not attest to future runtime contents or
to a real result that has not been inspected.

The pinned training configuration is SHA256
`716d1cc229e4bcab14431c29954ab1858e8bb81c6fa213c1e4c1155f345f9c95`.
Use the committed proposal's actual absolute config path recorded in the JSON;
do not silently rewrite config paths and continue using the old hash.

## Independent tests and measured review cost

**53 passed, 0 failed, 0 skipped**: 47 existing cases plus six independent cases.
The added tests check that preflight preserves all CPU RNG states and the
environment, that a fresh Python child sees the configured environment before
any PyTorch import, that computation AST and fixed-batch configuration stay
unchanged, that prepared-runtime hashes match, and that the old failure cannot
be treated as a state/RNG PASS. Full test output is `FIX_TESTS.log`.

```bash
cd /mnt/project/VLA-Drive-st-temporal-real4-fix-review-20261017
timeout 120s taskset -c 6,7 python \
  reports/st_world_policy/20261017/temporal_scale_real4_failure_review_v1/run_cpu_fix_review.py
```

The wrapper measured 4.667675 seconds, 658,841,600 bytes peak sampled child RSS,
and 139,020,567 wait-accounted logical read bytes. Pytest itself took 3.64 seconds.
The child reported CUDA uninitialized. CPU affinity was 6/7; limits were 120 s,
1 GiB RSS, 1 GiB reads and 16 MiB output. Limits are monitored by this small local
test wrapper; they are not a claim of a kernel-enforced memory cap. No GPU,
checkpoint/image/teacher payload, SSH, ledger mutation or live workload signal
was used. The wrapper can terminate only its own created CPU test child if its
budget is exceeded; that path was not triggered.

## Failed evidence and next-attempt boundary

The original v1 remains `FAILED_NO_AUTO_RETRY`, with no `MEASUREMENT.json` and
no recorded post-failure model/RNG equality. Its actual 416.604673488-second
invocation and 51,404,923,553 read bytes remain charged. Independent review
`901eb43dd46fa144f734734a040639c584f7b673` preserves those facts.

The proposed v2 is a distinct directory, with one invocation, 3000 s + 40 s
cleanup, 208 GiB reads, 96 GiB RSS, fixed four scenes and zero optimizer steps.
The aggregate arithmetic is:

- Time: `416.604673488 + 3000 + 40 = 3456.604673488 < 3640` seconds.
- Reads: `51,404,923,553 + 208 GiB = 274,743,222,945 < 274,877,906,944` bytes.

Only 134,683,999 bytes remain outside that new read cap under the original total;
the existing comprehensive parent/child accounting must be retained. A fresh
exact-source independent resource admission and Lead launch decision are still
required. This review performs no retry and does not clear historical T1's
`FAIL_LIVE_DESCENDANT_SCOPE`.

One documentation caveat was reported to the author: the v1 receipt identifies
a CuBLAS workspace error but does not locate the exact operator. The author's
phrase “first cuBLAS matmul” is not supported by the available traceback-free
receipt and is not endorsed by this review. This does not change the code-only
finding. A real v2 attempt may reveal other backend limitations; no CPU test here
claims actual CUDA/FA2 correctness, measured losses/gradients or scientific
coefficient validity.
