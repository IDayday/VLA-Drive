# Independent review of the actual real4 calibration failure

Reviewer: `/root/st_calibration_independent_correctness_v2`.
Review base: `112ef4c40c0b0f2e82df367b2a4a64e9fdcf3d30`.
Actual failed runtime: `09ca42cbf03be30250a5736e6fd13951072fd2ca`.

**PASS_FAILURE_CLASSIFICATION_ONLY** means that the small recorded artifacts
support the classification below. The calibration itself remains
**FAILED_NO_AUTO_RETRY**, with no successful GPU loss/gradient measurement.
This is neither a mathematical calibration approval nor a retry authorization.

## Actual failure

The immutable attempt is
`/mnt/project/st-world-policy-20261016/audits/loss_scale_calibration_real4_v1`.
Its admitted and resolved configurations match, including the fixed four scene
IDs and microbatch size one. `result/FAILED.json` records the CUDA/CuBLAS
deterministic-algorithm error requiring `CUBLAS_WORKSPACE_CONFIG=:4096:8` or
`:16:8`. `result/MEASUREMENT.json` is absent.

The actual pinned controller does not set `CUBLAS_WORKSPACE_CONFIG` before
launching its child, and the actual CLI enables deterministic algorithms without
checking that workspace prerequisite. This supports a missing execution
prerequisite. The failure receipt has no traceback or exact operator location;
it does not establish precisely which model operation first failed.

The wrapper recorded workload exit code 2, while its monitor itself reported
no resource failure or monitor error. Its cleanup recorded zero remaining owned
processes. This distinction must be preserved: successful resource accounting
does not mean successful calibration.

| Recorded measure | Value |
|---|---:|
| Measurement elapsed | 407.814903626 s |
| Invocation wall time | 416.604673488 s |
| Supervised child time | 412.671718519 s |
| Peak aggregate RSS | 26,929,520,640 bytes |
| Charged parent/child logical reads | 51,404,923,553 bytes |
| CUDA allocated peak | 7,126,431,744 bytes |
| CUDA reserved peak | 7,325,351,936 bytes |
| Optimizer updates | 0 |
| Newly persisted tensor output | 0 bytes |

These are the controller's actual recorded observations, within its admitted
numeric caps. They are not independently resampled GPU measurements, hard
kernel-enforced limits, or a throughput result. Zero newly persisted tensor
bytes does not mean zero checkpoint/teacher reads or zero GPU computation.
The attempt did incur the costs above.

## State and scientific claims

The calibration implementation has no optimizer, uses `autograd.grad`, and has
RNG restoration and parameter/buffer/mode assertions in `finally` paths. The
failure receipt records zero optimizer steps, but records neither post-state
hashes nor a completed RNG equality check. Therefore:

- Actual post-failure parameter/buffer/RNG equality: **NOT_RECORDED**.
- Real loss/gradient calibration and coefficient proposal: **ABSENT**.
- Scientific coefficient selection or training application: **false**.
- Successful CPU source tests from previous reviews do not turn this failed
  actual run into a GPU PASS.
- The separate historical T1 `FAIL_LIVE_DESCENDANT_SCOPE` remains unchanged.
  This review does not overwrite or rehabilitate that earlier resource result.

## Permissible repair scope and retry boundary

A minimal repair may pin and validate a supported workspace configuration
before CUDA initialization. It must retain deterministic algorithms, the
registered model/loss reductions, precision, fixed scene/time/patch draws and
zero-update semantics. Relaxing determinism to avoid the error would change the
execution definition and is not the proposed repair.

Any retry needs its own immutable attempt/output identity and fresh exact-source
resource admission, after independent review of the patch. The failed directory
and cost records must remain intact. Lead decides whether to run that bounded
attempt; this report does not launch or authorize it. No generic controller or
new retry loop is introduced.

## Read-only verification

`audit_actual_failure.py` performed 19 checks against the small receipts and
pinned source text. It did not import PyTorch, load weights/images/teacher
payloads, access SSH/GPU, signal processes, or modify the campaign ledger.
All input hashes and source identities are in `FAILURE_REVIEW.json`; full output
is `FAILURE_AUDIT.log`.

Executed from the isolated review worktree:

```bash
ulimit -v 1048576
timeout 120s taskset -c 6,7 python \
  reports/st_world_policy/20261017/temporal_scale_real4_failure_review_v1/audit_actual_failure.py
```

Measured audit subprocess: 0.026659 s; peak RSS 14,835,712 bytes; logical-read
delta 146,389 bytes; explicit input bytes 146,292. These audit costs are separate
from the failed CUDA attempt. The script deliberately refuses to overwrite its
review receipt.
