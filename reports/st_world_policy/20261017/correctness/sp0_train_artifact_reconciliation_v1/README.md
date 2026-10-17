# Independent SP0 TRAIN64 committed W artifact review

**PASS for the exact 64 committed W artifacts. The original launch remains STOPPED_NO_AUTO_RETRY.** This review does not grant a fit launch, extraction retry, resource admission, or scientific success.

The real extraction `RESULT.json` has SHA256 `4d24e4ed9a97744fe0d977c21284cb5fd64ec14c1348d71f6edadeff61245b27`. It reports rc0 and CLI COMPLETE64/64, but the supervisor reason is `OWNED_DESCENDANTS_AFTER_LEADER_EXIT`. These are separate facts. Its cleanup record reports six verified owned processes and zero remaining; this review's one read-only session scan found no matching old-session process. No process was signalled and no original receipt was changed.

## Exact checked artifacts

- Validator: unchanged `581399eb309677c20fe7464791b4f18685e947a6` runtime at `/mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017`.
- `spatial_probe.py` SHA: `e5d9e4306aae8311aefea9c8cc8ee777fd88424b412abcc6003128c6c7e0a20f`.
- Cache root: `/mnt/project/st-world-policy-20261016/caches/learned_w_probe_SP0_5000_v1/train`.
- Cache identity: `24741e0c9f1f8076158322bd92d7028142518fdb532cb43d2bc674d5dd11dcf3`.
- Signature SHA: `7c1f21f83295e50d22791d8664fb1f67fd3e2c5272d66ddbff64dbda56780b29`, identical to the original TRAIN plan signature.
- COMPLETE SHA: `17a8e9dd09447a2f2166a1dc8b216ada72e87b3525cfb3228d940dbd64a3d350`.
- Checkpoint identity remains `c87c59804b6237205e4b4a7a7e355b569c7cef5eaa499023f699111ec36d0beb`, tag `step_005000_1792217229597178910`. Only its identity metadata was checked; the checkpoint was not opened.

The previous independently used DEV16 verification script was adapted narrowly for TRAIN64, its exact failed result, signature, 64-scene manifest and one-attempt accounting. The validator called the original `identity`, `validate_binding`, `checked_rows`, and **one `read_world` call for each of all 64 real records**. It checked complete payload hashes, exact tensor keys, receipt/token/log/cache identity, W shape `[144, 2048]`, FP32 dtype, no gradient, finite values, boolean `[144]` masks and every mask bit true. It checked all 18,874,368 feature elements and 9,216 mask elements. The cache directory contains exactly the expected identity, completion, lock and 64 receipt/payload pairs, with no symlink, nonregular record or hardlink.

The original selected-scene order, no-target input contract and runtime/checkpoint binding agree with the original plan. COMPLETE, CLI_RESULT and embedded CLI result are equal. All pinned small JSON inputs were rehashed unchanged at the end. Each record in `ARTIFACT_RECONCILIATION.json` includes its original receipt SHA, verified payload SHA and file version (device/inode/size/mtime/ctime/link count), so a subsequent metadata-only reconciliation can refuse changed files without reloading tensors. The fit's own original `read_world` validation must still run when it consumes features.

This proves current artifact integrity and provenance commitments, not a fresh model recomputation or independent numerical equality to online W extraction. No checkpoint, image, raw log, GT, or teacher bank was read.

## Actual bounded audit cost

One execution only, exit0; CPU6/7, CUDA hidden and never initialized. Startup: 3.129652 s; artifact verification after import: 0.799752 s. Peak RSS: 631,943,168 bytes. External timeout: 120 s plus at most 20 s own cleanup; no timeout occurred.

The preflight stat census measured 75,516,928 payload bytes (64 files, each 1,179,952 bytes). Conservative hash-plus-mmap accounting is 151,033,856 bytes. Audit process `rchar` was 211,670,497 bytes, including 135,467,798 startup bytes. Charging the maximum of process rchar and startup plus nominal hash/mmap plus doubled small JSON and source allowance gives **286,986,706 bytes**, below the authorized 536,870,912-byte limit. This is logical/nominal accounting, not physical disk IO. Preliminary targeted source/receipt reads were below 1 MiB and are separate from that measured invocation; including that conservative allowance still leaves the audit below 512 MiB.

The original extraction charged 153,889,914,639 bytes under its existing `max(hash + nominal load, rchar)` rule. Together with its plan's 37,575,105,055 bytes, 83,412,887,250 bytes remained from that role's 256-GiB budget before this review. The JSON's remaining-after-review field subtracts the measured audit charge; it is an upper bound before preliminary small reads. These figures preserve the original counter scope, not a fresh native-load measurement.

## Minimal next integration

The existing DEV16 reconciliation cannot be reused directly. It pins another failed result/review, exactly 16 records, a DEV cache and the consumer `train_plan`. A new explicit TRAIN64 receipt must bind this review, the current stopped result and all current file versions; preserve `ARTIFACTS_ACCEPTED_SUPERVISOR_FAILED`; and make only the original CPU `fit` stage eligible.

The unchanged supervisor currently permits `previous_artifact_reconciliation` only for `train_plan`, while `fit` requires prior TRAIN extraction COMPLETE. A thin separately reviewed versioned adapter is needed for the exact TRAIN64-to-fit transition; do not relabel the stopped result, weaken the generic monitor, reuse DEV pins, or rerun extraction. The fit remains the original registered CPU diagnostic with fresh resource admission and its original feature validation.

`launch_approved=false`, `downstream_transition_approved=false`, `automatic_retry=false`, and `scientific_result=false` remain explicit in the independent evidence. There is no GPU allocation or model training authorization in this report.
