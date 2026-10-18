# Independent review of the explicit rollout-scale implementation

**PASS for the affected CPU/Gloo implementation at `39e18192877fc9c30b8ea3635fd1f45be026798b`.** This is readiness for a separately registered, resource-approved bounded SMOKE/PROFILE, not approval of formal training or evidence of planning improvement. The reviewer changed no implementation and dispatched no real experiment.

## What was checked

The new helper preserves the original FP32 scene-centered advantages. One complete old-policy rollout, including zero advantages, supplies the FP64 two-pass sample variance (`correction=1`); `1e-10` is added after the square root. Only the denominator uses the global mean. The numerator is not globally re-centered. The FP32 detached output preserves exact zero groups. An all-zero rollout reports `gain=null`, while retaining the arithmetic denominator reciprocal separately. The original BC and CFM execution is retained.

Actual two-process CPU/Gloo tests cover unequal rank partitions, the 24-scene tail, empty-local helper shards, global/microbatch gradient equivalence, and ten single-rank bad-input cases. Error votes occur before the statistics/gather that would otherwise strand a healthy rank. This is not a claim that the original trainer supports an empty backward rank, or that a failed NCCL process can be recovered by this helper.

The trainer computes the scale once before its existing microbatch loop. Its numeric identity, feedback v2 payload and native optimizer boundary retain the method and the last consumed common receipt hash. The receipt is verified before RNG restoration. The existing terminal return precedes the training loop. A real tiny DiT/Adam CPU interrupted continuation compares loss, gradient, model state and RNG; it does not stand in for CUDA/ZeRO2 qualification.

A pre-seal writer defect was found during independent review: a receipt for another same-shape batch could pass insufficient binding checks. The final writer validates the actual feedback/local identity, reward/mask/output tensor hashes, common per-rank record, policy and score protocol. Negative tests reject swapped feedback, altered output and corrupt artifacts. No unresolved blocker remains in the reviewed affected code.

## Executed evidence

The exact command and environment are in `TEST_EXECUTION.json`. The fixed implementation was checked out separately and remained clean. **37 passed, 0 failed, 0 skipped**, pytest 6.50 seconds; enclosing wall time 7.88 seconds, child CPU 10.48 seconds, maximum child RSS 699408 KiB. A single installed pynvml deprecation warning is unrelated to these changes. The resource counters are per the subprocess accounting described in the receipt; they are not a measured aggregate resident set of every simultaneous process.

The first invocation accidentally inherited the repository's `navsim` Python 3.9. Collection failed on an existing Python 3.10 type annotation before any test ran. Its log and receipt are retained. The passing invocation uses the already documented `/root/miniconda3/envs/ddp/bin/python`; no source was changed to get the tests through.

Tests comprise 29 module cases, 4 integrated cases, and 4 independently authored legacy/zero/terminal checks. The source and test hashes are in `REVIEW.json`. The author's broader 97-test run is separate evidence, not counted again here.

## Reuse of the completed original C1 control

**Reuse is justified if the new comparison retains the original Base, data/order, seed, topology, optimization and evaluation definition except for the explicit new advantage normalization.** Independent AST checks establish that all original FPO arithmetic functions are unchanged from `6714b1e9`. Fixed FP32 loss/gradient values match bitwise. C0–C4 legacy numeric and target identities and validated config match; feedback-v1 semantic payload and identity match without RNG consumption. File serialization bytes are not claimed identical.

Only `identity.json`, `resolved_config.json` and `RESULT.json` of the historical run were read (11291 bytes). They report the original 4096-update C1 as complete. Its `scientific_result` is `NOT_EVALUATED`; this review does not turn that into a quality result. Exact bindings and the matched-control conditions are in `LEGACY_CONTROL_BINDING.json`.

This is reuse of an existing old-method control, not permission to resume it under a different source fingerprint, rename it as the normalized arm, or claim identical new and old execution identities. A duplicate full control is not needed merely because an explicit unused normalization branch was added.

## Remaining real-execution boundary

Real-image integration, native CUDA/NCCL/ZeRO2 save/load, actual scored updates, GPU safety and planning outcomes remain untested here. The next bounded run needs its own resource and registration binding. No new scientific gate is imposed by this review.

The normalization changes relative policy-to-BC weighting and cross-update weighting; it is not a lossless optimization. It cannot establish safer trajectories by algebra alone. Full-TRAIN support counts must not be interpreted as a full-TRAIN gradient decomposition, and one-reuse ratio clipping is not a post-step trust region.

## Reproducing the independent tests

At the fixed implementation checkout, first extract three original source files without changing the checkout:

```bash
legacy_dir=$(mktemp -d /tmp/iqe-legacy-review.XXXXXX)
git show 6714b1e9fb2e0c8059b90b8b4d48cd40796fe345:starVLA/model/modules/flow_posttraining/fpo_objective.py > "$legacy_dir/fpo_objective.py"
git show 6714b1e9fb2e0c8059b90b8b4d48cd40796fe345:tools/st_world_policy/train_flow_policy.py > "$legacy_dir/train_flow_policy.py"
git show 6714b1e9fb2e0c8059b90b8b4d48cd40796fe345:tools/st_world_policy/flow_training_config.py > "$legacy_dir/flow_training_config.py"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 LEGACY_REVIEW_SOURCE="$legacy_dir" PYTHONPATH="$PWD:$PWD/tests/st_world" \
 /root/miniconda3/envs/ddp/bin/python -m pytest -q \
 tests/st_world/test_advantage_scale.py \
 tests/st_world/test_flow_rollout_scale_integration.py \
 /mnt/project/VLA-Drive-st-diagnostics-final-correctness-20261018/reports/st_world_policy/20261018/rollout_scale_runtime_correctness_v1/test_independent_runtime.py
```

The independent test file lives in the reviewer checkout; do not attempt to import the reviewer's older implementation snapshot instead of the fixed runtime.
