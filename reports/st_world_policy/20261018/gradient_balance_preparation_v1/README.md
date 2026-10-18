# C1 policy/BC gradient mechanism diagnostic — preparation

Status: **IMPLEMENTED / CPU FIXTURE VERIFIED / REAL EXECUTION NOT_RUN**.
No model, original image, GT, metric cache, optimizer state or rollout tensor
payload has been loaded during preparation. No GPU or official scoring call.
The paused feature-side research is untouched.

## Question and fixed selection

C1 completed its registered 4,096-update screening, but the prior logs contain
joint gradients rather than separate BC/policy gradients. A centered policy
scalar can be zero while its gradient is nonzero. This diagnostic asks whether
the real policy gradient is small relative to BC, aligned, or opposed; it does
not infer this from the scalar loss and does not change the policy weight.

Use C1's complete native `step_003584_1792287137286076754`, **before** the saved
`boundary_003584` update. Its four rank receipts identify the same old-policy
hash. This was chosen because the native checkpoint and the following rollout
are both available, not because a particular gradient or development result is
favorable. It is not valid to pair the final 4,096 weights with boundary 4,095
rollouts. One boundary remains a local diagnostic, not a training-population
estimate. Next sample IDs will be recovered from the actual manifest and saved
epoch 1 / offset 409, then checked against each rank's rollout.

`PREPARATION.json` pins the real small receipts. Payload validity is not claimed
until the execution reader hashes and loads them.

## Reused methods and adaptation

Existing evidence: `reports/st_world_policy/20261016/evidence/`:
`DESIGN_EVIDENCE_CARDS.json::FPO_GRIDWORLD_REFERENCE`,
`SOURCE_CODE_MANIFEST.json`, `FPO_REFERENCE_CPU.json`.
Pinned official source: `akanazawa/fpo@418c2554f7cd22d52e14c07d951280929d73bf2f`,
`gridworld/models/diffusion_policy.py::compute_cfm_loss` and
`gridworld/models/fpo.py::learn`; paper *Flow Matching Policy Gradients*,
CFM surrogate ratio and clipped objective. Existing navigation adaptation uses
coordinate **mean**, uniform MC50, shared old/current CFM noise and time,
within-scene centered actual PDMS and eval-with-grad policy mode. It is not an
exact likelihood ratio. The current experiment's definitions are unchanged.

Reuse mature `audit_flow_gradient.py` native BF16 reader, original S0 loader,
current-only inputs, exact rollout/CFM checks, and completed official ledger
reader. Reuse `DiTFlowPolicy.bc_anchor`, the original repeat8/Beta(1.5,1)/s=.999
BC and its normal train/dropout behavior. No new gradient estimator or loss is
introduced. The extension is four-rank ownership plus separate gradient
measurement, not a new trainer.

## Measurement definition

- Reconstruct the original frozen BF16 training encoder and native BF16 action
  execution values; do not use FP32 standalone inference values for this test.
- Encode 4 original rank batches of 8, in microbatches of 4. Require exact saved
  frozen-condition hashes, four-candidate 10-Euler endpoints, old MC50 and current
  MC50. A mismatch rejects replay; no tolerance is relaxed.
- Reconstruct original physical requests on the GPU and read the **one global,
  rank-major 128-request ledger key**. No scan of all 4,096 request directories,
  no new simulator/service, no metric-cache loading or retry.
- Restore each rank's checkpoint Python/NumPy/CPU/CUDA state and all five task
  generator states. Original BC noise/time and dropout consume their streams;
  policy evaluation/recomputation must not consume that dropout stream. Restore
  the diagnostic caller's RNG afterward, including failure paths. Stored rollout/MC
  tensors are reused rather than advancing their generator streams; this does
  not claim the complete post-update RNG equals the original training state.
- Each microbatch loss is its contribution with denominator **32 original
  scenes / 128 real candidates**, world-size factor 1. Sum all rank/microbatch
  gradients; do not average local means a second time.
- Backward passes execute on BF16 parameters. Convert each measured microbatch
  gradient to FP32 and sum on CPU. Dot products/norms are chunked FP64 reductions.
  Report full parameter inventory, DiT, qwen projection and other action groups.
- Report raw and weighted norms, cosine, combined vector norm, cross term, and
  first-order objective changes under combined gradient descent. Undefined
  zero-norm ratios/cosines are `null`, not invented zeros.

This is **not a bit-exact ZeRO2 optimizer replay**: separating objectives and
summing rounded BF16 microbatch gradients in FP32 differs from joint backward,
DDP/ZeRO reduction and accumulation rounding. There is no Adam preconditioning,
clip or optimizer update, so no claim of actual update direction, one-step PDMS
improvement, or exact whole-training continuation follows. CUDA RNG states from
four ranks are replayed sequentially on one identically configured architecture;
actual device properties are recorded. Saved policy/CFM replay must still pass.

## Execution and resource boundary

New thin entry:
`python -m tools.st_world_policy.audit_flow_gradient_balance --registration FILE`
validates only small receipts/config/source identity. Add `--execute` for real
reconstruction and gradients. Unknown fields, wrong source, wrong original
world/batch, mismatched checkpoint/rollout boundary, or an existing output are
rejected. `--help` is captured in `CLI_HELP.txt`.

A Lead registration must bind final integrated source SHA, original training
registration path/hash, exact native receipt path/hash and four exact rollout
receipt paths/hashes, separate absolute output, one admitted host/GPU UUID and
CPU threads. Required `diagnostic` object is:

```json
{"cell":"C1","global_scenes":32,"original_world_size":4,
 "optimizer_steps":0,"new_score_calls":0,
 "analysis":"separate_micro_gradients_fp32_sum"}
```

Required budget keys (proposed maxima, not admission): `max_seconds=1800`,
`max_read_bytes=206158430208` (192 GiB logical reads),
`max_rss_bytes=68719476736` (64 GiB), `max_output_bytes=8388608`.
The complete native checkpoint is about 11.47 GB including four Adam shards;
strict native verification reads those shards even though no optimizer is
constructed. Parent S0 verification/loading is additional and dominates reads.
Two full FP32 CPU gradient vectors require roughly 6.6 GB for the actual action
parameter inventory; precise counts/cost are measured at execution.

One authorized 80-GB GPU, CPU threads 2, no worker pool or distributed launch.
Set `CUDA_VISIBLE_DEVICES` to the registered UUID. `max_seconds` uses SIGALRM;
read/RSS checks occur at stage/rank boundaries and can exceed a bound inside a
single library call before rejection. An independent systems-approved outer
wall/RSS/owned-process wrapper must enforce process-level limits, account
startup/import reads and confirm no pressure/real-job contention. This module
is not a resource scheduler. No actual resources have been leased here.

## Tests and remaining acceptance

Initial default environment was Python 3.9 (`navsim`), which cannot import an
existing `torch.Tensor | None` annotation. This environment failure is retained
in `CPU_TESTS.txt`; no dependency or old source was changed. Correct existing
Python 3.10 `ddp` environment: **12 passed in 5.09 s**, normal dropout on in the
real reduced-width production action-head fixtures. `CPU_TESTS_DDP.txt` contains
the complete result. These tests cover rank ownership, boundary mismatch,
rank payload binding, analytic global reduction, weighted cosines, zero reward
signal, original BC/FPO joint-gradient comparison, train/eval modes, invalid
replay and RNG restore on exceptions. They are not real S0 evidence.

Independent correctness and resource reviews, exact real input/condition/CFM
replay, original global feedback identity, GPU memory/cost and real gradient
statistics remain **NOT_RUN/PENDING**. The producer cannot self-sign them. If
replay fails, preserve diagnostic failure; do not load mismatched samples,
change precision, drop dropout, approximate old CFM, or relabel a partial result
as the intended mechanism measurement.

Final targeted regression: **53 passed in 6.61 s** (34 existing audit tests +
19 new balance/registration tests); subprocess wall 7.875 s, peak child RSS
740,556,800 bytes, CPU user/system 9.400/0.778 s. Full output and measured
resources: `REGRESSION_FINAL.txt`, `CPU_TEST_RESOURCES_FINAL.json`.
All executed test invocations stayed within the 120-second preparation budget.
No real checkpoint tensor or train manifest was opened by these fixtures.
