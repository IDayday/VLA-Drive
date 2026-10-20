# Independent G1 training correctness review

Review date: 2026-10-20 UTC. Reviewer: `effect_navsim_feedback`, independent of the G1 model/trainer authors. Scope: G1 parser, losses and common small-module trainer. This review does not approve the reviewer's own feedback-wrapper/G0 code independently.

Decision: **PASS for a bounded single-GPU, at-most-32-update engineering qualification**, provided its immutable source includes `9f1ae2d1` and the repair-head fix `5ee472df`. Full learning, held-out improvement, uncertainty calibration and production CUDA/DDP recovery remain **UNTESTED** by this review. No simulator call or GPU training was performed by this reviewer.

## Sources and scope

- Original target: G1 `e93abdadeef0bbb4360c43e2e9dd4ed6dfb7e277`, common training `f43eca756bc22663d0a2c2c7d65ed14d4b923934`.
- Checkpoint interruption fix: `8663a82c134645d98ecbc887b9f6e43cace0dd80`.
- Native-to-compact row binding: `9f1ae2d1`; the review worktree is at this source.
- Absolute-control repair-head fix: independently loaded and tested from `5ee472df5c3d7bb66d41103c61dd08917e15fe56`. This commit was not an ancestor of the review worktree's `9f1ae2d1`, so it must be integrated into the qualification snapshot explicitly.
- No files in the owner implementation were modified by this review. Only the independent tests and evidence report are added.

## Findings, reproductions and resolutions

| Finding | Trigger and original code | Independent reproduction | Resolution | Remaining boundary |
|---|---|---|---|---|
| G1-R1: completed periodic checkpoint cannot be followed by a same-step pause save | `training.py::save` in `f43eca75` always writes immutable `step_N.pt`; SIGTERM after its periodic receipt sets pause before another step | A real signal to this test process after step-2 receipt causes `Immutable content conflict`. The original running checkpoint remains intact. | `8663a82c`: preserve canonical bytes, store different state under a content-SHA suffix; latest same-step receipt determines selection, terminal wins ties. | Signals targeted only the isolated test process; existing jobs were untouched. |
| G1-R2: payload-before-receipt crash prevents later resave | Step-4 payload persists, receipt creation raises; recovery selects step 2 then collides with the orphan at step 4 | Injected exception before receipt, then actual legacy resume reproduces conflict and confirms orphan bytes are unchanged. | `8663a82c`: preserve orphan, use content-addressed variant; resolve receipts rather than result, return the actual resolved path on terminal replay. | CPU exactness is not CUDA bitwise qualification. |
| G1-R3: self-hashed compact rows were not tied to their native candidate content | Original `PairData` checked the parent registration and row identities, but could accept rehashed labels or a forged source log independently of native receipts | Source inspection plus an independent negative test: modify a label, recompute row and manifest identities, and retain the expected reconstructed artifact; admission rejects the changed content. | `9f1ae2d1`: verify registration/Base, reconstruct the immutable dataset from canonical scene/candidate receipts, require exact compact artifact SHA, and require source-log equality with the context row. | Native files are validated once at first admission; they must remain immutable. The negative unit test isolates the checksum gate; the separate real load below exercises the full chain. |
| G1-R4: absolute-feedback control had an impossible pair-dependent auxiliary target | Old `ActionEffectModel.forward` predicted `repair_logit` only from `(base,base)` features, while `eligible_repair` depends on the proposed trajectory | For fixed base/context, the repair-logit gradient to a changed proposal is exactly zero in source `9d6dddce`; the same test has a finite nonzero gradient in `5ee472df`. | `5ee472df`: all groups' repair head reads both trajectory branches using a signed feature difference, preserving parameter count. | This makes the control capable of observing the target's inputs. It does not establish equal fit difficulty or scientific superiority. |

The historical fault tests dynamically load their pinned original source. They deliberately continue to demonstrate the original failures after the implementation is fixed, without asserting that current code should fail.

## Independent executable checks

Command, in `/mnt/project/VLA-Drive-action-effect-feedback-20261020`:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/action_effect/test_g1_training_independent_review.py
```

Actual result: **6 passed, 1 warning in 3.57s**. The warning is the environment's deprecated `pynvml` package. Python 3.10.20; PyTorch 2.5.1+cu124; CUDA hidden; at most two Torch CPU threads.

Coverage:

1. Historical same-step periodic-save/pause conflict.
2. Historical payload-before-receipt orphan conflict.
3. Current continuous versus interrupted training: next scene IDs and pair draws match; all private parameters, complete Adam state, scheduler, sampler, consumed count, Python/NumPy/Torch RNG and independent pair generator match exactly on CPU. Validation deliberately consumes random numbers and is correctly isolated. Terminal replay executes zero new steps, consumes no scene and repairs the result using the actual resolved artifact.
4. Full loss composition versus three microbatches with unequal valid-channel counts, a fully empty scene and an empty middle microbatch: all six task values and the shared-parameter gradient match at a predeclared FP32 absolute tolerance `1e-6`; no relative-tolerance relaxation. Ranking mass is `1.37`, retained without clamping it to a scene count.
5. Old/fixed absolute repair-head proposal gradient.
6. Rehashed compact-label mutation is refused before training.

This is a focused independent run. The owner's historical whole-suite counts are not inherited as new evidence, and no whole-suite repetition was requested.

## Real immutable-data parser admission

An actual `ContextReader` and `PairData` load was executed on CPU, including canonical feedback reconstruction and one actual context shard record. This is real cached S0 output and actual official-feedback data, not a tensor fixture.

Inputs:

- Binding: `/mnt/project/action-effect-repair-20261020/CAMPAIGN_BINDING.json`.
- Contexts: `/mnt/project/action-effect-repair-20261020/contexts/MANIFEST_000064.json`.
- Pair dataset: `/mnt/project/action-effect-repair-20261020/g0/population_000064/PAIR_DATASET.json`.
- Pair bytes: `18,518,378`.
- Pair SHA256: `d20b65bef61a0d041683ec5b0b65987a51f4b5ef669e1e87955bbcd9f40a45dc`.

Result: **PASS**, measured first-load wall **71.14474904537201 seconds**. No simulator execution, loss, optimizer or GPU was involved. The reconstruction only reuses immutable existing bytes; no old evidence was replaced.

Admitted G1 fitting population:

- 45 scenes across 44 source logs; 806 pairs after preregistered joint-intervention exclusion.
- Longitudinal 343, lateral 340, larger-support 123.

Held-out training-log validation population:

- 19 scenes across 17 source logs; 405 pairs.
- Longitudinal 143, lateral 140, larger-support 51, joint 71.
- `PairData` rejects any fit/validation log overlap. Joint interventions are never sampled for G1 fitting and remain explicitly reported in validation.

Actual deployed-context tensor shapes, after batching one record:

- memory `[1,152,2048]`, comprising the actual 144 world and 8 action tokens.
- ego `[1,1,4]`, the source's normalized history delta pose, **not a velocity vector**.
- valid mask `[1,152]`.
- One actual shard was loaded into the bounded CPU LRU.

Actual labels are the seven official `0..1` terminal quantities (`score`, NC, DAC, EP, TTC, comfort, DDC), raw progress in meters, and three event times in seconds. The bound scales are `1` for score-like quantities, `20m` for progress and `4s` for times. Missing first events remain masked rather than becoming zero-time events.

## Loss, leakage and scope conclusions

`PairData.predict` passes only current frozen memory/ego/mask and the two physical trajectories. `targets` constructs scores, masks, transitions and temporal supervision separately. `batch_context` does not forward GT, future actors, metric caches or raw training targets. The model rejects unexpected context keys and detaches frozen current representations. Proposed trajectory gradients are intentionally retained for future A1 training, not propagated into the frozen S0 generator.

Sampling is uniform over fit scenes followed by uniform valid pairs within the sampled scene. Each task combines additive numerators and denominators before division across microbatches. Ranking retains physical score-gap mass. Per-scene channel means avoid giving a scene with more valid channels more pointwise weight. The two losses use different declared denominators rather than accidentally cancelling gap weights.

The current trainer is a **single-process small-head trainer**. Distributed reduction utilities exist, but this review does not label it a qualified DDP training entry. A same-runtime requirement is stated in recovery receipts; training device/backend flags are not yet a comprehensive independently enforced migration contract. Use the same recorded software/device/determinism settings for bounded qualification, and do not infer cross-backend bitwise continuation from these CPU results.

First admission currently reconstructs and validates all native receipts for each newly created `PairData`, costing 71.1s for the 64-scene pilot. This is startup/data-audit cost, not steady-state head speed. Before scaling many arms, an immutable verified-data receipt or in-process validation reuse can remove redundant work while preserving the exact native/compact content identities. Do not repeat the full scan per optimizer step.

## Bounded GPU qualification handoff

The reviewed code may proceed to a separate, immutable 32-step engineering run with real cached data after integrating both required fixes. Keep the registered long scheduler horizon and use `--stop-after 32`; do not relabel a shortened cosine timeline as continuation. Record GPU model, actual update times/memory, finite task losses, nonzero intended gradients, complete checkpoint/receipt and independent-process resume. If this run is resumed, hold source/data/runtime identity fixed. No scientific positive-gain conclusion follows from a successful short run.

Real image replay, true simulator correctness, G0 science, CUDA exact recovery, G1 generalization and final deployment export are separate evidence owned by their assigned reviewers and experiment dispatcher.
