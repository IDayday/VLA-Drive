# Independent Route A trainer and G0 resource review

Reviewed source: `e637f2fe99a54a268c0405aa8107787da6ebacf5`; specifically `train_refiner.py`, `effect_guided_refiner.py`, refiner/loss definitions and the already-reviewed common checkpoint loop. No common/collector/G0 implementation was modified.

Decision: **PASS for affected CPU mathematics, private-parameter isolation, sampling and same-runtime CPU recovery. Formal A1/ScoreRefiner admission is conditional on explicit own-campaign G1 provenance binding**, discussed below. Actual GPU qualification and scientific repair/generalization are not established by these tests.

## Sampling and supervision

The production `train()` separates verified local repairs from high-quality identity targets, rejects an empty bucket and builds equal-sized virtual reference lists using their least common multiple. This preserves exact 50/50 probability under the existing uniform sampler and uniform probability within each bucket; it does not promise exactly half of every random minibatch. No extra 0.5 factor is applied to the same sample's supervised loss. Duplicate references are declared exposures, not new unique scenes.

Only `role=fit` and `trainable=true` enters optimization. Validation entries remain separate. Target trajectories/coefficients must be finite, within the registered bounds, reconstruct the target through the same canonical basis and use only grid-exposed coefficients. Identity targets require exact Base geometry and all-zero coefficients. Compact records are reconstructed from native receipts, Base labels/trajectories are checked against independent native feedback and context rows, and paired endpoints must share scene/context/reference/numeric/protocol identity.

Actual target metadata and frozen context remain separate: the model receives only detached current memory/ego/mask plus Base trajectory. Targets, metrics and simulator contexts are supervision. Cached head training loads no full Base and runs no VLM or simulator. Frozen teacher/Base modules are returned to eval by the outer wrapper's `train()` override.

## Fairness and gradient objective

A0, A1 and ScoreRefiner create the same coefficient refiner with the same initialization and active coefficient mask. Teacher weights are separate, frozen and excluded from private optimizer/checkpoint tensors. A0 has no teacher. A1 selects the trained delta-feedback mode; ScoreRefiner selects trained absolute-score mode and masks guidance to its actually trained score-like heads. Event guidance is disabled explicitly, so untrained absolute-model event heads cannot leak random supervision into the comparison.

Supervised fit, magnitude and preservation objectives are shared. A1/ScoreRefiner additionally compare predicted effects to verified target effects and penalize predicted safety degradation. Each microbatch numerator is divided by the corresponding complete-update valid-scene count. Thus masks and high-quality counts do not change the objective when the batch is split, and graphs can be released after each microbatch. The proposed trajectory retains gradient through the frozen teacher; its input context and Base trajectory do not update the upstream model.

The continuous coefficient refiner can propose combinations outside the discrete G0 sample grid. That is recorded as a real generalization requirement, not a guarantee from geometry screening. Native controller tracking and PDMS must be evaluated independently after training. Geometric rejection may return Base, which must be reported separately from successful learned repair.

## Independent CPU checks

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/action_effect/test_route_a_independent_review.py
```

Actual result: **6 passed, 1 warning in 4.40s**. The warning is deprecated `pynvml`. Tests use synthetic cached contexts; no real GPU, Base model, GT or simulator was loaded.

- The actual train entry with three repair/five keep scenes builds30 references: each repair appears5 times, each keep3, giving15 references per class and no invented unique scenes.
- All three arms have identical initial private parameter names/values. Required teacher modes are delta-feedback and absolute-score respectively.
- All three arms' full-batch versus microbatch gradients and task statistics agree with fixed `rtol=1e-5, atol=1e-6`, including a fully invalid effect-label scene and unequal valid label/safety/preservation counts. Expected denominators: five supervised scenes, three keep scenes, four effect-valid scenes, two safety-valid scenes.
- Teacher parameters/gradients remain unchanged; outer training mode leaves it in eval.
- A1 actual shared-loop continuous versus interrupted/resumed CPU training produces identical next scene IDs and complete model state, including frozen teacher/buffers. Terminal replay consumes no data and takes zero optimizer steps.

No whole-suite count is inherited and no previously approved large matrix was rerun.

## Own-teacher provenance finding

The inspected loader strictly verifies checkpoint/receipt/run identity, matching Base, declared allowed mode, positive optimizer step and compatible private tensor names/shapes. These prevent accidental wrong model/mode/Base loading. However it did not yet follow `identity.data_artifact` to a complete pair manifest with the expected current campaign registration. Merely writing `external_weights_loaded=false` does not establish that chain.

The Lead has requested a focused correction: optional `expected_registration_identity` on `load_effect_teacher`, verification of the pinned actual G1 pair artifact and its own immutable registration, and matching it to the A/B repair manifest registration. A/B formal training has not started. That correction and its author's independent review are separate from this pre-fix report; do not read this report as approval to omit the provenance check.

Common checkpoint identity preserves Base/data/teacher/config/runtime dependencies, optimizer/scheduler, sampler and RNG. On resume, re-created frozen dependencies must match their immutable pins; saved training RNG is restored after construction. Learned private buffers are not present here: coefficient bounds and active-mask buffers are deterministic config-derived constants. CPU exactness does not imply cross-backend CUDA exactness.

## Actual 256-scene G0 resource metadata

Read raw registrations, complete-scene JSON and filesystem sizes only. No canonical feedback payload was decoded or rescored. Structured data: `ROUTE_A_G0_256_RESOURCE_METADATA.json`.

- Population:256 complete scenes, 199 fit/57 validation.
- Current-population reservations:4431 new calls, zero retries. Original reused store contains6007 canonical records total, including1576 earlier completed calls.
- Shared native-store logical size:96,417,500 bytes across24,028 files. Canonical compressed bodies occupy79,920,971 bytes; admissions7,327,257 bytes; completion receipts9,169,272 bytes; lock files have zero logical bytes.
- New-root logical files excluding the shared directory symlink:1,008,558,100 bytes at this observation.
- Observed shared free bytes:340,994,117,632, about317.6GiB. Declared floor is300GiB and the new-root cap8GiB. Other concurrent workloads may change availability.

| Population | Worst-case26 calls/scene, total distinct candidates | Maximum further calls beyond current6007, before explicitly bounded retries |
|---|---:|---:|
| 512 | 13,312 | 7,305 |
| 1024 | 26,624 | 20,617 |

These are conservative planning bounds, not actual future call counts. Both fit below the registered32768 call ceiling. At the current average16,051 logical bytes per canonical result including admission/receipt, all1024 scenes would be roughly0.40GiB of shared native logical files. Reserve up to three times that estimate for filesystem blocks/directories and variation; future actual use must be measured. Wrapper copies, scene/compact JSON, validation receipts, TRAIN/DEV features and private checkpoints are separate allocations, already included in the broader new-root plan.

**Safe next action from the observed resource state:** continue first to512, recheck shared free space/logical output cap/shared-native growth, then proceed to1024 if guards still pass. Keep explicit per-invocation call/retry ceilings. The existing common output traversal does not count the shared cache symlink target toward the new-root cap, so record that growth separately. No automatic checkpoint deletion is justified or authorized.

## Scientific limitation in this population

255 scenes are `NO_CHANGE_REQUIRED`; only one has `VERIFIED_LOCAL_REPAIR`, and it belongs to fit. There are currently **zero verified repair targets on the57 held-out validation scenes**. A balanced training run would therefore repeatedly expose one genuine fit repair. It can establish engineering/fit behavior but cannot demonstrate repair generalization or a robust learned improvement law from this population. Keep exposures and unique support explicit.

Continuation to the already locked larger cohort can test whether useful opportunities are sufficiently supported. Do not alter DEV or repair thresholds to manufacture gains. Full held-out evaluation remains required, and a repaired training sample is not a held-out success. No scientific positive verdict follows from these source/CPU checks.
