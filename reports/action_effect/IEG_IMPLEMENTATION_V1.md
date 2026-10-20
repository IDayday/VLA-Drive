# IEG: independent residual Flow implementation

Implementation base: `385d4c53e49a52b89e7845352d69230e4e6503c1`. Worktree `/mnt/project/VLA-Drive-effect-internalization-20261020`, branch `feature/effect-internalization-20261020`. These modules are an original adaptation of the audited local S0, not copied external training code or externally initialized weights. No routeA checkpoint, trained scorer or external effect network initializes an IEG adapter.

Primary-source design evidence was read at `/mnt/project/VLA-Drive-action-effect-evidence-20261020/reports/effect_repair/RELATED_WORK_AND_REUSE.md` (its exact SHA is in the accompanying implementation JSON). Its B0/B1 shared-path card and FlowDPG/DFQL approximation boundaries are followed: a repaired-trajectory student is an established baseline, and a one-step learned-effect surrogate is not an exact simulator gradient or a proof about final ODE actions.

## Implemented API

- `EffectFlowAdapter(EffectFlowAdapterConfig(...))`: private 2-layer256-wide4-head cross-attention module by default. Inputs are frozen original planner condition `[B,8,2048]`, raw noisy actions `[B,8,4]`, and Flow time `[B]`. It has random body weights and an exactly zero final layer. Its explicit initialization seed is isolated using `torch.random.fork_rng`, preserving the caller's training RNG. A fixed bounded multiplier alpha is configured; the residual itself is not artificially clipped to a small displacement.
- `EffectInternalizedFlow(original_action_head, adapter, arm=...)`: freezes all Base action parameters, permanently keeps their modules in eval, and exposes only adapter parameters to the optimizer. Each arm is constructed independently. It reuses `DiTFlowPolicy` and the unchanged original head's projection/velocity function.
- `PairedFlowSample(minus, plus, noise, times)`: explicit detached supervision. `sample_original_pair` makes one draw per supplied row using two independent resumable generators. A trainer preserving original repeat8 must explicitly repeat the same selected endpoint/condition/label rows eight times **before** sampling; do not count these as eight unique scenes. No hidden sampler expansion or changed exposure count is performed inside the model.
- `model(condition, pair, valid_scenes, ...)` returns a dedicated `losses` dictionary of scalar losses plus diagnostic tensors. Only `losses` should be weighted/summed. `global_denominator` is the valid raw-element count for the complete optimizer batch, including microbatch accumulation. Under DDP each rank scales its local sum by world size/global count; ranks with zero valid rows retain a graph-connected zero.
- `model.predict_action(condition, initial_noise=eps)` takes no feedback, target, teacher, metric cache or future input. It executes the original ten Euler velocity evaluations with the internal adapter at each step, and returns the raw8×4 trajectory. It never generates a complete Base trajectory and then refines it. Disabling the adapter delegates to the original head's complete `predict_action`.

No new VLM is owned by these modules. The caller obtains the original planner condition once from the shared S0 or a correctly bound frozen-condition cache. Normal deployment converts raw XY/sin/cos through the original `decode_ego`; the wrapper does not substitute angle channels or a different normalizer.

## Three distinct matched-capacity objectives

Let source minus trajectory be `a−`, verified repair be `a+`, shared noise be epsilon and source time be u. Original S0 uses:

```
x_u(a) = (1-u)*epsilon + u*a
v*(a)  = a-epsilon
u      = (.999-Beta(1.5,1))/.999
```

The small possible negative time tail is retained. No uniform-time substitution, clipping or sampler change is made.

**B0** is ordinary repaired-trajectory FM on `x_plus`, with target `a_plus-epsilon`. Since this campaign first freezes the Base, B0 optimizes the same private adapter capacity as B1, not a much larger fully trainable action head. This distinction must remain in result tables.

**B1-trajectory** uses the common source bridge `x_minus`, and total velocity target `a_plus-epsilon`. Its exact residual target at that state is:

```
g_target = (a_plus-epsilon)-stopgrad(v_base(x_minus,u))
```

It equals `a_plus-a_minus` only if the original velocity is already the exact source conditional velocity. The analytic test includes a deliberately imperfect Base and shows that blindly adding endpoint difference produces the wrong full Euler endpoint. Merely learning this corrected target is still a trajectory-supervision method, not proof that physical feedback has independent value.

**B1-effect** retains the B1 shared-path velocity target and adds separately returned `effect_consistency` and `effect_direction` losses. A verified frozen/eval G1 teacher is passed only to the training call; absence of teacher or real paired physical deltas is an error. The teacher receives a strict current-only mapping:

```
{'memory': frozen W/A, 'ego': current ego[1,4], 'mask': valid memory mask}
```

No GT/feedback key is accepted. The Lead's `RawActionEffectTeacher` bridge converts raw base/modified actions with the original physical normalizer before querying G1. Expected teacher signature is `teacher(effect_context, base_raw, intervention_raw) -> delta[B,E]`. Teacher parameters remain frozen, while gradients through the intervention reach the adapter.

The current effect auxiliary queries a clean velocity-derived proxy `a_hat = epsilon + v_new(x_minus,u)`, comparing `teacher(z,a_minus,a_hat-a_minus)` with the true verified pair delta. This proxy is not the complete Euler endpoint and can lie outside G0 intervention support. The trainer must explicitly record its applicability/support mask and out-of-support rate. No result may call this exact terminal-effect optimization. Final acceptance must evaluate complete Euler10 trajectories in the real fixed NAVSIM context. A future completed-rollout effect objective would be a separately declared compute/gradient variant.

Per-task effect scales are explicitly supplied positive fixed constants; missing labels are masked. Consistency averages squared dimensionless errors over globally valid effect labels. Direction loss compares the predicted signed effect with zero intervention and weights by true normalized effect magnitude over the **global distribution**, retaining the gap cost rather than dividing it away per scene. DDP additionally requires its global weight denominator. Default objective weights are deliberately not hidden inside the model: the immutable campaign config must declare them, and B0/B1-trajectory reject silently supplied effect supervision.

## Preservation and inference

`preservation_loss` evaluates frozen Base and adapted velocities at the identical `x_minus`, noise and time for high-quality replay. It does not require a second full trajectory rollout. Caller-side sampling fixes the replay ratio before heldout evaluation. No deployment safety guarantee follows from this loss, zero initialization or parameter isolation.

The original raw action dimension, timestep buckets and Euler10 are checked by the reused local adapter. No teacher, separate refiner or physics simulator is registered in the deployable state dict. The initial zero residual and disabled paths exactly match the production DiT class's full Euler10 in bounded CPU tests; after real adapter updates, the final sampled action changes, and frozen Base state remains unchanged.

## Checkpoint and integration boundary

The common campaign trainer owns native checkpoint receipts, scheduler, scene sampler, independent noise/time generators, optimizer state and all hashes. Construct/load the exact arm/config first; restore model/optimizer/generators and finally global RNG, then continue without reseeding. A bounded test verifies the next stochastic adapter update exactly in the same CPU runtime. This is not claimed as untested production CUDA/DDP exact resume.

For a compact canonical head artifact, save the **resolved pure-dict** original `action_model.full_config` including QwenOFT's actual constructed `DiTConfig`, and its strict canonical FP32-master state dict. Reconstruct via existing `get_action_model(OmegaConf.create(config))` without instantiating Qwen, then strict-load every action parameter. Receipt must bind Base checkpoint78f870…, source1493/protected-source hashes, config, action artifact SHA and FP32/Euler/raw representation. Do not reconstruct dimension from the `DiT-B` label alone or substitute rounded BF16 module tensors. This exporter remains owned by the Lead.

## Actual verification and remaining evidence

The current CPU run returned **13 passed, 1 deprecation warning in6.13s**. Full output is [FLOW_CPU_TEST_OUTPUT.txt](FLOW_CPU_TEST_OUTPUT.txt). Command: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python -m pytest -q tests/action_effect/test_flow_internalization.py`. Default full-width adapter has **2,706,948 parameters**; this count is measured, not latency/FLOPs evidence.

`tests/action_effect/test_flow_internalization.py` uses the **real production FlowmatchingActionHead/DiT/ActionEncoder/projection at reduced CPU width**, plus explicitly marked analytic/synthetic teacher fixtures. It covers full Euler10 zero/disabled identity, corrected-residual counterexample, distinct B0/B1 state distributions, finite real backward and adapter updates, Base/input gradient isolation, eval-mode protection, reproducible initialization, independent storage, original Beta support and generator restoration, unequal-rank/microbatch reduction arithmetic, all-invalid loss, teacher/current-context boundary, global gap weighting, preserve loss, exact next stochastic update after full small-state restore, and strict new-process export with NAVSIM imports forbidden.

These tests do not constitute real pretrained S0/image integration, real physical feedback, trained G1, real adapter training, production DDP/NCCL or any planning improvement. Those statuses remain pending the common G0/G1 dataset and Lead resource admission. No GPU, training or simulation task was dispatched by this implementation agent, and no current Base/ST job was changed.
