# L1_CENTER_V1: qualified and dispatched bounded correction

The registered original L1/L2/L3 runs and their immutable source/evaluations remain intact. Their conditional networks learned a nearly global distribution because the first tanh saturated. This correction changes **only** the small root policy's input: subtract one immutable TRAIN512 mean before its existing LayerNorm. It does not change the frozen Qwen-Flow generator, root Gaussian, PPO, reward, data order, optimizer, learning rate, prior regularizer or budget. There is no feature-wise std division, new network, extra training seed or new control matrix.

The original policy's default behavior is preserved: a `None` center creates no state-dict key or RNG draw and performs no extra arithmetic. CPU comparison against the actual immutable97 class at both zero and nonzero checkpoints confirms identical keys, identity, outputs and every parameter gradient. The centered policy adds one fixed non-trainable buffer and seals its source statistic.

The mean is fitted in FP64 over one frozen feature vector per unique TRAIN scene, then stored FP32. Both visits to all 512 scenes were bitwise identical. Its asset and tensor hashes, source model, common protocol and TRAIN manifest are bound in the training identity. It was not fitted on DEV or test, and contains no qualification actor weights.

Formal execution requires both the unchanged common screen lock and a **separate root-approved correction lock**. The original [draft](../../../configs/qwen_flow_exploration/latent_center_budget_draft.json) remains deliberately `AWAITING_ROOT_REVIEW` with `root_dispatch_approved=false`; the runtime rejects that draft. After real-model qualification, the main agent published the distinct [approved lock](ROOT_APPROVED_LATENT_CORRECTION_LOCK.json), SHA `ab546a6082e20d521e42d4702862a34c2ed206c4e4a689b97a516c2767065829`. It permits exactly one correction. The fixed plan is seed42, B16/G16, 64 fresh rounds, four inner updates, 256 Adam updates and 16,384 newly scored candidates. Peak LR1e-4, eight warmup updates, joint clip0.2 and prior coefficient0.01 remain L1 values. Qualification policies cannot initialize a formal run.

The deployment factory reloads and verifies the exact mean from the training identity, strictly loads the centered policy and checks the mean buffer hash. Primary inference retains the original `native_qwen_sourceflow_ODE` CPU scene-bound root innovation, batch-one small-policy execution, and one source-shift5 ten-step ODE trajectory. Sampled and deterministic mean-root readouts remain separate. Original arms reject centering flags instead of silently enabling them.

## Executed CPU acceptance

[Acceptance v2](L1_CENTER_V1_CPU_acceptance_v2.json) passes seven checks using the real mean and saved initial behavior batch:

- Default class equivalence to immutable97, including all parameter gradients.
- Same seed42 trainable initialization; only the fixed mean buffer is added.
- Exact initial `N(0,I)`, original root-stream replay, joint density and B1/B16 outputs.
- Four real Adam steps on unchanged saved roots/rewards/advantages; all gradients finite and mean buffer unchanged.
- Atomic interruption/resume at inner1, inner2 and inner3; policy, Adam, scheduler, old batch, mean and subsequent root RNG match continuously executed updates bitwise.
- Formal execution without its separate correction lock is rejected.
- The actual `latent_train` CLI runs a centered qualification with a tiny source ActionDiT and explicitly mock scalar, then resumes an inner1 interruption to the identical four-step terminal state. This last check validates orchestration, not NAVSIM planning.

The real saved GPU densities differ from CPU reduction by at most 7.63e-6. CPU arithmetic qualification records this difference and computes the same underlying initial Gaussian on CPU for both sides; original old-probability artifacts are never rewritten.

## Executed real-model qualification and formal dispatch

[Real-model qualification](CENTER_real_model_qualification_v1.json) completed 16 TRAIN scenes × G16 = 256 new canonical scorings and four real Adam updates. The initial inner ratio is exactly 1, all frozen generator parameters/buffers remain unchanged, and the run publishes a legal COMPLETE checkpoint. Worker wall is 151.89 seconds, including 54.12 seconds of rollout, 8.94 seconds of condition construction and 8.02 seconds of scoring wait. This one-round qualification uses one warmup update; the formal fixed window uses eight. Its 256 candidates are a separate qualification cost, and its policy weights were not used to initialize formal training. This is chain/numerical acceptance, not a PDMS efficacy result.

The main agent dispatched the fresh formal run at **2026-10-24 20:53:08 UTC**, on authorized `training-vla-zt` GPU1, from immutable source `7ec40a7f3e11496d7fa9ae3a0700a02926da5eed`. Original immutable97 training and 9ecd evaluation artifacts remain unchanged. The [execution index](CENTER_execution_index_v1.json) records the actual command, run identity, approval, source and evaluation queue. This is a dispatch-time status snapshot, not a claim that training or evaluation has finished.

Executed command:

```bash
CUDA_VISIBLE_DEVICES='' /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.qwen_flow_exploration.latent_center_validate \
  --output /mnt/project/qwen-flow-three-routes-20261024/qualification/L1_CENTER_V1_CPU_acceptance_v2.json
```

## Actual formal command and registered evaluations

The main agent owns GPU assignment and process launch. The recorded formal command executes from `/mnt/project/qwen-flow-three-routes-20261024/snapshots/latent_center_v1`:

```bash
/mnt/project/flow-safe-improvement-20261013/venv/bin/python -u \
  -m scripts.analysis.qwen_flow_exploration.latent_train \
  --protocol /mnt/project/qwen-flow-three-routes-20261024/common_protocol.json \
  --arm L1_CENTER_V1 --seed 42 \
  --center-normalizer /mnt/project/qwen-flow-three-routes-20261024/qualification/latent_center_saved_data_v2/TRAIN512_feature_mean.pt \
  --correction-budget-lock /mnt/project/qwen-flow-three-routes-20261024/LATENT_CORRECTION_LOCK.json \
  --output /mnt/project/qwen-flow-three-routes-20261024/runs/L1_CENTER_V1_seed42
```

Do not launch that command again while its registered owner is active. Resume adds `--resume <same run or legal checkpoint>` and preserves all other arguments. The finite evaluation owner `finite_queue_center_eval` is registered on authorized tsinghua-zt3 resources and waits for successful formal completion before executing DEV_SELECT sampled-root, TRAIN sampled-root and DEV mean-root evaluations. It uses the same immutable source and reconstructs the exact mean from the checkpoint identity. These queued evaluations have not been claimed complete here. The YAML [recipe](../../../configs/qwen_flow_exploration/latent_center_v1.yaml) remains the original preparation record; the approved lock and actual run identity bind active execution.

Recovered input dependence in a four-step probe does not establish a PDMS improvement or prove that collapse cannot recur over the full window. The correction's value must be judged after its full fixed-window, single-output evaluations. No additional controls, seeds or correction variants are registered.
