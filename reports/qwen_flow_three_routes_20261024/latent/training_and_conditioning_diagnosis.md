# Terminal training audit and conditionality diagnosis

All three latent-policy screening runs completed the registered window on 2026-10-24. Deployment evaluation remains separate; this document does not infer planning improvement from successful updates.

## Actual training and cost

Each arm has 64 fresh collections, 16 scenes per collection, 16 candidates per scene, four real Adam updates per collection, and 256 final optimizer updates. The 512 TRAIN scenes are each visited exactly twice: 1,024 scene exposures and 16,384 newly scored candidates per arm. Scanning the actual collection records verifies these counts and unique candidate identities. The four updates within each collection retain the same behavior and batch digests. Adam counters and the scheduler end at 256.

Checkpoint publication seals, payload hashes, final learned-policy tensor hashes, and imported source-file hashes match their identities and immutable training snapshot `97a66c1`. Every run completed in one invocation without rollback. The entire frozen generator's named parameters and buffers retain the same hash before and after training in all arms. The post-training evaluator uses the separately pinned evaluation snapshot; this does not change the trained policy.

| Arm | Worker wall, seconds | ODE generation | Condition prefill | Scoring wait | Root-policy updates |
|---|---:|---:|---:|---:|---:|
| L1, conditional + prior KL | 4,354.78 | 3,452.71 | 434.85 | 179.85 | 47.48 |
| L2, global + prior KL | 4,285.96 | 3,461.26 | 393.03 | 172.04 | 26.15 |
| L3, conditional without prior KL | 4,285.66 | 3,428.85 | 392.51 | 183.01 | 44.80 |

These are shared-GPU worker occupancy and measured phase times, not exclusive GPU compute. ODE generation dominates; the small policy optimizer takes about 0.6–1.1% of total wall time. Each arm makes 40,960 actual batched DiT velocity calls with chunk size four, equivalent to 163,840 candidate-step velocity evaluations. The scorer receives 1,024 candidate batches and performs 16,384 logically independent candidate evaluations. Small condition-LRU reuse reduces actual Qwen prefills to 1,021; it does not change scene-exposure counts. Loading, hashing and checkpoint costs remain in total wall rather than being hidden as free work.

## Signal availability

| Arm | Exact tied groups / 1,024 | Tied full-score | Tied zero-score | EP-only effective | Groups with safety-component variation |
|---|---:|---:|---:|---:|---:|
| L1 | 345 | 339 | 6 | 627 | 36 |
| L2 | 339 | 333 | 6 | 641 | 31 |
| L3 | 345 | 339 | 6 | 622 | 36 |

Here EP-only means the score varies and EP varies while NC/DAC/TTC/Comfort/DDC remain identical. It represents roughly 92–94% of non-tied groups. This is evidence about the actual behavior samples, not proof that every update harmed safety. Tied full-score groups dominate tied groups; zero groups are a small fraction. No bad candidates were removed to create these counts.

The final training log evaluates the current distribution **before the last Adam step**, on the last 16-scene saved batch. Its joint prior KL is 1.12274 for L1, 0.00044593 for L2, and 2.04788 for L3. Thus equal learning rates and update counts do not produce matched root-distribution changes. L1 versus L2 cannot establish the necessity of scene conditioning without retaining this alternative explanation. Full frozen terminal-policy diagnostics come from the sealed evaluation records.

## The conditional networks learned nearly global distributions

A CPU-only probe uses the actual step-0 and step-256 checkpoints on identical saved inputs from the final 16 TRAIN scenes. No new model rollout or scoring is performed. Initial first-layer preactivation RMS is 0.57392, with no `|preactivation| > 3` and no first-tanh `|activation| > 0.99`. At the terminal checkpoint:

| Quantity | L1 | L3 |
|---|---:|---:|
| First-linear preactivation RMS | 5.08082 | 5.32022 |
| Fraction `|preactivation| > 3` | 100% | 100% |
| First-tanh fraction `|activation| > 0.99` | 100% | 100% |
| Root mean total RMS | 0.245352 | 0.343225 |
| Root mean between-scene RMS | 2.51e−8 | 1.88e−8 |

The sixteen inputs do differ: pooled-feature between-scene RMS is 0.02799. Their large common feature direction dominates their differences: context RMS is 6.37394 with between-scene RMS 0.03224; visual RMS is 1.25998 with between-scene RMS 0.02105. The first layer can move along this shared direction and saturate, making its output almost independent of scene.

The evidence distinguishes learned collapse from an initialization or near-zero-standard-deviation error. Pooling is an arithmetic mean over each real nonempty boolean token mask, followed by concatenation into 7,168 features. No TRAIN feature mean/std normalizer exists. Existing LayerNorm normalizes across features within each scene, with `eps=1e-5`; its actual denominator is 4.85–4.93, far from zero. Reward normalization appears only in detached advantages, never in actor condition. First-linear weight/bias changes account for preactivation-shift RMS 4.909/5.142, while changes in LayerNorm affine parameters contribute only 0.0093/0.0102.

This one-batch probe does not establish that every TRAIN/DEV scene has the same distribution. The complete evaluation records must confirm the population-wide between-scene mean/log-std variation. It does establish that nonzero parameter gradients and a conditional input interface are insufficient evidence of useful conditional learning. A deployment gain from these checkpoints could instead reflect a learned global root adjustment.

## One bounded follow-up, subsequently approved

If terminal evaluation justifies another experiment, the smallest normalization change is to subtract one immutable TRAIN-only pooled-feature mean **before the existing LayerNorm**. Keep the same small network, tanh, likelihood, clipping, learning rate, prior coefficient, data order and candidate budget. This removes the dominant shared feature direction without dividing by small feature-wise standard deviations. It must use a separately named run, for example `L1_CENTER_V1`, with a TRAIN-manifest-bound mean tensor/hash, original zero-initialized root prior, and unchanged frozen generator. Old runs remain intact.

Before such a run, use the exact existing initial behavior batch for a paired four-inner qualification: verify initial Gaussian parameters/densities remain identical, feature variation survives, gradients are finite, and old roots/rewards remain unchanged. The normalizer is a declared method change, not a repair silently inserted into completed training. No evidence yet says it improves PDMS. At the saved-data probe stage no new arm had been launched; the later approved real-model qualification and formal dispatch are recorded in [CENTER_V1](CENTER_V1.md).

That bounded **CPU saved-data qualification has now run**, without generator inference or new scoring. Across all 64 saved collections, all 512 scenes have two bitwise-identical feature vectors. A mean fitted once per unique TRAIN scene has tensor hash `f5b6b0a799d4143f752a1f18cbd81d239edc51ee2c9ffed4945e0a79528d0a88`; its protocol, base-model and TRAIN-manifest identities are sealed. The private mean asset is approximately 28 KiB, not a trained policy. Its identity is explicitly qualification-only.

Both unchanged and centered inputs use the exact same initial policy parameters, saved first-batch roots, real rewards/advantages, four Adam updates, warmup learning rates, PPO clipping and prior coefficient. Initial distributions are exactly `N(0,I)` and first-inner ratios exactly one on the CPU arithmetic path. Root/reward/advantage tensors and original artifacts remain unchanged. The saved GPU-to-CPU joint-density rounding difference is separately reported as at most 7.63e−6; it is not hidden by rewriting any original old probabilities. This is not a new GPU equivalence qualification.

After those four updates, full TRAIN512 root-mean between-scene RMS is 4.84e−7 with unchanged inputs and 0.001516 with mean centering. First-tanh between-scene RMS is 0.00157 versus 0.49653, and saturation remains below 0.002% for the centered probe. Gradients are finite in both cases. These observations establish bounded feasibility and recovered input dependence, **not a PDMS gain or protection against collapse after 256 updates**. Qualification policy weights are not exported. The first probe attempt failed because the analysis script assigned into an immutable dataclass; it was corrected using `dataclasses.replace`, with the failed attempt preserved separately. Formal runs were unaffected.

Executed entrypoint:

```bash
CUDA_VISIBLE_DEVICES='' /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.qwen_flow_exploration.latent_center_probe \
  --output /mnt/project/qwen-flow-three-routes-20261024/qualification/latent_center_saved_data_v2
```

Evidence: [terminal audit](terminal_training_audit_v1.json), [saved-feature probe](saved_feature_conditionality_probe_v1.json), [initial/terminal diagnosis](conditionality_minimum_diagnosis_v1.json), [CENTER qualification](CENTER_saved_data_qualification_v2.json). Only aggregate diagnostics are committed; hidden-feature vectors, normalizer tensor and raw candidate records remain private.
