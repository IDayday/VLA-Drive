# First-wave method evidence — 2026-10-16

This is an evidence handoff for `WMVLA_ST_CONSEQUENCE_MULTI_AGENT_PLAN_20261017.md` v1.0. It is not an implementation-correctness sign-off, resource approval, or permission to launch formal training. No GPU, model weights, simulator, or training job was used in this audit. Public source was read at fixed commits; a small CPU test exercised the upstream FPO loss.

## Main findings

1. DINOv3 already exposes post-normalization dense patch tokens excluding CLS and storage tokens. Preserve the rectangular grid. Post-normalization means LayerNorm, not unit-length normalization. The `layernormbf16` factory name selects a different epsilon, not an automatic BF16 cast.
2. DINO-Foresight's official PCA is non-whitened but **does standardize feature channels first**. Non-whitening must not be conflated with leaving original channel scales unchanged. Its dense masked future predictor is useful precedent; our sparse `(view, xy, time)` reader and explicit temporal-difference loss are new adaptations.
3. The selected FPO PyTorch GridWorld path uses action-coordinate **mean** squared error. Paper Eq. 8 writes a squared L2 **sum**. With 32 raw coordinates, these imply a factor of 32 in log-ratio before clipping. This is a substantive contract choice, not a harmless reporting difference.
4. The same official path draws uniform MC times and fresh Gaussian CFM noise, stores them with old losses, clips each MC old-minus-current difference before averaging and exponentiating. Other directories in the official repository implement different time directions and ratio estimators. Do not blend them.
5. Original VLA BC uses a Beta-derived time sampler. Keep BC unchanged and use a separate, registered FPO time/noise stream. Group-relative bandit advantages, the BC anchor, and navigation rewards are our adaptation of FPO, not the GridWorld algorithm verbatim.
6. The DiT's dropout creates an additional migration issue: identical old/current weights do not guarantee ratio one if old loss uses eval and current loss uses train. Explicitly choose and test coupled stochastic realizations or a declared common deterministic policy-loss evaluation mode before formal training.

## Evidence identities and read scope

| Source | Fixed code revision | Full-text material actually inspected |
|---|---|---|
| DINOv3 | `6876159a11b4df116f30f667f8c9888617df0751` | Paper architecture/RoPE, Gram anchoring, high-resolution and distillation sections; corresponding feature, patch and hub implementations |
| DINO-Foresight | `d98f765d2b743ec90fb38bc657a39c07c7ca4ab0` | Sections 3.1–3.4, Eq. 1–2, experimental setup, Appendix A.1 and loss/resolution ablations; PCA, preprocessing, dense predictor and loss source |
| FPO | `418c2554f7cd22d52e14c07d951280929d73bf2f` | Sections 3.1–3.5, Eq. 1–20, Algorithm 1, Appendix A.1.1–A.1.4 and limitations; full selected PyTorch rollout/loss/update callchain and contrasting JAX paths |
| ResWorld | `a5ae973001dd6001d95b3389020a9e263d73a56e` | Sections 3.1–3.5, residual construction and fusion equations; BEV alignment and residual-token implementation |
| DFQL | `8b6b15803cc1e5605ad1524553a2c95e103b0bad` | Official paper retrieved; actor implementation inspected. Full paper derivation and local execution not yet verified |
| RFO | No official code binding | Method sections read from full paper; official implementation not located. Conditional branch remains blocked on code verification |
| OpenFlamingo | `655f693fbfa04cd6e9a987d960654624d48917cf` | Authors' open implementation read; paper full-text verification not yet complete. Not DeepMind's official Flamingo implementation |

PDF hashes, page counts and local copies are in `public_sources/PAPERS.json`; exact inspected source hashes and links are in `SOURCE_CODE_MANIFEST.json`. The inspected method sections are full-text evidence; this report does not claim every bibliography or experiment appendix was independently reproduced.

## DINOv3: portable feature contract

Primary sources: [paper](https://arxiv.org/abs/2508.10104), [fixed feature implementation](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/models/vision_transformer.py), [fixed hub configuration](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/hub/backbones.py).

`DinoVisionTransformer.forward_features_list` returns `x_norm_patchtokens` after final patch LayerNorm and after removing one CLS plus `n_storage_tokens`. `get_intermediate_layers(norm=True, reshape=True)` uses the actual image height and width divided by patch size. For the official ViT-L/16 factory, width is 1024, depth 24, heads 16 and storage-token count 4. These are source facts, not evidence that a particular local checkpoint is this model; bind the actual teacher weights and config separately.

`norm_layer_dict['layernormbf16']` is `nn.LayerNorm(eps=1e-5)`, whereas `layernorm` uses `1e-6`. The alias alone does not prescribe weight or autocast precision. The model's Gram objective uses normalized features during teacher training; this is not a reason to L2-normalize inference targets or temporal deltas.

`PatchEmbed.forward` uses strided convolution. Divisibility assertions are commented out upstream, so our wrapper should reject incompatible image dimensions rather than silently truncate a border. With width×height conventions explicitly recorded, 384×288 gives a 24×18 native patch grid; code tensor order remains height×width. Never infer a square grid from token count.

Project adaptation: native-grid sparse targets, W48 versus W96, output-query independence, and a navigation-specific teacher cache. Required controls are P0/P1/P2/P3 with fixed population, actual grid-coordinate checks, CLS/storage exclusion, teacher eval/freeze, online/cache equality and observed precision. Literature does not establish that 96 W tokens or a particular sparse sample count is optimal.

## DINO-Foresight: PCA and temporal boundary

Primary sources: [paper](https://arxiv.org/html/2412.11673v2), [PCA fitting](https://github.com/Sta8is/DINO-Foresight/blob/d98f765d2b743ec90fb38bc657a39c07c7ca4ab0/pca.py), [feature/loss path](https://github.com/Sta8is/DINO-Foresight/blob/d98f765d2b743ec90fb38bc657a39c07c7ca4ab0/src/dino_f.py), [attention](https://github.com/Sta8is/DINO-Foresight/blob/d98f765d2b743ec90fb38bc657a39c07c7ca4ab0/src/attention_masked.py).

`pca.py` reads the training split, concatenates features, computes each channel's mean and standard deviation, standardizes channels, and fits sklearn PCA without whitening. `pca_transform` applies the same channel statistics, PCA centering and component projection. Its inverse restores PCA mean and original channel scales. A centered-only PCA on raw DINOv3 is a legitimate new choice, but it is not the exact upstream preprocessing. Lock this choice before comparing 256/384 dimensions, fit on allowed training scenes only, and reject zero/invalid standard deviations rather than propagating NaNs.

The original predictor operates on dense multi-layer DINOv2 feature grids with future positions masked. Temporal attention connects corresponding spatial positions, spatial attention operates inside each frame, and the primary objective is SmoothL1 (beta 0.1). The code also supports MSE and L1. High-resolution experiments distinguish native-resolution feature crops and resolution adaptation; they do not prove our VLA sparse readout works.

Our reader conditioned on view/xy/physical time, MSE choice, single-layer DINOv3 targets and explicit delta supervision must be labeled project adaptations. Cache absolute teacher features under one transform; compute deltas in that same fixed coordinate space. Do not separately normalize each delta to unit length, which would discard magnitude. Same image-grid position across moving cameras does not identify the same physical point. Temporal differencing may capture appearance, ego-motion and object-motion together; it is not a verified motion-vector or object-dynamics target.

Minimum controls: absolute-only versus absolute+delta at identical history/input; shuffled future time; copy-current baseline; masks and actual target dt; train-only PCA identity; aligned sampling positions across times; inverse/projection tests; full versus sparse subset reduction. Do not attribute lower auxiliary MSE alone to improved planning.

## FPO: selected PyTorch reference contract

Primary sources: [paper](https://arxiv.org/html/2507.21053v2), [policy](https://github.com/akanazawa/fpo/blob/418c2554f7cd22d52e14c07d951280929d73bf2f/gridworld/models/diffusion_policy.py), [update](https://github.com/akanazawa/fpo/blob/418c2554f7cd22d52e14c07d951280929d73bf2f/gridworld/models/fpo.py), [defaults](https://github.com/akanazawa/fpo/blob/418c2554f7cd22d52e14c07d951280929d73bf2f/gridworld/main.py).

Let `a[b,d]` be a detached actual old-policy rollout action, `eps[b,n,d]` independently drawn CFM Gaussian noise, and `t[b,n]` uniform on `[0,1)`. CFM draws are distinct from the initial noise used to generate `a`. Old and current computations share the exact saved `a`, `eps` and `t`.

The **selected official PyTorch implementation** is:

```text
x_t[b,n,d] = (1-t[b,n]) * eps[b,n,d] + t[b,n] * a[b,d]
target[b,n,d] = a[b,d] - eps[b,n,d]
ell_theta[b,n] = mean_d((v_theta(obs[b], x_t[b,n], t[b,n]) - target[b,n])^2)
z[b,n] = clamp(ell_old[b,n] - ell_theta[b,n], -3, 3)
rho[b] = exp(clamp(mean_n(z[b,n]), -3, 3))
L_policy = -mean_b(min(rho[b]*adv[b], clamp(rho[b],1-clip,1+clip)*adv[b]))
```

The second log clamp is redundant after the first plus mean, but belongs to the audited source. Upstream default PPO clip is 0.2 and GridWorld default MC count is 50. The candidate group size of four in our plan does not determine MC count. Choose/register MC count against real cost and variance; do not silently transplant 50 or confuse it with four policy rollouts.

Paper Eq. 6–9 averages CFM losses over MC draws and then exponentiates; its Eq. 8 writes L2-sum over action coordinates, and it does not specify the implementation's per-MC clamp. Selecting the PyTorch reference therefore requires explicitly recording mean reduction and its clamp as the reproduction contract. Switching to sum is a different scale. A future arbitrary temperature or division by raw dimension must not be introduced invisibly to make clipping statistics look favorable.

Section 3.4 and Appendix A.1 express general epsilon-prediction objectives with time weights. Uniform-time OT velocity MSE already represents the selected weighting. Do not multiply the epsilon-space `exp(-lambda/2)` factor onto velocity MSE a second time. The ratio remains a surrogate rather than exact action likelihood or exact KL; small-MC bias and clipping matter. Its paper's direction argument is not a blanket guarantee that every finite-MC, clipped, navigation adaptation is unbiased.

The repository contains materially different variants: `playground/src/flow_policy/fpo.py` reverses interpolation direction and supports either mean-before-exp or a per-MC-exp branch; `phc/puffer_phc/policy.py` also has alternate time distributions. These are not interchangeable with the chosen GridWorld contract. Its actor is a small deterministic MLP with no dropout.

### VLA adaptation boundary

The source `1493deda247107efe076b9294863a6753391298a`, `GR00T_ActionHeader.forward`, uses the same noise-to-action velocity target but returns a scalar mean over batch/time/channel and uses its existing Beta-derived BC time sampler. `predict_action` performs Euler integration with the existing action representation and ten steps. A policy-gradient adapter needs access to unreduced predicted velocity and explicit FPO time/noise; it must not use a GT endpoint as the action receiving reward. Retain original normalizer, ten-step inference, BC weight/time distribution and output conversion.

The current implementation computes BC time as `(noise_s - Beta_sample)/noise_s`; preserve this source behavior rather than silently substituting uniform time throughout training. Contextual-bandit group rewards, fixed reference BC, event-shaped navigation rewards, stop-gradient observations/rewards, and on-policy refresh rules are additions that must be registered and tested independently.

Before formal use resolve the DiT stochastic-function contract. Same old/current parameters plus different dropout masks can yield a nonunit ratio. Saved old losses are valid only for their exact observation/action/noise/time and policy mode. Confirm initial ratio one under the chosen production policy-loss mode, gradient direction for both advantage signs, finite log-ratio and clipping rates, weighted global DDP reduction, no gradients through true rewards, actual sampled actions, version-stale rollout rejection and checkpoint/RNG/collector recovery.

### Executed reference test

`reference_fpo_cpu.py` imports the actual upstream `DiffusionPolicy.compute_cfm_loss`; it does not replace the official function with a mock. It ran on CPU with torch 2.5.1+cu124, float64, 3 scenes, 4 MC draws and 32 action coordinates.

`FPO_REFERENCE_CPU.json` records exact match to the hand-derived coordinate mean, exact unit ratio when old=current, the expected CFM decrease for positive advantage and increase for negative advantage, and no extra gradient through detached reward. An analytic counterexample distinguishes `exp(mean(delta))=1` from `mean(exp(delta))=1.5430806348`. This validates the reference arithmetic only. Real DiT dropout, VLA inputs, reward simulation, DDP, optimizer recovery and formal policy improvement remain NOT_RUN by this agent.

## History and conditional extensions

ResWorld: [paper](https://arxiv.org/html/2602.10884v1), [fixed implementation](https://github.com/mengtan00/ResWorld/tree/a5ae973001dd6001d95b3389020a9e263d73a56e/projects/mmdet3d_plugin/resworld). It aligns historical BEV into a common current frame before applying shared selection and taking temporal latent differences. The implementation's ordered subtraction depends on frame ordering. This is supporting evidence for testing temporal residual information, not justification to call unaligned camera-grid subtraction equivalent or to restore its trajectory-refinement architecture.

OpenFlamingo: [authors' implementation](https://github.com/mlfoundations/open_flamingo/blob/655f693fbfa04cd6e9a987d960654624d48917cf/open_flamingo/src/helpers.py). `PerceiverResampler` uses learned latent queries, media+latent K/V, residual attention/FFN and final LayerNorm; it flattens frame and patch axes within a media item. Its helper has no input padding-mask argument and does not mix the media axis. A small history reader with missing-frame mask, ego transforms, 2 layers and 16 latents is an adaptation, not a line-for-line reproduction. Full paper audit for this optional choice is still pending; do not claim a mature implementation already supplies our all-missing behavior.

DFQL: [official paper](https://proceedings.mlr.press/v306/cao26f.html), [actor source](https://github.com/2012060/Direct-Flow-Q-Learning/blob/8b6b15803cc1e5605ad1524553a2c95e103b0bad/agents/dfql.py). The actor source stops gradients through intermediate Euler states and applies a terminal critic action-gradient to collected velocities with time-dependent `1/(steps-i)` weights, together with BC. It includes robot action clipping, optional Q normalization and Bellman critic training. These must not be ported mechanically into navigation's complete-trajectory contextual bandit. Full derivation/local reference execution remain pending, so this is not an approved alternative to FPO.

RFO: [paper](https://arxiv.org/html/2602.03501v1). The method uses differentiable rollout, stability flow matching on recent actions and exploration flow matching on sampled bounded actions. A naked BPTT loss is not the complete method. No author-confirmed implementation was bound in this audit; status is BLOCKED_CODE_DISCOVERY for code-level migration. Conditional branches do not block the verified FPO reference work.

Video/geometry teachers remain NOT_RUN / NOT_AUDITED in this wave. No absence of evidence is being converted into a negative scientific result.

## Handoff and approval boundary

- SpaceTime received exact DINO feature/PCA contracts and the sparse/delta adaptation boundary.
- HistoryData received OpenFlamingo's exact helper contract and missing-mask/media-axis risks.
- Lead received FPO noise/time/mean/clamp details, source variants, the original BC boundary and dropout risk before code integration.
- The method evidence supports implementation and bounded discardable tests. Correctness and Systems must review the final integrated SHA/config/input/precision/topology; this document provides neither approval.
- No source worktree, existing checkpoint, scorer, training process or resource policy was changed.
