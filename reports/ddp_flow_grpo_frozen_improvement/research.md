# F-only failure investigation: primary sources and falsifiable changes

2026-09-18. Original F-SFT step100000 only. U and historical long training remain stopped.
Repository locks and downloaded source SHA256s are in `repositories.json` and
`source_manifest.json`; a downloaded file is not evidence that its method works here.

## What is actually established

The previous complete seed42 development evaluation (1696 scenes / 16 whole logs)
fell from F-SFT 93.4827 to step200 91.8825 **v1 PDMS**, and from 93.6295 to
91.8215 **v2 EPDMS**. Progress increased while v1 TTC and v2 two-frame comfort fell.
See `../ddp_flow_grpo_world16/step200_diagnosis/README.md` for component scores,
paired log bootstrap, missing-comfort handling and protocol identities. These are
development results, not complete Navtest or a five-seed performance claim.

The 3101 readable historical F reward groups include 1271 constant groups. Of
1830 nonconstant groups, 1521 (83.11%) vary only in progress, with the other
returned safety/comfort components equal to one. Nonconstant group standard
deviation median is 0.00251635. Three other cache files are unreadable and explicitly
listed in `historical_reward_summary.json`; neither zero filling nor silent exclusion
is used to call this a complete-cache audit. Their failures do not establish a cause
for the score regression.

All 32 scenes / 256 saved candidates from the original first two behavior batches
were rescored with genuine v1 caches constructed from raw scenes. There were zero
v1/v2 ordering reversals among 511 strictly comparable pairs. One candidate failed
v1 TTC; its saved v2 advantage was not positive. This narrow early sample does not
establish late-policy alignment. The attempted late-buffer audit encountered EIO;
its original log remains `runs/frozen_rl_research/behavior_audit_late.log`.

An actual arithmetic defect was reproduced: constant nonbinary rewards can get
nonzero advantages from FP32 mean residuals divided by tiny epsilon. For example,
eight identical 5/7 values produce -0.05625 on CPU. **The same G=8 CUDA cases did
not reproduce it**, and none of the first 32 saved training groups had constant,
nonzero advantages. It is a portability/numerical correctness fix, not proof of
the historical GPU regression's cause. Centering in FP64 after subtracting an
observed group value fixes this without changing the network precision.

## Source comparison and transfer limits

| Primary implementation/paper | Relevant evidence | Decision for DDP |
|---|---|---|
| [Flow-GRPO](https://github.com/yifan123/flow_grpo/tree/879042cf5707f8b90daa98d147d7deac2317c5da) | `stat_tracking.py` uses FP64 rewards; `global_std=True` keeps a per-prompt mean and a whole-batch denominator. Many released presets enable it. Gaussian reference loss and no-update ratio checks are explicit. | Implement selectable global **behavior-batch** denominator over all ranks and accumulation microbatches. Retain population variance, our epsilon and fixed saved advantages. Image-model learning rates/noise are not transferred wholesale. |
| [Dr. GRPO](https://github.com/sail-sg/understand-r1-zero/tree/dfca49dd460ee7cc8e4a5a162c876a7fd6993b87) | `train_zero_math.py` explicitly removes division by group std to avoid difficulty reweighting. | Supports the concern that tiny progress differences receive disproportionate weight. DDP has fixed action/chain lengths, so the paper's variable-answer-length correction is not relevant. |
| [TRL](https://github.com/huggingface/trl/blob/21f1d28217ed75a2237df1aef2a3082703cc4163/trl/trainer/grpo_config.py), [PPO Lite](https://arxiv.org/abs/2508.08221) | Exposes group/batch/no reward scaling and distinguishes their effects. | Use as an independent mathematical cross-check, not a framework migration. Global batch includes constant groups and between-scene reward variation. |
| [SimWAM](https://github.com/H-EmbodVis/SimWAM/tree/68b426c162827cb7701396895dbb3572d29f3420) | Released task: action-DiT LoRA r16/alpha32, G8, K10, noise .1, clip .02, 4 inner updates, LR5e-5, BC .1, EP:TTC:comfort 10:5:2. Action noise is intentionally lower than image Flow-GRPO. | These are **not** full 2.23B-parameter updates. Its error-to-zero fallback, truncated noise and different reward weights are not adopted. LoRA is now user-authorized as a later alternative, not automatically a remedy for absent reward variation. |
| [ReCogDrive](https://github.com/xiaomi-research/recogdrive/tree/6b8d8f5e01346c71094651c81dcaf66405dbc04e) | `recogdrive_agent.py` defaults to DDIM; planner supports flow/DDPM/DDIM. GRPO uses group-normalized rewards, diffusion std floors, and per-denoising-step weights. | DDP also uses a DiT: DiT architecture and flow-vs-diffusion sampling objective are different axes. DDIM noise/floor constants cannot be copied into DDP's score-corrected SDE without changing its distribution and likelihood. |
| [DPPO](https://github.com/irom-princeton/dppo/tree/cc7234ad7ff39a8f32de3af903606723a16f0648) | Diffusion-chain policy gradients, denoising-dependent weighting, advantage normalization and optional behavior-cloning regularization. | Useful checks for chain reuse and retention; its log-prob clipping/denoising discounts are not silently installed in our objective. |
| [ReinFlow](https://github.com/ReinFlow/ReinFlow/tree/e722e151bed767f3ffef47527cf697f2358af55d), [paper](https://arxiv.org/abs/2505.22094) | Learns bounded exploration noise along flow transitions, with tractable conditional Gaussian likelihoods and optional regularization. | A principled alternative if fixed isotropic noise is ineffective. It changes the exploration policy; would require explicit covariance/std contracts, full old/current/reference validation and a separate experiment. |
| [AutoVLA](https://github.com/ucla-mobility/AutoVLA/tree/ba34eed74ce6729e7986592d0e66cbaca397b4fa) | Released NuPlan preset freezes vision, trains language via LoRA r8, LR3e-5, KL .04, categorical action tokens and CoT penalties. | Its learning rate is not evidence that the current full-parameter flow actor should use 3e-5. Different action probabilities and reward scaling. |
| [DanceGRPO](https://github.com/XueZeyue/DanceGRPO/tree/15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344) | Documents ratio=1 sanity checks, diffusion-step accumulation and stabilization concerns for image/video optimization. | Cross-check recomputation and gradient scale; image-specific LR/EMA recommendations are not evidence of driving-score improvement. |
| [FlowCPS](https://github.com/IamCreateAI/FlowCPS/tree/b402c7a3d5b2539e6c42fed6c6c042bab9a1be05), [paper](https://arxiv.org/abs/2509.05952) | Addresses coefficient changes in SDE sampling. Released patch uses negative squared error as its surrogate log-prob, omitting Gaussian variance normalization. | Useful train/inference mismatch hypothesis. **Do not copy this surrogate and describe it as our exact Gaussian probability.** ODE comparison must use the same initial noise. |
| [Flow-Factory](https://github.com/X-GenGroup/Flow-Factory/blob/7d5cb9a035e2aafc683e51dffbb53ede47d2fa0b/guidance/algorithms.md) | Compares Flow-SDE, constant-noise SDE, CPS and mixed ODE/SDE approaches; discusses fixed batch layouts and train/inference consistency. | A broader source map, not validation of all algorithms on DDP. No new framework, mixed-step or algorithm bundle is installed. |
| [NAVSIM](https://github.com/autonomousvision/navsim/tree/0a380a9063d7162ec93d0f51e9990ebac585f720) | v2 single-scene components, human penalty filtering, final aggregation and adjacent-frame comfort differ from v1 PDMS. | Preserve vendored official evaluation. Training reward remains explicitly v2 single-scene without two-frame comfort. Neither raw `pdm_score` nor its file name is v1 PDMS. |
| [WAM-Flow](https://arxiv.org/abs/2512.06112) | Discrete-flow driving GRPO uses a mean-centered group baseline, KL anchoring, and balanced EP:TTC:comfort weights 5:5:2; RL budget .5 epoch/103k scenes. | Independent driving motivation for checking normalization and component balance. It is not the same continuous action distribution or backbone. |
| [Colored Noise PPO](https://arxiv.org/abs/2312.11091), [Lattice](https://arxiv.org/abs/2305.20065) | Structured noise can produce more coherent exploration than independent actuator perturbations; Lattice explicitly models full Gaussian covariance. | Candidate future remedy for waypoint jitter rather than meaningful trajectory variation. Any adaptation needs covariance-correct drift, sampling, log-prob and KL, not post-hoc smoothing of candidates. These papers do not establish a DDP gain. |
| [FINO](https://arxiv.org/abs/2602.18117), [FPO for VLA](https://arxiv.org/abs/2510.09976) | Study exploration beyond imitation distributions and alternatives to exact likelihood-based flow RL. | Reviewed as alternatives if the measured bottleneck requires them. No critic, new architecture or likelihood surrogate is silently added. |

## Controlled experiments

The old G8 recipe was run from the original F checkpoint for eight actual updates,
8 GPUs / accumulation2 / global scene batch16, with saves at4 and8. Both inner
epochs reuse each behavior batch. This is a diagnostic, not continuation of the
degraded step200/300 model.

User subsequently authorized G16. New configs are
`configs/flow_grpo/frozen_research_g16_{group,global_batch}.yaml`; they retain
noise .1, K10, chunk1, inner2, LR1e-6, Adam(.9,.95), wd.001, clipping .02,
KL .01, action SFT .1 and all original nonvisual trainable parameters. They are
bounded to eight updates and **not released for long training**. The two arms
differ only in advantage normalization. Global scaling also changes RL strength
relative to fixed KL/replay; an improvement cannot be attributed exclusively to
scene weighting without further loss-balance evidence.

Before those comparisons, the user's diversity requirement is tested using
64 fixed, evenly spaced training-list positions (61 logs; zero dev overlap),
G16, ODE and SDE noise [.05,.1,.2,.3]. Every trajectory, reward and SDE chain is
saved. The first8/all16 comparison uses a nested candidate bank, not a claim that
separate G8 and G16 RNG layouts are identical. Seed42+token fixes initial noise;
all SDE strengths use the same initial and innovation random numbers.
No best-of-N candidate is deployed or substituted into training. The diagnostic
max reward only quantifies available signal. No threshold is a performance pass.

Full trajectory collapse, centimeter-scale jitter, reward saturation, objective
mismatch and excessive parameter drift are distinct hypotheses. Increasing G or
switching to LoRA alone does not resolve all of them. Model/optimizer parameter
changes remain separate, explicit experiments; published v1/v2 evaluators stay fixed.
