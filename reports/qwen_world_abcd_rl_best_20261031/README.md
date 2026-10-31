# Own-best-parent ABCD reinforcement learning

The user authorized starting RL now from each architecture's currently best complete NAVTEST IL checkpoint, independently of the original100k DEV-only campaign gate. The four protected deployment parents are A20k (85.1231263), B16k (84.3067954), C12k (85.3869184), D12k (86.0014574). These are initialization scores, not new RL results. Their selection used NAVTEST; it is explicitly not a blind test selection.

A/B/C start with source PPO/W3 and SafetyEP credit. The trainable scopes follow their different deployment graphs: A action attention, B shared predictive/action attention, C action/world attention and world-KV bridges. Frozen C160 inputs may be cached; trainable B/C prefixes must be rebuilt under gradients. Reference modules are independent, while immutable frozen base parameter storage is shared to avoid a redundant model allocation. Tiny real-architecture comparisons establish identical chains, four losses, gradients and Adam updates against independent full reference copies.

D follows the original guide's [AUGUstus](https://arxiv.org/html/2610.16985v1) native language-layer action denoising design. Its first action-only JaT-inspired trial uses joint24 Gaussian ratios, PPO clip0.2, native Gaussian IL KL0.001, LR1e-6,100-step warmup and constant LR. Gaussian innovations are unclipped. The source rigorous eta0.1 lateW3 sampler and joint24 reduction are declared adaptations, not the paper's complete kernel or official coordinate-mean implementation. No CoT reinforcement or published external score is claimed for this run.

All runs use seed42 and the complete legal TRAIN90,890/147logs, physically disjoint from DEV1,696/15logs and NAVTEST12,146/44logs. The legacy103,288 population also contains DEV and embargoed TRAIN logs; it cannot be used wholesale under the ABCD split. Each run has fresh Adam/update0 and its own frozen selected IL reference. A/B/C LR5e-5, warmup100 then constant; source physical anchor beta50 retained.

Initial finite gates are1/5/10k for A/B/C and1/5k for D. The config50k is an identity ceiling, not an automatic allocation. The owner natively resumes Adam/RNG/cursor and pending rollout across gates. DEV and full NAVTEST are evaluated asynchronously; no partial mean is published. Further training requires an evidence-based new registration, with NAVTEST every2k beyond10k.

Live artifacts: `/mnt/project/qwen-world-abcd-rl-best-20261031`.
Persistent report: `/mnt/project/qwen-world-rl-optimization-20261030/report.html`, ABCD first,30min plus milestones.

Resource authorization adds the exact newly granted hosts `training-vlawm-zt4` and `training-vlawm-zt5` as temporary until withdrawal. At launch zt4 is idle8×A800; zt5 SSH is refused. Existing IL/RL jobs and guardians remain untouched. Physical GPU locks, known-owner checks and measured headroom apply to every new child.
