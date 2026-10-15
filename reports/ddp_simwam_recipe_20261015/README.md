# DDP: SimWAM public recipe transfer

Status at 2026-10-15 05:45 UTC: both formal eight-GPU runs are updating from
the same SFT/LoRA initialization. Full TRAIN screening and the fixed hard list
are complete. Implementation, real-model acceptance, exact eight-GPU recovery,
and paired SFT deployment evaluation are available.
**Neither formal 20,000-update run is complete. Fixed5k evaluation now shows
native single-ODE gains on complete DEV, with a clear TTC cost for source
reward. Final Navtest is still pending.** This report is updated by stage.

[The complete5k result](interim_ode_update005000.md) is DEV native PDMS
93.6192 SFT →94.0763 source /94.3121 canonical. Source adds16 TTC violations
and recovers1; canonical adds1 and recovers1. New joint violations on the
fixed SFT-safe population are20 /2. These intermediate results do not select
the final checkpoint or establish safe generalization to Navtest.

The 05:33 analysis uses sealed cost snapshots at source17,600 / canonical11,000
updates. The [matched first10k analysis](interim_training_10k.md) includes
20,000 fresh scene groups per variant and shows >99.6% nonzero-PG groups, but
also a progress/TTC tradeoff in stochastic training samples. These are not
deployment scores. A missing SSH alias stopped the first milestone controller;
[the recovered queues](controller_recovery.md) now run from the CPU-only local
control host, with inference/scoring still on authorized Tsinghua workers.
Both update5000 DEV1696/TRAIN512 evaluations are complete. The vla-zt2 withdrawal is enforced;
the current two training hosts are unchanged.

Branch: `feature/ddp-simwam-recipe-transfer-20261015`, starting from
`90782b6b5a10a77541c22b29f90b83d7309d7081`. Old C/N experiments were stopped at
recoverable native boundaries by user instruction: C update 4068 / raw 50812,
N update 6990 / raw 87700. All 58 checkpoint files were verified; their
controllers remain stopped. G-series and unrelated jobs were not stopped.

## What is transferred

The source is [SimWAM at the locked commit](https://github.com/H-EmbodVis/SimWAM/tree/1433ee7e06a62bd8b89da74455df0d82bda7b726).
The actual Hydra result is [resolved_source_recipe.yaml](resolved_source_recipe.yaml).
[upstream_lock.json](upstream_lock.json) records source and vendor hashes.
[recipe_delta.json](recipe_delta.json) traces 93 resolved fields, active and
inactive branches, runtime consumers, and three patch categories. The copied
LoRA, scheduler and epoch sampler files are unchanged; required action and
trainer methods are extracted with their function bodies unchanged. The MIT
license is retained. No Wan/SimWAM model weights were downloaded.

Both variants use the original F-SFT checkpoint
`9a26685aa3838a2e1b89ab2d92997664afde4b259fed6683851f42781ad16eb4`.
All original model weights are frozen. Only attention LoRA A/B train:
96 actual self/cross-attention q/k/v/o projections, 192 tensors and 4,718,592
parameters, r16 / alpha32 / dropout0. Original Qwen, history module,
qwen_proj, action encoding, 24-block DiT topology, time buckets and decoder
are preserved. Inference merges LoRA into the original deployment class.

The shared recipe is G8/K10, shifted sigma grid (shift5), last three stochastic
steps `[7,8,9]`, rigorous score-corrected SDE noise level 0.1, std floor 0.001,
innovation clamp ±5, relative elementwise logprob clamp `[peak-7,peak]`,
dimension-mean PPO clip 0.005, discount1, global **raw-reward** population std,
group centering, tied-group PG mask, and four actual AdamW optimizer updates
per fresh rollout. There is no DAPO resampling, learned noise, cost multiplier,
GT replay, BC, adaptive KL, or entropy term.

The permanent SFT anchor is the weighted MSE between random-transition means
at the same saved states, beta50. It is not a final-trajectory KL. The DDP
channels are `[x_norm,y_norm,sin(relative_heading),cos(relative_heading)]` at
poses relative to current t0, **not consecutive waypoint displacements**.
The logical `[.2,1,1]` anchor is mapped to `[.2,1,.5,.5]`, denominator `H*2.2`.
This preserves logical group weights, not equivalence to heading-angle MSE.

| Configuration | Training readout from one actual simulation | Official evaluation |
|---|---|---|
| R_SOURCE | NC×DAC×(10 EP+5 TTC+2 Comfort)/17 | Canonical v1.1 PDMS |
| R_CANONICAL | NC×DAC×(5 EP+5 TTC+2 Comfort)/12 | Canonical v1.1 PDMS |

The resolved config diff contains only the EP weight and variant name.
The existing batched simulator is reused, while each candidate's official
aggregation remains `[fixed official reference, that candidate]`. Both reward
readouts reuse the same simulated submetrics. Infrastructure failures abort;
they do not become reward zero. Legal bad trajectories remain in the group.

## Version and execution boundaries

[Paper v1](https://arxiv.org/abs/2608.07468v1) reports 91.5 PDMS; the locked
GitHub README reports 91.9. The resolved default has 20,000 updates while a
README command has 170,000. The HF model-card identity and separate license
metadata are in [version_boundaries.json](version_boundaries.json). There is
no verified original run identity linking the current default to the released
91.9 weight. Those versions are not spliced together into a claimed recipe.

An important actual-source detail: YAML declares max_grad_norm=1, but its
default DeepSpeed JSON omits gradient_clipping. The optional bridge from YAML
is disabled, and Accelerator's DeepSpeed clipping call does not apply torch
clipping. A real source-engine update confirmed effective clipping **0.0**.
The transferred configurations explicitly preserve that effective value;
adding clipping1 would be a separately named algorithm change.

DDP parameters, latent, velocity, transition means, probabilities and Adam
state use FP32; small advantage statistics use FP64. Source BF16 latent
storage is not claimed identical. The platform is native torch DDP with
FP32 LoRA AdamW instead of source Accelerate/ZeRO1. Four optimizer steps,
equal rank averaging, exact inner-boundary recovery, and output-level
numerics are tested. One-card serial vs eight-card Adam parameters are **not
bitwise equal**; the measured differences are disclosed in
[validation.md](validation.md).

Source clipped innovations and clipped density proxies are deliberately
retained. Shared LoRA also changes the deterministic prefix, while gradients
are computed on the saved stochastic transitions only. This is the public
empirical estimator, not an unbiased full mixed-chain policy gradient.

## Data, resources and fixed budget

Existing log-isolated manifests: RL_TRAIN 101,592 scenes / 1,176 logs;
RL_DEV 1,696 / 16; Navtest 12,146 / 136. RL_DEV is unseen by RL updates,
not necessarily unseen during SFT. Historical Navtest observations remain
acknowledged.

The hard list uses this DDP SFT's **single original uniform ODE**, seed2101,
canonical score<0.9 across the complete TRAIN population. Compatible previous
single-candidate scores are reused only after exact output comparison; G16
mean/best/worst census statistics do not define membership. The list was fixed
before training and is shared by both variants: **22,017 scenes / 1,063 logs**,
from all 101,592 TRAIN scenes. All payload hashes were checked, with no missing
scene. The manifest identity is
`8fd5b7b027b9e4d6d5df9e83a2cffd69ebe02dab176e1d4d5017091fe7aff39b`;
[hard_population.json](hard_population.json) contains the source, actual file
hash, population counts and verification receipt without publishing tokens.

Of the hard scenes, 20,971 pass NC/DAC/TTC but have lower quality/progress;
373 fail TTC only, 637 DAC only, 22 NC+TTC, 10 DAC+TTC, and 4 NC only.
This is a **single fixed-seed SFT census**, not a G8/G16 tie-group census or
proof of safety across all noise. The difficulty filter still predominantly
selects low-progress cases, so its high PG availability alone cannot establish
better safety learning.

The already corrected raw-condition cache is sealed at identity
`80a2dd4b417a3154cbbf575044daea586d945b54e014eee22269d7bd48d39e0b`.
This recipe additionally caches the permanently frozen qwen_proj output;
that separate cache is never valid for old full-head experiments. In this
installed torch runtime, original CUDA FP32 autocast really casts the BF16
initial root; disabling autocast is not equivalent. The native baseline
retains the original execution path and validates exact original outputs.

[experiment_registration.json](experiment_registration.json) fixes, per
variant, training seed42, 8 GPUs, 20,000 actual optimizer updates, 5,000 fresh
global rollouts, 40,000 scene exposures and 320,000 newly generated candidates.
Checkpoint interval is 100 updates, separately from evaluation milestones.
The source 100-update warmup and constant LR5e-5 are preserved.

Only the four permanently authorized Tsinghua hosts are used. The latest
user explicitly permits sharing GPUs. Completed screening used 48 workers on
24 GPUs; paired SFT DEV/TRAIN evaluation used 16 workers on the remaining eight.
The deterministic subpartition changes execution concurrency, not scene
membership, seeds or scientific budget. Unknown GPU owners are preserved.
Withdrawn temporary hosts are excluded from tasks and automatic retries.

Formal source is the immutable snapshot at
`12b8e0d75691d7bed04ab02072020e130f711a3b`. R_SOURCE runs on
`training-vlawm-tsinghua-zt4`; R_CANONICAL runs on
`training-vlawm-tsinghua-zt3`. Registered milestone evaluation uses
`training-vlawm-tsinghua-zt` and `training-vlawm-tsinghua-zt2`, respectively.
[formal_dispatch.json](formal_dispatch.json) records real commands, live
admission, PIDs and source archive identity. Both first behavior adapter hashes
are `7b7f31eb8e320cd5248abb75f758a902c8a04907d9fee2e63741ea978a0d5099`;
the first source/canonical rollout scores match and only the training readout
differs. Four-inner updates keep the original behavior chain and probabilities.

Early snapshots are available, with different completed exposure and therefore
**not a matched efficacy comparison**:

| Run | Completed updates | Fresh scene exposures | Nonzero-PG groups | Progress-only groups |
|---|---:|---:|---:|---:|
| R_SOURCE | 500 | 1,000 | 99.40% | 710/1,000 |
| R_CANONICAL | 304 | 608 | 99.51% | 447/608 |

These counts include all eight ranks, count each fresh group once, and do not
count four-inner reuse as four sets of candidates. No global collection in
these snapshots was anchor-only. Nonzero advantage is update availability,
not evidence of a useful driving improvement. Full/tied/zero categories,
per-inner ratios and the raw-reward std are in `R_SOURCE_early_training.json`
and `R_CANONICAL_early_training.json`.

At the first sealed 400/300-update timing points, end-to-end training rates
were approximately 1.17/0.73 updates/s for source/canonical. Linear remaining
training estimates were 4.65/7.48 hours, excluding evaluation; these are measured
host-specific early projections, not promises or algorithm speedup claims.
Updates dominate the measured time. Scorer time is recorded separately from
scoring wait, and shared GPU-wall is not exclusive physical GPU compute.

## Evaluation and results still required

Every fixed stage 0/5k/10k/15k/20k has two separate single-candidate readouts:
`E_NATIVE` (unchanged original uniform10 ODE) and `E_SHIFT5` (shift5 ODE).
Both use the same raw observation, seed2101, postprocessing and v1 scorer;
neither selects between candidates. Both have their own matched SFT baseline.
The fixed final selection rule is each variant's update20000 endpoint, not
the best intermediate checkpoint. Full Navtest waits for endpoint identity
locking. No extra training or inference seeds are scheduled.

The original final controller dispatch is preserved in
[final_queue_dispatch.json](final_queue_dispatch.json). Its idle controller was
replaced by local CPU-only administration from commit
`62790cbc0df4acaaee3faa77427751443eca5ea2`, after a successful transport check to
all four worker hosts. [Recovery evidence](controller_recovery.md) records this
routing-only change; inference retains the original immutable evaluation source.
It waits for both completed training budgets and all DEV stages, locks the
actual endpoint weights, then evaluates SFT and both endpoints on all12,146
Navtest scenes using the same frozen evaluation source. Live GPU inventory and
physical UUID checks precede admission. It neither restarts training nor
extends the experiment. No Navtest candidate has been generated yet.

Completed SFT controls (single seed2101):

| Population | E_NATIVE PDMS /100 | E_SHIFT5 PDMS /100 |
|---|---:|---:|
| Complete RL_DEV, 1696 scenes | 93.6192 | 93.5747 |
| Fixed TRAIN diagnostic, 512 scenes | 80.3316 | 79.1099 |

These are **SFT baselines, not RL gains**. The TRAIN diagnostic contains
directed difficult cases; its mean is not an unbiased full-TRAIN estimate.
DEV's different difficulty does not replace Navtest. Detailed metrics,
kinematics and execution costs are in `SFT_DEV_baseline.json` and
`SFT_TRAIN_DIAGNOSTIC_baseline.json`.

Complete paired5k DEV/TRAIN results, safety/zero transitions, gain/loss sums,
kinematics and the matched saved-batch gradients are now available in
[the interim evaluation](interim_ode_update005000.md) and
[training analysis](interim_training_10k.md). Later fixed milestones, complete
20k curves, final Navtest and final cost accounting remain **pending**.
Qualification results establish implementation evidence, not driving efficacy.
No Navtest92-point claim is made.

## Executed entries and native recovery

The repository module entries are real; executable commands and receipt
locations are in [validation.md](validation.md). Private assets remain under
`/mnt/project/simwam-recipe-20261015/`; the repository receives only code,
configuration and small aggregated evidence, never weights/data/cache/images.
Merged exports support original deployment, but exact training continuation
requires the unmerged native checkpoint with saved optimizer/RNG/inner chain.
