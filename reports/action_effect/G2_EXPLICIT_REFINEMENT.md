# G2: explicit physical-effect-guided trajectory refinement

Status: **formal endpoint training complete; partial matched policy results
now available, full effect-guidance comparison pending**. Base remains the same mature original S0.
No second VLM, simulator, external scorer or training feedback is used at deployment.

A0 ordinary refiner, A1 effect-guided refiner and the own-trained absolute-score
control share the same2,711,824-parameter coefficient network. Current W/A/ego and
original8-point physical trajectory produce8 coefficients over the same four-window
longitudinal/lateral quintic basis. The output layer starts at zero; reconstruction
is bit-exact identity at zero while its limiting derivative remains nonzero.
The fixed simulation grid did not identify window1, so coefficients1 and5 stay off;
active columns are `[1,0,1,1,1,0,1,1]`. Newly combined coefficients can exceed the
finite sampled grid's support, which is a reported generalization question.

Only verified local repairs or explicit identity-preservation targets enter
training. Fit contains **6 unique repairs and785 identity targets**; validation
contains1 repair and232 identities. All arms use the predeclared1:1 repair/keep
reference distribution, with replacement and full actual exposure reporting.
Repeated updates do not create more unique repair situations. This tiny support
is a limitation, not evidence for a broad safety-repair claim.

A0 learns physical XY/periodic yaw and coefficient targets, bounded change cost,
and high-quality identity preservation. A1 adds scaled effect-consistency and
predicted NC/DAC/TTC degradation costs from the frozen own delta G1 teacher.
Teacher gradients can flow through the proposed trajectory but never update
teacher/Base weights. ScoreRefiner uses its own absolute-score G1 mode. All
feedback targets remain on the loss side. Default introduced-event guidance is
off; uncalibrated uncertainty is not a safety certificate or hidden deploy gate.

The formal recipe is1000 updates, batch32, microbatch8, AdamW1e-4,wd.01,50 warmup
and fixed1000 cosine. A0/A1 have independent optimizers, outputs and seeds42/43/44;
ScoreRefiner uses the registered main seed. Source is immutable per run. Actual
new16-update A0 qualification on real cached S0 conditions consumed512 scene
presentations,4.535s training segment and166,050,304 peak allocated GPU bytes.
It is diagnostic, not the formal1000-step result or real-image deploy latency.

Deployment is one shared image/VLM forward, original Euler10 trajectory, then one
small refiner. The deploy geometry screen is preliminary: normalized history ego
is not an observed initial velocity, unlike the native offline controller context.
A screen pass is not native-simulation verification. Actual controls separately
include one deterministic random local perturbation and a fixed late-brake basis
perturbation; they have no label/GT/oracle input. Their real official DEV results
must be retained even if they worsen the Base.

Required matched results: original zero/low-score repair, high-score→zero/new hard
failures, NC/DAC/TTC, progress/comfort, displacement, intervention timing and actual
image-to-final p50/p95/peak memory. Fixed DEV has1696 scenes/16 logs; complete scores
and original denominators are mandatory. The current context cache's completion
or a frozen teacher's finite gradient is not a G2 scientific PASS.

See [A_TRAINING_AND_EVALUATION_ENTRY.md](A_TRAINING_AND_EVALUATION_ENTRY.md),
[A_NATIVE_JOIN_REVIEW_FIX.md](A_NATIVE_JOIN_REVIEW_FIX.md), the index's source/data
pins, and [RUNBOOK.md](RUNBOOK.md). Final policy results: **NOT_RUN/PENDING**.

## Actual completed evidence added after the bounded smoke

All7 predeclared A models reached1000: A0/A1×seeds42/43/44 and one absolute-score
refiner. Training completion is factual; effect-guidance benefit remains pending.
The input/output/source receipts and run timing/memory are retained in the index.

The complete current-protocol1696/16-log controls are:

| Method | PDMS/100 | Δ points versus S0 | Paired-log95% interval, points | Zeros | Old zeros repaired / new zeros | High→zero |
|---|---:|---:|---|---:|---:|---:|
| S0 |91.388363|0|—|42|—|—|
| RandomLocal |90.485193|−.903170|[−1.449017,−.027245]|58|5/21|17|
| FixedLateBrake |91.243356|−.145007|[−.405284,+.078208]|40|3/1|1|

RandomLocal adds4NC/18DAC/6TTC failures; FixedLateBrake adds0/1/1. Braking improves
TTC mean but reduces progress. Both controls use exactly one prescribed change,
not an oracle search or trained weights. They illustrate that local changes can
harm high-score cases; they do not establish that our learned effect guidance fixes
that problem. Original strict comparison and all component statistics are in
[G2_UNTRAINED_CONTROLS_COMPARISON.json](G2_UNTRAINED_CONTROLS_COMPARISON.json).

A0 seed42's fullfit791 native score rises94.503553→94.923807, +.420254 points.
The6 **predeclared H_train** scenes (6 logs) rise35.285061→90.048776, direct mean
+54.763715 points. All3 train zeros are repaired and no new NC/DAC/TTC or zero
appears on that fit population.785 identity scenes have mean+.004890 points.
[The H-set audit](A0_PREDECLARED_HARD_SET_GAIN.json) joins by exact scene/log and
uses the original target_kind, never a subset chosen after seeing outputs.
This establishes fitting the specific trained situations, not held-out repair.

Complete A0 DEV means by fixed seed42/43/44 are91.394335/91.397026/91.376070,
respectively. Their tiny mixed directions are not stable generalization evidence
or a reason to pick the best seed. The single predeclared H_val case is not yet
admitted here; its result must be shown as n=1, without broad-confidence claims.
Final A1/absolute-guided paired policy and severe-risk comparisons remain pending.
## Later immutable DEV scoring snapshot

At 2026-10-20T03:57:06 UTC, all seven trained routeA DEV scoring results have
COMPLETE receipts over the same1696 scenes/16 logs and current score protocol.
[The snapshot](ROUTE_DEV_SCORING_SNAPSHOT.json) validates the exact RESULT pins,
registration agreement, complete scene/log population, valid counts and reported
means; it does not repeat native-receipt validation or replace the required
paired safety comparison.

| Arm | Seed42 PDMS/100 | Seed43 | Seed44 |
|---|---:|---:|---:|
| Current fixed S0 | 91.388363 | same fixed policy | same fixed policy |
| A0 ordinary refiner | 91.394335 | 91.397026 | 91.376070 |
| A1 effect-guided refiner | 91.235007 | 91.235403 | 91.202189 |
| Absolute-score refiner | 91.257100 | not budgeted | not budgeted |

A1's complete-population mean is below A0 in each matched training seed. These
actual means do not support an effect-guidance advantage at this endpoint. The
paired new-zero/NC/DAC/TTC counts, source-log intervals, training-holdout repair
analysis and actual deployment latency remain separate required evidence; no
seed is selected from this DEV table and no safer-policy claim is made.
