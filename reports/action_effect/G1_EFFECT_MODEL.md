# G1: paired physical-effect model and six matched controls

Status: **real training and matched held-out pair comparison complete; paired-
delta superiority is not supported at this registered budget**.
Exact inputs and factual checkpoint receipts are pinned by
[EVIDENCE_INDEX.json](../effect_repair/EVIDENCE_INDEX.json). This report does not
turn a finite loss or a completed optimizer loop into a learning/generalization
PASS.

## Fixed task and supervision

The shared frozen original S0 gives W144×2048, A8×2048 and current ego4. Ego is the
normalized previous relative pose, **not velocity/acceleration**. No GT, future
actor, native feedback, real score or external scorer enters the prediction
context. The physical base/modified trajectories are8×3 ego-frame metres/radians.
Labels are modified-minus-base from the same real scene, protocol, official
reference input and background log. Zero and pair-swap contracts are tested.

The six models allocate the same **2,749,603** FP32 parameters, width256,2 layers,
4 heads, dropout0, seed42: absolute score/components; terminal PDMS only; complete
absolute available feedback; paired full delta; trajectory-only; shuffled scene.
The first three predict separate potentials, whose difference is used for
intervention comparisons. The latter delta modes use antisymmetric ordered
features. All added parameters are our random initialization. The privileged
upper-bound arm is BLOCKED because no registered legal privileged asset exists.

Available labels are7 normalized terminal scores/components, raw progress metres
and3 first-event times with individual masks. Time-grid supervision contains41
collision/road/TTC values. Scales are1 for normalized labels,20m for raw progress,
4s for times. Unobserved first events are masked rather than assigned time0.
Continuous clearance, continuous TTC, dense native progress and comfort timestamps
are not available; no substitute values are invented.

Losses separately reduce value, absolute-gap ranking, event transitions, temporal
grid, uncertainty and repair eligibility using full-batch valid denominators.
The main coefficients are1,.2,.2,.2,.05,.2. Ranking retains native gap magnitude;
it never normalizes that magnitude away inside each scene. All targets, raw
features and masks are cached; no VLM is run during head fitting.

## Actual training and fair comparison

The1024 fixed TRAIN cohort has791 fit and233 held-out scenes by complete log.
Base saw the original source TRAIN data; this holdout concerns new effect fitting,
not Base-unseen observations. The strict real bank contains20678 pairs across983
scenes. G1's fitting sampler actually contains758 unique non-joint pair scenes.
Joint long/lat intervention combinations are withheld from fitting and separately
reported. No-pair requested scenes must remain accounted for and must not be
misrepresented as silently successful predictions.

All six seed42 models reached **1000 updates /32000 scene presentations each**.
Their unchanged recipe is batch32/microbatch8, AdamW1e-4,wd.01,50 warmup, fixed1000
cosine to.1 peak, clip1. Training segment wall was814.284–823.012s per arm on the
assigned A800s. This includes that segment's validation/checkpoint work, excludes
initial dataset/model admission, and is not a matched end-to-end compute-budget
claim. Different objectives can activate different heads despite equal allocation.

Full held-out CPU evaluation was completed separately using each explicit final snapshot,
its immutable pair/context bank and all eligible held-out labels. The report tool
retains full physical-unit errors, log/scene means, native within-bank recommendation
gain, regret, new risks, event-class calibration and joint OOD diagnostics. Dense
prediction error is not policy PDMS; top-k truth is an offline oracle upper bound.

The actual comparison below tests delta versus absolute controls on unseen
post-training logs, retained physical-risk labels and withheld joint interventions.
It does not establish useful superiority or calibrated uncertainty. Rare event
support and first-event censoring limit what is identifiable. The separately
registered `.2→0` temporal-grid-only ablation uses
an independent output and leaves all11 physical means and other objectives intact;
it cannot by itself prove policy-level temporal credit.

The actual bounded-source recipes, teacher origin and mathematical reviews are
[G1_FIXED_COMPARISON_PROTOCOL.md](G1_FIXED_COMPARISON_PROTOCOL.md),
[G1_A_MODULE_CONTRACT.md](G1_A_MODULE_CONTRACT.md) and
[G1_COMPARISON_ENTRY.md](G1_COMPARISON_ENTRY.md). Use the existing explicit
`compare_effect_models evaluate` entry for original full-request evidence, then
the new `compare_eligible_effect_models` versioned reporting entry in the runbook.
The original full-population bootstrap guard is retained. Do not select a
better teacher from DEV: the main delta teacher is predeclared step1000 seed42.

The quantitative completed comparison and its explicit negative/limited conclusion
follow below. The temporal ablation is separately compared below; downstream
policy results remain separate.

## Actual full held-out pair-task comparison

The new reporting admission retains the original233 requested validation scenes,
then validates every native prediction before defining the complete **225 eligible
scenes /114 logs /4615 pairs**. The8 structural no-intervention scenes stay explicit,
with no invented absolute score. Original COMPLETE/METRICS/predictions are unchanged.
All6 actual checkpoint scene samplers, pair RNG and32000 presentations match.
[Compact JSON](G1_ACTUAL_ELIGIBLE_RESULTS.json) pins the complete derived artifact;
[CSV](G1_ACTUAL_ELIGIBLE_RESULTS.csv) retains every arm, not only a favorable row.
Independent affected-scope54CPU cases and the actual six-run read-only
verification are recorded in [ELIGIBLE_REPORT_INDEPENDENT_REVIEW.md](../effect_repair/ELIGIBLE_REPORT_INDEPENDENT_REVIEW.md).
The complete12,391,751B report was copied unchanged into the persistent campaign
artifact root, with its SHA87ed59c1 preserved; it is not stored as raw data in Git.

| G1 arm | Direction accuracy, scene % | Score-Δ RMSE,0–1 | Selected native Δ, points | Regret, points | High→zero | New NC/DAC/TTC |
|---|---:|---:|---:|---:|---:|---|
| Absolute score/components |92.5380|.202215|−.751100|3.279963|4|1/3/8|
| Terminal PDMS |87.7108|.178102|−1.410583|3.939445|5|1/4/8|
| Absolute full feedback |93.5561|.223283|−.631859|3.160722|4|1/3/8|
| Paired delta |90.0516|.213995|−.778716|3.307579|4|1/3/6|
| Trajectory only |86.9084|.198451|−1.167315|3.696178|4|1/3/8|
| Shuffled context |87.2675|.207424|−1.954137|4.483000|6|1/5/8|

All means concern the eligible intervention-bank task, **not full233 policy PDMS
or DEV**. Oracle opportunity is2.528863 points on this225-scene grid. For paired
delta, `gain=oracle−regret=2.528863−3.307579=−.778716 points`. The model switches
224/225 scenes:87 switches gain>.01,11 lose>.01, while severe tails include4
high-score→zero cases. Positive progress mean accompanies worse hard-safety
components, so a gain in progress or direction classification is insufficient.

Paired delta minus absolute-score selected increment is−.027616 points; its2000-
draw paired-log95% interval is[−1.911268,+1.840791]. Against full absolute feedback
it is−.146857 points, interval[−2.003349,+1.745996]. These do **not** establish an
advantage of the paired-delta method. All other predeclared contrasts are retained
in the compact JSON. One training seed cannot establish training-seed stability.

The~90% direction statistic is not a success verdict: non-tie support is997 pairs,
and action selection amplifies residual errors over a candidate set. The concrete
counterexample is good-looking direction accuracy with negative actual recommended
consequence and new safety failures. A confidence/abstention or rare-cost diagnosis
may be justified later, but no threshold/loss/model is changed using these results.

Current G1 conclusion: **training is real; useful independent guidance superiority
has not been established**. This is a scientific negative/limited result, not a
code or data-contract failure. RouteA/B outcomes still need their own actual
comparisons; the separate temporal comparison follows below. These G1 observations
do not retrospectively choose a new teacher or revise the predeclared policy pool.

The public JSON also retains all11 physical-label valid counts/RMSE/MAE/masks,
rare-event confusion and calibration diagnostics, every intervention-family
subset (including withheld joint combinations), and dense temporal errors for
all six arms. First-event disappearance is not converted to a zero-time label;
censored-time regression remains unavailable. Low dense-grid MAE under many
unchanged states is not proof of useful event timing or temporal responsibility.

## Actual dense temporal-grid-only ablation

The extra registered seed42/1000-step run is complete. The independent read-only
comparison verifies all225eligible scenes/114logs/4615pairs, matching Base, model
capacity, all11 mean labels, actual scene/pair RNG and32000 presentations, complete
optimizer/scheduler recipe and backend. Only dense-grid weight differs0.2 versus0;
first-event means and event-transition/ranking supervision remain unchanged.

| Dense-grid weight | Native recommendation Δ, points | Regret, points | High→zero | New NC/DAC/TTC |
|---|---:|---:|---:|---|
|.2|−.778716|3.307579|4|1/3/6|
|0|−.018745|2.547608|2|1/1/7|

Full minus zero is−.759971 points,95% paired-log interval[−2.109284,+.328035].
This fails to establish additional recommendation utility from the dense temporal
auxiliary objective at the registered budget. The point estimate is worse, but
the interval crosses0; it does not prove all temporal information is harmful.
The0-weight control remains slightly negative with2high→zero cases and new hard
failures, so it is not a validated deployment rule.

Full reduces dense-head MAE.02252→.00791 and raises scene direction accuracy
.8928→.9005, yet actual selection utility worsens. Predicting a dense auxiliary
target better is therefore insufficient evidence of useful physical credit.
[The independent report](../effect_repair/TEMPORAL_ABLATION_INDEPENDENT_ACTUAL_REPORT.md)
and [pinned JSON](../effect_repair/ELIGIBLE_REPORT_INDEPENDENT_ACTUAL_TEMPORAL.json)
preserve the complete4,163,548B report, exact sampler proofs and execution logs.
All10 original artifact hashes remain unchanged; the derivation used no forward,
GPU, optimizer step or new simulation. No primary teacher or policy pool is changed.
## Signed-change imbalance diagnostic

[The original-prediction audit](G1_DIRECTION_IMBALANCE_AUDIT.md) checks all six
unchanged4615-pair files. Its997 meaningful score changes comprise390 improvements
and607 deteriorations across178 scenes/99 logs. Always-negative direction accuracy
is60.883% per pair and65.710% scene-macro, compared with delta's90.371%/90.052%.
Delta's positive recall95.385%, positive precision82.667%, negative recall87.150%
and balanced accuracy91.267% rule out the simple majority-class explanation.

They do not establish a causal/context or selection advantage. Absolute-score and
full-feedback balanced accuracies are93.538% and94.426%; trajectory-only/shuffled
controls remain strong. The same delta model's actual recommendation gain remains
negative. The997 interventions are not independent scenes, and classification on
all meaningful changes is distinct from safely selecting a scene's best action.

## Completed downstream policy check

All registered downstream policies have now been evaluated on the same complete1696/16-log DEV. A1's three-seed mean91.224200 is below A0's91.389143 and it creates four new DAC/high-to-zero failures in each seed. B1-effect mean91.400210 is below simple correction FM/B0 mean91.408385; its comparison to the weaker B1-trajectory has positive point estimates but all paired-log intervals include0. No G1 teacher was changed based on these results. These actual outcomes reinforce that signed-change classification and a differentiable learned proxy are not sufficient evidence of reliable improvement guidance. [The final comparison](FINAL_COMPARISON.md) preserves every seed, risks and intervals.
