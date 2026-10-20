# Final physical-effect study comparison

The real pipeline is implemented and trained, but the proposed incremental physical-effect advantage is not established at the registered budget. Verified local repairs exist and ordinary refinement fits them. Effect-guided explicit refinement adds severe failures; effect-Flow does not beat ordinary correction FM/SFT. All positive and negative results below use completed actual artifacts, with no best-seed or DEV-based recipe selection.

All scores here use the same fixed S0, observation/noise protocol, official single-trajectory reference scorer and complete DEV1696/16-log population. PDMS is displayed on a0–100 scale; differences are percentage points, not relative percentages. Each training seed42/43/44 has its own trained parameters and the same1696 scenes;5088 repeated scene evaluations are not5088 independent scenes. SD is sample SD across three training seeds, not a population confidence interval. Per-seed intervals resample the16 complete source logs2000 times with fixed bootstrap seed42; inference-noise seed remains the original42. No best seed or method is selected from DEV.

## Complete policy results

Current fixed S0: **91.388363/100**,42 zero scenes. The following are independent trained endpoints; none are combined into an oracle deployment pool.

| Method | seed42 | seed43 | seed44 | Mean | Sample SD | Range |
|---|---:|---:|---:|---:|---:|---|
|A0|91.394335|91.397026|91.376070|91.389143|0.011402|91.376070–91.397026|
|A1|91.235007|91.235403|91.202189|91.224200|0.019063|91.202189–91.235403|
|ScoreRefiner|91.257100|not budgeted|not budgeted|91.257100|NA|91.257100–91.257100|
|B0|91.402832|91.432876|91.389449|91.408385|0.022240|91.389449–91.432876|
|B1-trajectory|91.342493|91.341241|91.388426|91.357387|0.026888|91.341241–91.388426|
|B1-effect|91.407752|91.356723|91.436154|91.400210|0.040249|91.356723–91.436154|
|RandomLocal|90.485193|not budgeted|not budgeted|90.485193|NA|90.485193–90.485193|
|FixedLateBrake|91.243356|not budgeted|not budgeted|91.243356|NA|91.243356–91.243356|

The repeated-seed mean changes versus S0 are A0+.000780, A1−.164163, B0+.020022, B1-trajectory−.030976 and B1-effect+.011847 points. Small positive means are not robust safety-preserving method success. All per-seed intervals and severe-risk counts are retained:

| Policy / train seed | Gain vs S0, points [paired-log95% CI] | Zero total | Old-zero repaired | New zero | High→zero | New NC/DAC/TTC |
|---|---|---:|---:|---:|---:|---|
|A0/42|+0.005972 [+0.000980,+0.009069]|42|0|0|0|0/0/0|
|A0/43|+0.008663 [+0.002548,+0.012517]|42|0|0|0|0/0/0|
|A0/44|-0.012293 [-0.100973,+0.014305]|42|0|0|0|0/0/2|
|A1/42|-0.153356 [-0.411284,+0.182525]|45|1|4|4|0/4/0|
|A1/43|-0.152960 [-0.410360,+0.182985]|45|1|4|4|0/4/0|
|A1/44|-0.186174 [-0.402899,+0.022816]|46|0|4|4|0/4/1|
|ScoreRefiner/42|-0.131263 [-0.288192,+0.027164]|45|0|3|3|0/3/0|
|B0/42|+0.014469 [-0.062418,+0.176092]|42|1|1|0|1/0/0|
|B0/43|+0.044513 [-0.007723,+0.178127]|41|1|0|0|0/0/0|
|B0/44|+0.001086 [-0.004618,+0.004961]|42|0|0|0|0/0/0|
|B1-trajectory/42|-0.045870 [-0.163073,+0.003741]|42|0|0|0|0/0/2|
|B1-trajectory/43|-0.047122 [-0.164385,+0.002596]|42|0|0|0|0/0/2|
|B1-trajectory/44|+0.000063 [-0.002238,+0.001632]|42|0|0|0|0/0/0|
|B1-effect/42|+0.019389 [-0.085559,+0.157561]|41|1|0|0|0/0/1|
|B1-effect/43|-0.031640 [-0.107805,-0.005771]|42|0|0|0|0/0/1|
|B1-effect/44|+0.047791 [-0.004001,+0.183757]|41|1|0|0|0/0/1|
|RandomLocal/42|-0.903170 [-1.449017,-0.027245]|58|5|21|17|4/18/6|
|FixedLateBrake/42|-0.145007 [-0.405284,+0.078208]|40|3|1|1|0/1/1|

A0 has two tiny positive within-run intervals and one seed with2 new TTC failures. A1 introduces4 new DAC/high→zero cases in each seed. B0's three intervals cover0 and one seed adds NC. B1-effect's seed43 interval is negative and all three add TTC. RandomLocal and fixed braking controls demonstrate that small geometric changes or always slowing are not automatically safe improvements.

## Does physical-effect supervision add value?

| Same-training-seed contrast | Seed | Difference, points | Paired-log95% CI |
|---|---:|---:|---|
|A1_minus_A0|42|-0.159327|[-0.418906,+0.179354]|
|A1_minus_A0|43|-0.161623|[-0.420992,+0.176620]|
|A1_minus_A0|44|-0.173880|[-0.404341,+0.114317]|
|B1-effect_minus_B0|42|+0.004920|[-0.096617,+0.057246]|
|B1-effect_minus_B0|43|-0.076152|[-0.223725,-0.000967]|
|B1-effect_minus_B0|44|+0.046705|[-0.006330,+0.185957]|
|B1-effect_minus_B1-trajectory|42|+0.065259|[-0.010106,+0.220169]|
|B1-effect_minus_B1-trajectory|43|+0.015482|[-0.010760,+0.102640]|
|B1-effect_minus_B1-trajectory|44|+0.047728|[-0.003679,+0.185990]|

A1−A0 is negative in all seeds (descriptive mean−.164943 points), with4 new DAC/high→zero cases relative to A0 each time; mean-difference log intervals cross0. B1-effect−B0 has mixed signs (mean−.008176), with one negative interval. B1-effect−B1-trajectory is positive in all point estimates (mean+.042823), but all intervals cross0 and B0 remains the stronger simple baseline. These findings do not support the full independent physical-supervision claim, and no multiple-comparison/significance or universal-safety assertion is made.

## Training fit, held-out support and target opportunity

| Method / seed42 | H_train6 direct gain, points | Train zeros repaired /3 | H_val1 direct gain | Identity fit785 gain | Identity val232 gain |
|---|---:|---:|---:|---:|---:|
|A0|+54.763715|3|+0.000000|+0.004890|+0.011656|
|A1|+54.669161|3|+0.000000|+0.001983|+0.008449|
|ScoreRefiner|+38.839639|2|+0.000000|+0.021753|+0.043486|
|B0|+38.754851|2|+0.000000|+0.002593|+0.015770|
|B1-trajectory|-0.033012|0|+0.000000|+0.003663|+0.002974|
|B1-effect|+16.669296|1|+0.000000|-0.111670|-0.005715|

The six TRAIN repairs and one held-out TRAIN repair were fixed by native feedback before training. The table uses exact scene/log/metric joins; the same original Base is preserved. A0/A1 genuinely learn their training targets. All six main methods leave the unique held-out repair zero, so no method has demonstrated held-out repair success on this limited support. Ordinary/identity replay improvements are not relabeled as hard repair. Three training seeds are completed, but one held-out repair cannot prove broad generalization.

G0's1024-scene actual bank contains7 selected finite-grid local repairs, including4DAC-zero,1TTC and2progress-only; no original NC failure occurs. The minimum-target oracle gain is+.418584 points, not a learned policy or a global maximum. Harmful interventions are abundant:331 high→zero light-lateral pairs/172 scenes and109 joint pairs/69 scenes. The positive repair support is only6 unique fit scenes, heavily re-exposed by the balanced training sampler. A larger network or more training steps alone is not justified as the missing solution.

## G1 and temporal evidence

Six matched1000-step G1 models are evaluated on all4615 valid native pairs/225 source-eligible scenes/114 logs, while the original233 requested denominator and8 structural no-pair scenes remain explicit. Every arm's actual bank recommendation gain is negative. Paired-delta gain−.778716 points, regret3.307579 against oracle opportunity2.528863, with4 high→zero/newNC1/DAC3/TTC6; superiority over absolute scoring/full consequences is not established.

Direction accuracy is not just majority-class prediction:the997 meaningful signed changes comprise390 improvements/607 deteriorations across178 scenes/99 logs. Always-negative reaches60.883% pair or65.710% scene-macro accuracy; delta reaches90.371%/90.052% with91.267% balanced accuracy. Absolute-score/full-feedback balanced accuracies are93.538%/94.426%; even trajectory-only/shuffled controls remain strong. Good direction classification does not establish within-scene extreme-choice calibration or safe correction gradients.

Removing only dense temporal-grid weight changes recommendation gain from−.778716 to−.018745 points. Full minus zero is−.759971,95%paired-log interval[−2.109284,+.328035]; both retain all11 mean labels, first-event times and event transitions. Added dense-grid utility is unsupported, not proof that all time information is useless. Native event/basis timing association is not a causal responsibility estimate, and a policy-level temporal-credit benefit has not been identified.

## Engineering and cost

The original S0 checkpoint/normalizer/current three-image inputs/FP32-master inference are unchanged. A and B share one original VLM; A adds one small refiner after the original Euler10, B adds a velocity adapter within its sole Euler10. No deployment simulator, GT, real feedback, teacher or second encoder runs. Actual one-scene image-to-final mean/p50/p95 seconds are S01.329762/1.329242/1.332660, A11.336315/1.335788/1.342403 and B1-effect1.366010/1.365946/1.368230, n=8 after2 warmups. These warm-file-cache bounded latencies are not full-population throughput. Private deployed parameters are A2,711,824 and B2,706,948; original shared S02,951,547,396. Operator FLOPs are explicitly partial lower bounds, not whole-model costs.

Same-backend actual CUDA resume is independently proven for G1 and B0 at next update9/final16; this is not blanket B1-effect/DDP/cross-device or terminal-zero-update qualification. Label poison, one-shared-forward, exact one-scene cached output and no forbidden deployment dependencies are actual engineering evidence. See [REGRESSION_AND_RESUME_REPORT.md](REGRESSION_AND_RESUME_REPORT.md) and [LATENCY_AND_PARAMETER_SUPPLEMENT.md](LATENCY_AND_PARAMETER_SUPPLEMENT.md).

The parent's actual strict native comparison finished EXIT0 in595.908296s for18 methods plus S0. Its JSON is pinned at SHA256 `8ed42603b817d3ed183bb027c4aad733a8dafb9e21d4577e265338e823710b4c`, identity `08bf17b4588c7aad1bb0bbb4db06a09af97aae2a5b928aade1c808f28287f110`, path `/mnt/project/action-effect-repair-20261020-budget-v2/comparisons/final_dev1696_all_seeds_v1/COMPARISON.json`. All19 official DEV score results cover1696 valid scenes,0 invalid/unresolved. Twelve additional official seed42 fit/validation results cover791/233 scenes. The read-only2.052s reporting derivation checks original RESULT/COMPLETE/registration pins and scene/log alignment, then calls only `paired_summary`; it performs0 new native feedback decodes, model forwards, simulations or updates. See [FINAL_POLICY_STATISTICS.json](final_analysis/FINAL_POLICY_STATISTICS.json), [FINAL_SYSTEM_RESULTS.csv](FINAL_SYSTEM_RESULTS.csv) and [SAME_SEED_METHOD_CONTRASTS.csv](SAME_SEED_METHOD_CONTRASTS.csv).

The optional8-thread reporting IO patch was CPU-tested, but the original serial comparison finished first and was lawfully reused. No production parallel-validation speedup is claimed. The earlier14/16 scoring snapshot remains a timestamped appendix; final completeness uses this19-policy matrix, not that snapshot. No NAVTEST, ACTIVE update, Base training change or external Scorer weights occurred.

## Research verdict and next evidence-backed direction

The tested contribution is **not supported**, rather than silently declared successful. Finite local repair opportunity and train fitting are real; stronger held-out repair, paired-effect superiority, causal temporal credit, minimum-edit superiority to unconstrained score search and effect-Flow superiority to simple SFT remain unproven. Unavailable v2/privileged/matched-RL controls are stated as unavailable.

The most justified next campaign would first enlarge legal TRAIN repair support across source logs and risk types, then validate the learned effect model on actual within-support intervention ranking/calibration and input-gradient proposals before policy updates. Preserve simple A0/B0 controls and pre-register the new cohort/selection rules before reading its held-out results. A1 proxy-gradient extrapolation and the weaker off-path velocity surrogate are distinct hypotheses to test, not diagnosed causes or reasons for automatic extra model/feature searches. Current DEV results must not be used to tune individual scene thresholds or select a favorable seed.
