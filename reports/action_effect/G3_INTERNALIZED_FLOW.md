# G3: effect internalization in one Flow generation — actual results

All nine independently initialized/trained routeB endpoints completed1000 updates and full matched DEV scoring. Deployment integration succeeds with one original Euler10 generation. At this endpoint, effect supervision has not established an advantage over ordinary correction-trajectory FM/SFT.

All scores here use the same fixed S0, observation/noise protocol, official single-trajectory reference scorer and complete DEV1696/16-log population. PDMS is displayed on a0–100 scale; differences are percentage points, not relative percentages. Each training seed42/43/44 has its own trained parameters and the same1696 scenes;5088 repeated scene evaluations are not5088 independent scenes. SD is sample SD across three training seeds, not a population confidence interval. Per-seed intervals resample the16 complete source logs2000 times with fixed bootstrap seed42; inference-noise seed remains the original42. No best seed or method is selected from DEV.

Current S0 is **91.388363**,42 zeros.

| Method | seed42 | seed43 | seed44 | Mean | Sample SD | Range |
|---|---:|---:|---:|---:|---:|---|
|B0|91.402832|91.432876|91.389449|91.408385|0.022240|91.389449–91.432876|
|B1-trajectory|91.342493|91.341241|91.388426|91.357387|0.026888|91.341241–91.388426|
|B1-effect|91.407752|91.356723|91.436154|91.400210|0.040249|91.356723–91.436154|

## Per-seed policy outcomes

| Policy / train seed | Gain vs S0, points [paired-log95% CI] | Zero total | Old-zero repaired | New zero | High→zero | New NC/DAC/TTC |
|---|---|---:|---:|---:|---:|---|
|B0/42|+0.014469 [-0.062418,+0.176092]|42|1|1|0|1/0/0|
|B0/43|+0.044513 [-0.007723,+0.178127]|41|1|0|0|0/0/0|
|B0/44|+0.001086 [-0.004618,+0.004961]|42|0|0|0|0/0/0|
|B1-trajectory/42|-0.045870 [-0.163073,+0.003741]|42|0|0|0|0/0/2|
|B1-trajectory/43|-0.047122 [-0.164385,+0.002596]|42|0|0|0|0/0/2|
|B1-trajectory/44|+0.000063 [-0.002238,+0.001632]|42|0|0|0|0/0/0|
|B1-effect/42|+0.019389 [-0.085559,+0.157561]|41|1|0|0|0/0/1|
|B1-effect/43|-0.031640 [-0.107805,-0.005771]|42|0|0|0|0/0/1|
|B1-effect/44|+0.047791 [-0.004001,+0.183757]|41|1|0|0|0/0/1|

B0's three means are slightly above S0 but all three log intervals include0; seed42 adds one NC failure/new zero while repairing one old zero. B1-effect has two positive and one negative gain; seed43's within-run interval is negative and every seed adds one TTC failure. Its three-seed mean91.400210 is below B0's91.408385. The+.011847 mean difference from S0 is not evidence of an independent physical-supervision contribution.

| Same-training-seed contrast | Seed | Difference, points | Paired-log95% CI |
|---|---:|---:|---|
|B1-effect_minus_B0|42|+0.004920|[-0.096617,+0.057246]|
|B1-effect_minus_B0|43|-0.076152|[-0.223725,-0.000967]|
|B1-effect_minus_B0|44|+0.046705|[-0.006330,+0.185957]|
|B1-effect_minus_B1-trajectory|42|+0.065259|[-0.010106,+0.220169]|
|B1-effect_minus_B1-trajectory|43|+0.015482|[-0.010760,+0.102640]|
|B1-effect_minus_B1-trajectory|44|+0.047728|[-0.003679,+0.185990]|

Effect minus ordinary B0 has mixed signs and mean−.008176 points; seed43 favors B0 with its log interval excluding0, while the other two intervals include0. Effect minus B1-trajectory has positive point differences in all three seeds and mean+.042823, but all three intervals include0. This is a limited directional observation relative to the weaker source-bridge surrogate, not a verified advantage over SFT or a robust multi-seed effect benefit. No seed or candidate pool is selected after these results.

## Does the new head learn the fixed repairs?

| Method / seed42 | H_train6 direct gain, points | Train zeros repaired /3 | H_val1 direct gain | Identity fit785 gain | Identity val232 gain |
|---|---:|---:|---:|---:|---:|
|B0|+38.754851|2|+0.000000|+0.002593|+0.015770|
|B1-trajectory|-0.033012|0|+0.000000|+0.003663|+0.002974|
|B1-effect|+16.669296|1|+0.000000|-0.111670|-0.005715|

The fixed six training-repair direct gains distinguish fitting from tiny all-population averages. B0 repairs2/3 training zeros; B1-trajectory none; B1-effect1/3. B1-effect also reduces identity-fit785 mean by.111670 points and introduces2 TTC failures there. All three methods leave the sole held-out repair scene zero. The held-out support is n=1 and cannot support a generalization claim. [The direct subset audit](PREDECLARED_FIT_VALIDATION_SUBSET_RESULTS.json) fixes all scenes before learned outcomes and records all12 source-result pins.

## What is actually optimized

The original raw action is8×4 normalized XY/sin/cos. Actual FM uses `x_u=(1-u)*epsilon+u*a`, target `a-epsilon`, the original Beta-derived u including its small negative tail, repeat8 independent noise/time replicas, and the unchanged normalization/sampler. One generation is10 Euler network evaluations, not one call.

B0 trains the plus-target bridge `x_plus` against `a_plus-epsilon`. B1-trajectory trains the source bridge `x_minus` against residual `a_plus-epsilon-stopgrad(v_base(x_minus,u,z))`. This is an explicitly declared off-path surrogate, not the exact CFM derivative of the source bridge or a proved simulator/optimal-control gradient. B1-effect uses the identical velocity target plus native11-channel difference/direction and detached256-dimensional verified-pair effect-representation supervision.

The own frozen G1 teacher is queried only on the real actually simulated physical endpoints, not `eps+velocity` proxy trajectories. Student forward uses only frozen F_plan, noise state and time. No actual physical target, effect embedding, feedback or future observation is a forward input. The fixed loss is velocity1+keep1+delta.2+direction.2+representation.2. Three arms allocate the same2,775,567 private parameters and common initial velocity weights/seed schedule. Exports remove68,619 training-only auxiliary parameters, retaining2,706,948 velocity-adapter parameters. A checkpoints do not initialize B; shared S0 remains frozen.

## Actual deployable cost and conclusion

Actual B1-effect online inference called visual1,language1,original DiT10,adapter10,refiner0 and no G1/simulator. It does not produce a complete S0 trajectory before another correction. The one-real-scene2-warmup/8-repeat image-to-final mean/p50/p95 are1.366010/1.365946/1.368230s, versus S01.329762/1.329242/1.332660s. The adapter adds about36.247ms in that bounded measurement, not zero computation. Peak allocated is12,860,709,888B with all small heads resident. One-scene output equality to the exact frozen cache/checkpoint is bitwise; no population-wide online/cache equivalence or whole-VLA FLOP total is claimed. See [LATENCY_AND_PARAMETER_SUPPLEMENT.md](LATENCY_AND_PARAMETER_SUPPLEMENT.md).

The parent's actual strict native comparison finished EXIT0 in595.908296s for18 methods plus S0. Its JSON is pinned at SHA256 `8ed42603b817d3ed183bb027c4aad733a8dafb9e21d4577e265338e823710b4c`, identity `08bf17b4588c7aad1bb0bbb4db06a09af97aae2a5b928aade1c808f28287f110`, path `/mnt/project/action-effect-repair-20261020-budget-v2/comparisons/final_dev1696_all_seeds_v1/COMPARISON.json`. All19 official DEV score results cover1696 valid scenes,0 invalid/unresolved. Twelve additional official seed42 fit/validation results cover791/233 scenes. The read-only2.052s reporting derivation checks original RESULT/COMPLETE/registration pins and scene/log alignment, then calls only `paired_summary`; it performs0 new native feedback decodes, model forwards, simulations or updates. See [FINAL_POLICY_STATISTICS.json](final_analysis/FINAL_POLICY_STATISTICS.json), [FINAL_SYSTEM_RESULTS.csv](FINAL_SYSTEM_RESULTS.csv) and [SAME_SEED_METHOD_CONTRASTS.csv](SAME_SEED_METHOD_CONTRASTS.csv).

The engineering single-generation mechanism is proven in the tested scope. The scientific test—effect knowledge beats ordinary correction SFT under matched data, budget and safety—is **not supported at this endpoint**. A matched same-compute RL control, privileged teacher upper bound, NAVSIMv2 and stronger repair support are unavailable or untested; none is silently substituted.
