# G2: explicit trajectory refinement — actual completed results

All seven formal routeA models completed1000 updates; all seven full DEV evaluations and the two untrained controls are complete. Engineering integration is real. Effect guidance did not improve on ordinary refinement at this endpoint and introduced high-to-zero failures. No external scoring weights, second VLM or deployment simulator is used.

All scores here use the same fixed S0, observation/noise protocol, official single-trajectory reference scorer and complete DEV1696/16-log population. PDMS is displayed on a0–100 scale; differences are percentage points, not relative percentages. Each training seed42/43/44 has its own trained parameters and the same1696 scenes;5088 repeated scene evaluations are not5088 independent scenes. SD is sample SD across three training seeds, not a population confidence interval. Per-seed intervals resample the16 complete source logs2000 times with fixed bootstrap seed42; inference-noise seed remains the original42. No best seed or method is selected from DEV.

Current S0 is **91.388363**,42 zero scenes. Means and training-seed spread:

| Method | seed42 | seed43 | seed44 | Mean | Sample SD | Range |
|---|---:|---:|---:|---:|---:|---|
|A0|91.394335|91.397026|91.376070|91.389143|0.011402|91.376070–91.397026|
|A1|91.235007|91.235403|91.202189|91.224200|0.019063|91.202189–91.235403|
|ScoreRefiner|91.257100|not budgeted|not budgeted|91.257100|NA|91.257100–91.257100|
|RandomLocal|90.485193|not budgeted|not budgeted|90.485193|NA|90.485193–90.485193|
|FixedLateBrake|91.243356|not budgeted|not budgeted|91.243356|NA|91.243356–91.243356|

## Actual risks and paired changes

| Policy / train seed | Gain vs S0, points [paired-log95% CI] | Zero total | Old-zero repaired | New zero | High→zero | New NC/DAC/TTC |
|---|---|---:|---:|---:|---:|---|
|A0/42|+0.005972 [+0.000980,+0.009069]|42|0|0|0|0/0/0|
|A0/43|+0.008663 [+0.002548,+0.012517]|42|0|0|0|0/0/0|
|A0/44|-0.012293 [-0.100973,+0.014305]|42|0|0|0|0/0/2|
|A1/42|-0.153356 [-0.411284,+0.182525]|45|1|4|4|0/4/0|
|A1/43|-0.152960 [-0.410360,+0.182985]|45|1|4|4|0/4/0|
|A1/44|-0.186174 [-0.402899,+0.022816]|46|0|4|4|0/4/1|
|ScoreRefiner/42|-0.131263 [-0.288192,+0.027164]|45|0|3|3|0/3/0|
|RandomLocal/42|-0.903170 [-1.449017,-0.027245]|58|5|21|17|4/18/6|
|FixedLateBrake/42|-0.145007 [-0.405284,+0.078208]|40|3|1|1|0/1/1|

A0's two tiny positive-seed gains have positive within-run log intervals; seed44 loses score and adds2 TTC failures. Its three-seed mean gain is only+.000780 points. It is not a broad robust repair success. A1's mean is below A0 in all three matched training seeds; its four new DAC failures per seed are also four original-high→zero scenes. Mean difference intervals cross0, so an across-log statistically significant average deterioration is not asserted. The observed severe new failures still violate the desired preservation behavior. The absolute-score control also introduces3 DAC/high-to-zero cases. Unconditional local changes are not a safe solution:RandomLocal creates21 new zeros, and FixedLateBrake trades progress against risk and still introduces one high-to-zero case.

| Same-training-seed contrast | Seed | Difference, points | Paired-log95% CI |
|---|---:|---:|---|
|A1_minus_A0|42|-0.159327|[-0.418906,+0.179354]|
|A1_minus_A0|43|-0.161623|[-0.420992,+0.176620]|
|A1_minus_A0|44|-0.173880|[-0.404341,+0.114317]|

## Training fit versus fixed held-out repair support

| Method / seed42 | H_train6 direct gain, points | Train zeros repaired /3 | H_val1 direct gain | Identity fit785 gain | Identity val232 gain |
|---|---:|---:|---:|---:|---:|
|A0|+54.763715|3|+0.000000|+0.004890|+0.011656|
|A1|+54.669161|3|+0.000000|+0.001983|+0.008449|
|ScoreRefiner|+38.839639|2|+0.000000|+0.021753|+0.043486|

`H_train`/`H_val` are fixed by original `REPAIR_DATASET.target_kind == verified_local_repair`, before any learned outcome. The other785/232 rows are the fixed identity-preservation population, not a favorable post-hoc subset. All model Base labels exactly join the original scene/log/metrics. A0/A1 genuinely fit the six training repairs; neither repairs the sole held-out repair scene, which remains zero. The small validation improvements come from identity scenes' progress, not repair of that failure. One held-out repair cannot establish broad generalization or distinguish its causes. The full6+1 direct endpoint audit and all12 source-result pins are in [PREDECLARED_FIT_VALIDATION_SUBSET_RESULTS.json](PREDECLARED_FIT_VALIDATION_SUBSET_RESULTS.json).

## Mathematical and deployment boundary

A0/A1/absolute-score control use the same2,711,824-parameter randomly initialized private coefficient refiner, current frozen W/A/ego and original physical trajectory. The same K8 four-window longitudinal/lateral basis is reconstructed; unsupported columns1/5 remain off. Zero coefficients exactly preserve the input. A1's learned G1 gradient is a surrogate through a prediction network, not a simulator derivative. Its allowed2m/.6m coefficients and combinations can exceed the single-axis1m/.3m feedback grid. The negative outcomes do not identify extrapolation as the sole cause, but provide no basis for treating those gradients as reliable physical repairs.

The primary export removes the frozen training teacher. It does not deploy calibrated effect confidence or claim native safety from its geometric screen. Real independent online execution of S0+A1 used one visual/language forward and the original10 DiT steps, plus one refiner call, no simulator/teacher/labels. On one actual TRAIN scene with2 warmups/8 repeats, mean1.336315s, p501.335788s, p951.342403s, versus S01.329762/1.329242/1.332660s. A1 adds2,711,824 parameters and about6.553ms in that bounded warm-file-cache measurement; all small heads were resident. This is not full-population latency. [The actual cost report](LATENCY_AND_PARAMETER_SUPPLEMENT.md) preserves memory, FLOP lower bounds and exact one-scene online/cache equality.

The parent's actual strict native comparison finished EXIT0 in595.908296s for18 methods plus S0. Its JSON is pinned at SHA256 `8ed42603b817d3ed183bb027c4aad733a8dafb9e21d4577e265338e823710b4c`, identity `08bf17b4588c7aad1bb0bbb4db06a09af97aae2a5b928aade1c808f28287f110`, path `/mnt/project/action-effect-repair-20261020-budget-v2/comparisons/final_dev1696_all_seeds_v1/COMPARISON.json`. All19 official DEV score results cover1696 valid scenes,0 invalid/unresolved. Twelve additional official seed42 fit/validation results cover791/233 scenes. The read-only2.052s reporting derivation checks original RESULT/COMPLETE/registration pins and scene/log alignment, then calls only `paired_summary`; it performs0 new native feedback decodes, model forwards, simulations or updates. See [FINAL_POLICY_STATISTICS.json](final_analysis/FINAL_POLICY_STATISTICS.json), [FINAL_SYSTEM_RESULTS.csv](FINAL_SYSTEM_RESULTS.csv) and [SAME_SEED_METHOD_CONTRASTS.csv](SAME_SEED_METHOD_CONTRASTS.csv).

The earlier14/16 [snapshot](ROUTE_DEV_SCORING_SNAPSHOT.json) is a dated appendix, superseded for completeness by the final report. Scientific conclusion:training fit is established; held-out repair generalization and an incremental effect-guidance benefit are not established. Existing S0 weights and ACTIVE are unchanged.
