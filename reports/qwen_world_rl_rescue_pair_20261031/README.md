# Rescue pair: zero-score recovery and high-score retention

The independent CPU owner was approved and started by ROOT at PID 2325892, start tick 1485516830. The two native eight-rank learners had finite startup metrics at phase updates 51 and 68; startup is evidence of execution, not a performance result.

Both arms initialize current LoRA and scene/predictive memory from SafetyEP10k, retain the original frozen IL70 reference, and start fresh AdamW, phase RNG seed42 and phase update0. Replay and CFM share the W2 sampler, SafetyEP advantages and static scene stream. CFM adds verified positive teacher supervision. The initial additional budget is 5k, with full NAVTEST and TRAIN1024 evaluations at additional1k/5k. The fixed comparison is SafetyEP10k and the same-phase other arm.

| Data evidence | Count |
| --- | ---: |
| Selected TRAIN diagnostic population | 15 |
| Safety10k canonical zero / positive-low / high scenes | 6 / 3 / 6 |
| Verified improving low-scene teachers | 9 |
| Retention teachers / eligible candidates | 512 / 541 |
| Retention physical logs covered / eligible | 159 / 159 |
| Joint log, current-speed and navigation strata | 410 / 410 |
| Previously below.8 recovered retention scenes (including previouszero) | 5 (3) |
| Exact original-normalized teacher endpoints | 521 |
| Maximum native FP32 codec roundtrip error | 0 |

Each sealed5k stage contains10000 fresh-scene exposures:8000full,1000low,1000high, with eight distinct scenes per global batch. Stage0 has8481 distinct scenes. Full103288 is the eligible population, not a claim that the initial pilot traverses every scene. Full draws also encounter teacher tokens: token-based auxiliary eligibility in stage0 is low1001/high1039, while selected-pool ratios remain exactly80/10/10.

Stage0 through9 are independently sealed through the standard `SceneStream` adaptive storage mode; selection remains static. The original teacher bank and original census were preserved. Extra sealed stages prevent data starvation if later continuation is approved; they do not commit a50k training budget.

This archive contains exact running owner/analyzer source, the bank/stage generators, and small receipts. Runtime copies at the asset paths in SOURCE_ARCHIVE.json remain authoritative. Large teacher NPZs and the103288-token stream manifest are referenced by SHA rather than committed. Labels are official-metric verified, not human-optimal or real-world safety certificates. NAVTEST tokens and logs never enter training selection. TRAIN diagnostics include training-feedback scenes and are not heldout.

The analyzer reports same-phase CFM-minus-Replay and each-minus-Safety10k,44-log paired confidence intervals, zero repairs and new zeros. The fixed Safety10k high-score cohort is defined before new results at score>=.9:8850 NAVTEST scenes and768 TRAIN scenes. No new score has been inferred from startup metrics.
