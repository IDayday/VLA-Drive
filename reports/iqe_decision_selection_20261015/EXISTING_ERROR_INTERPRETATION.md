# What the existing fixed-pool errors show

These are descriptive calculations on already completed R00 predictions and
official labels, with the same1696 dev scenes /16 logs. They do not select a new
main candidate pool. PDMS is shown out of100; changes are percentage points.
Base90.5302, K4 Oracle95.3517, R00 selected90.6960. Available gain4.8215pp is
mostly unrealized: R00 captures3.44%, with regret4.6557pp. Its gain CI crosses0.

| Selected specialist | Selections | Positive gain over full dev | Negative cost over full dev | High→0 |
|---|---:|---:|---:|---:|
| road_direction | 281 | +0.4523pp | −0.6310pp | 4 |
| interaction | 316 | +0.4257pp | −0.3554pp | 1 |
| safe_progress | 439 | +1.0240pp | −0.7498pp | 6 |

Safe-progress supplies the largest positive benefit and the largest negative
cost, including6 of11 high-to-zero replacements. Road choices have negative net
contribution. Interaction has positive net contribution but still has severe
errors. The labels are training specializations, not guarantees of physical
capability. None of these numbers justify declaring an expert universally safe
or useless.

The108 Base-low scenes show+9.0034pp average selected improvement, but the1588
Base-high scenes lose0.4352pp on average and contain all11 high-to-zero cases.
At current ego speed2–5m/s,568 scenes lose0.7828pp and contain6 of those11 cases.
Current navigation1 has1128 scenes, gain−0.4380pp, and9 high-to-zero cases.
These strata overlap and are not independent causal groups. They are diagnostic
descriptions, not a new hard router or a fitted dev rule.

Selected NC failures number33 and have35.0072pp mean negative change; selected
TTC failures number110 and have22.2328pp mean negative change. This motivates
keeping actual error magnitude rather than winner counts. A low regret on many
near-ties does not cancel a few large costs scientifically.

| Post-hoc subpool | Oracle | R00 restricted argmax | High→0 | New NC/DAC | Oracle loss versus K4 |
|---|---:|---:|---:|---:|---:|
| Base + road + interaction | 93.6424 | 90.3781 | 7 | 10 / 2 | 1.7092pp |
| Base + interaction + progress | 94.8477 | 90.8917 | 9 | 7 / 6 | 0.5040pp |
| Base + progress | 93.8363 | 90.6962 | 9 | 5 / 7 | 1.5154pp |
| Full K4 | 95.3517 | 90.6960 | 11 | 14 / 4 | 0 |

Removing progress reduces some severe selections but discards1.7092pp Oracle
opportunity and does not yield a positive R00 selected gain here. Removing road
retains most Oracle opportunity, but new DAC failures increase from4 to6. This
is not an across-the-board safety improvement. Confidence intervals for all
listed selected gains include0. The full fixed K4 remains the main experiment;
no dev-selected subpool is promoted. R00 has candidate-independent scores, so
restricted argmax is mathematically permissible for this old head. New M_set
has set attention and would require new subset forwards; its K4 logits must not
be truncated and called native K2/K3 inference.

Existing rich-bank coverage contains a true low-quality near candidate within
0.1m ADE for139 of140 severe R00 training errors across67 logs. It includes the
exact deployed candidates by construction. This shows that such negatives were
present, not that their boundary was learned or will generalize. No new rich
bank, trajectories or scoring jobs were generated for this audit.

Exact per-scene/per-log records, score/context hashes, cohort denominators and
the full eight Base-containing subpools are preserved under the runtime
`existing_audit/` and the report CSVs. Prior dev has been repeatedly observed;
these descriptions are exploratory.
