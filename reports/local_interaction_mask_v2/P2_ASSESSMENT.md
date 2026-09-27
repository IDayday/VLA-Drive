# Main graph comparison at the finite32pass endpoint

BothALL andMASK completed7296updates/233088presentations from the same initialization, source, current graph, label association and batch32schedule. Each used about1.189GPUh including both training segments, loading and holdout evaluation. The registered16→32extension consulted only world-task curves. This report does not contain a planning score.

The fixed training-domain holdout has64scenes/59logs,608active predicted neighbors and54accepted matched neighbor tracks. Both models were compared on the same406valid neighbor timepoints; only45neighbors have final-time labels forFDE. Complete target coverage is reported separately and must not be inferred from this matched cohort.

| Metric, metres | ALL | MASK | Interpretation |
|---|---:|---:|---|
| All-hidden graph egoADE | 2.209 | 2.225 | MASK slightly worse; these are internal graph ego predictions. |
| All-hidden neighborADE | 4.002 | 3.928 | Small aggregate reduction; stationary reference is3.637. |
| Dynamic neighborADE | 4.548 | 4.614 | MASK slightly worse; stationary reference is5.536. |
| Static neighborADE | 3.344 | 3.100 | MASK lower, but both worse than stationary1.345. |
| Ego-hidden privileged-condition egoADE | 2.189 | 2.197 | Known neighbor futures are diagnostic inputs, unavailable at deployment. |
| Neighbor-hidden privileged-condition neighborADE | 1.564 | 1.836 | Only3complete selected trajectories/24timepoints; insufficient population for a strong comparison. |

The average ego/neighborADE still changed4.05%forALL and1.93%forMASK between24and32passes. The fixed upper limit is not evidence of convergence. There is no clear broad advantage for actor masking at this endpoint; low coverage and a still-developing public foundation prevent interpreting it as a general rejection of the idea.

## Principal unresolved problems

1. Current-instance quality limits the experiment. The full training audit retains4406/26365supported/relevant motion targets. Plausible camera projection does not verify object existence, and dense false-positive graphs remain among the retained cases.
2. Mask supervision is sparse. Of actual neighbor-hidden training tasks,89.433%have no valid hidden future target. The counts are exposed in `P2_MAIN_EPOCH32.json`. No future-valid node selection or resampling was introduced. Reliable current detections and identity association must improve before substantially stronger claims about the masking objective are possible.
3. Relation dependence is poorly identified. Of64holdout scenes,53permit the current-graph related-vs-weak diagnostic, but only3actually remove labeled future points on both sides. Mean egoADE changes after related/weak future removal are0.01149/0.000109m forALL and0.00687/0.000406m forMASK. These are privileged, potentially out-of-distribution condition removals, not graph-message ablations, causal effects or deployment gains.
4. Planning transfer remains an independent question. The two graph bridges are now training alongside the completed initial current-memory control. The same actualDiT holdout output was verified for all three at zero gate. Only locked fullNavtest paired results can determine whether any deployed planning benefit exists beyond extra current-memory capacity.
5. Public-baseline maturity limits external conclusions. The shared foundation trained7284scenes for24passes and still changes near its finite endpoint. It is reproducible from publicQwen, but is not the paper's larger training recipe or proof that a fully trained baseline would show the same difference.

The bounded P3controls continue to the requested planning endpoint without changing selection rules, model capacity, training seed or metrics from these diagnostic results. They measure downstream consequences under this explicitly limited graph quality. They do not certify that the graph has already learned sufficient interaction structure.

Machine-readable evidence: `P2_MAIN_EPOCH32.json`, `FINAL_MASK_vs_ALL_common_tracks.json` and its sceneCSV, both `FINAL_G_LOCAL_*_relations.json`/CSVs, `P3_IDENTICAL_INITIAL_EVALUATION.json`, and the full graph audit/role-coverage files. The nearest control also completed32passes. On41common neighbor tracks/308valid points, relationMASKADE is4.218m versus nearestMASK3.981m; internal egoADE is2.225m versus2.343m. This mixed result does not establish a general relation-selection advantage. See `FINAL_MASK_vs_NEAREST_common_tracks.json`; do not compare its different cohort directly with the54track ALL/MASK table above.

The user's supervision-data challenge prompted the full partition in `SUPERVISION_REVIEW.md`: future labels exist; most empty tasks arise from predicted-slot nonassignment or rejection. The shared foundation omitted motion supervision before freezing, unlike the cited WCog semantic-stage recipe. Treat this as a limitation of the implemented training design, not evidence that NAVSIM lacks actor supervision.
