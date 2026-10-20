# G1 direction sign and class-imbalance audit

Read-only derivation at 2026-10-20T03:59:29.285348+00:00. Source is the six original COMPLETE PAIR_PREDICTIONS artifacts, each exact SHA256/byte count checked against its original receipt. All4615 native-pair IDs, scene/log IDs, score labels and score masks agree across arms. No model forward, simulation, build, optimizer update or training-config change was performed.

There are997 valid score pairs with `abs(after.score-before.score)>0.01`:390 positive (39.1174%) and607 negative (60.8826%). They span178 scenes and99 source logs; the full eligible bank contains4615 pairs across225 scenes/114 logs. The other8 of233 originally requested validation scenes contain no valid native intervention. These are dependent interventions, not997 independent scenes. The47 eligible scenes without a non-tie score pair are excluded only from this classification rate, not from action-recommendation evaluation.

The always-negative diagnostic gets60.8826% pair-weighted accuracy and65.7098% scene-macro accuracy. It predicts deterioration for every eligible pair; no training or model is involved. The existing roughly90% direction rate is scene-macro, not the pooled pair rate.

| Arm | Pair accuracy % | Scene-macro accuracy % | Positive recall % | Positive precision % | Negative recall % | Balanced accuracy % |
|---|---:|---:|---:|---:|---:|---:|
|absolute_score|92.5777|92.5380|97.9487|85.2679|89.1269|93.5378|
|terminal_pdms|88.2648|87.7108|98.4615|77.5758|81.7133|90.0874|
|absolute_feedback|93.3801|93.5561|99.2308|86.0000|89.6211|94.4259|
|delta_feedback|90.3711|90.0516|95.3846|82.6667|87.1499|91.2673|
|trajectory_only|87.6630|86.9084|97.6923|76.9697|81.2191|89.4557|
|shuffled_context|87.1615|87.2675|96.9231|76.5182|80.8896|88.9063|

Use score prediction sign at zero:positive if `mean_delta[score]>0`, negative if `<0`; an exact predicted zero is wrong for either eligible nonzero class. Positive precision is `TP/(TP+FP)`, positive recall `TP/P`, negative recall `TN/N`, balanced accuracy their equal average. Pair accuracy weights every qualifying pair equally. Scene-macro accuracy first averages the qualifying pairs within each scene and then averages178 scene rates; its value exactly matches the previously published derived reports. No division by per-scene score-gap sum is used.

For delta_feedback, the pooled confusion counts are372 true positives,529 true negatives,78 false positives and18 false negatives, with0 exact predicted-zero cases. Its positive recall95.3846% and balanced accuracy91.2673% show that its direction rate is not explained by always predicting the majority negative class. This does not establish superiority:the absolute-score/full-feedback controls have higher balanced accuracies in this same bank, and trajectory-only/shuffled-context controls are also strong. There is no independent log-bootstrap superiority calculation in this descriptive diagnostic.

Direction classification must not replace utility. The same fixed delta model still has negative actual recommendation gain, high-to-zero changes and new hard failures. Classification covers all997 meaningful signed changes, whereas deployment picks an extreme among several correlated candidates, often for an already high-quality Base. Calibration, ranking within a scene and the cost of false-positive recommendations remain distinct. These statistics neither repair the negative policy result nor justify changing the teacher after inspecting DEV.

The full source pins, exact counts, confusion matrices, formulas, same-pair assertion, unchanged-source checks and CPU cost are in [G1_DIRECTION_IMBALANCE_AUDIT.json](G1_DIRECTION_IMBALANCE_AUDIT.json). The compact table is also available as [CSV](G1_DIRECTION_IMBALANCE_AUDIT.csv). The already completed native-data admission remains [G1_ACTUAL_ELIGIBLE_RESULTS.json](G1_ACTUAL_ELIGIBLE_RESULTS.json); this audit does not repeat native parsing or rewrite original evaluation evidence.
