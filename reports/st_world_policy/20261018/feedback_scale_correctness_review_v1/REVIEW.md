# Independent correctness review of derived rollout-scale evidence

This review concerns `tools/st_world_policy/analyze_feedback_scale.py`, a read-only analysis of the already verified compact C1 TRAIN receipts. It does not approve a new advantage rule, training, profile, model execution, scoring or resource allocation. Exact source and input pins, final status and executed test results are in `REVIEW.json`.

## Population and numerical meaning

The code uses saved original FP32 centered advantages, not new rewards or a reconstruction from aggregate means. Every JSON value must round-trip exactly through float32. Finite values then enter Python FP64 arithmetic with `math.fsum`: global mean, a second pass over squared deviations, division by N−1 and square root. N includes all four candidates of every scene, including zeros; 32 scenes give 128 and the 24-scene tail gives 96. No MC, trajectory-time or Flow dimension enters this count. The reported gain is 1/(std+1e−10), with epsilon outside the square root.

For all-zero updates, standard deviation is zero and gain is null: the numeric 1e10 value is not presented as an actual policy amplification. Zero advantage energy also gives a null bound fraction. Tiny positive standard deviation, nonzero constant groups, zero-variance nonzero input and other numeric exceptions are explicitly flagged. The tool describes anomalies; it does not suppress them or repair a candidate normalization implementation. Incomplete G4 rows, support mismatches, duplicate scenes within one update and nonfinite compact advantages fail.

The per-update energy bound N·δ² is a conservative bound from the primary maximum residual, not an attribution of how much gradient or reward information came from roundoff. A ratio at least one is unresolved, not proof of roundoff dominance. The final code retains per-update fraction distributions and additionally reports sum advantage energy, sum rounding-energy bounds and their ratio. The ratio of these sums is distinct from the equal-update average of ratios. Empty strata and zero total energy retain null denominators.

All reported quantiles are exact pooled per-update values using the inherited linear q·(n−1) rule. They are not averages of shard quantiles. Standard deviation is not replaced by pooled whole-training RMS. The four policy-version windows remain [0,1024), [1024,2048), [2048,3072), and [3072,4096).

## Support and exposure

Event-present strata and metric-discriminative strata are distinct. An update can contain a road/collision/TTC event with no corresponding between-candidate metric variation. `safe_EP_only_update` requires nonzero C1 support, every active group marked EP-only, and no D/C/T event anywhere in the update. Strata can overlap; they are not a partition whose counts should be added.

The population of each update stratum includes every scene in its selected updates. Unique scene/log and active-scene/log unions are reported separately from repeated scene/candidate presentations. A scene changing its log identity is rejected. These counts do not establish independent samples or causal gradient contributions. D/C/T support comes from counterfactual metric labels on C1 trajectories; it is not a result for separately trained C2/C3/C4 policies. Frequent EP-only active groups do not by themselves prove that full-training policy gradients or gated score energy are EP-dominated.

## Identity and read-only joins

The final tool verifies and executes the exact bytes of the imported primary helper, then pins the approved primary plan, its producer source and the final COMPLETE RESULT. It takes existing lifecycle locks without creating or writing those locks, checks all eight shard completion hashes, compact document checksums, inventory and original payload/COMPLETE pins, exact policy-version coverage and expected presentation counts, and reconciles nonzero support with the primary RESULT. This consumes the primary analysis' existing verified facts; it does not reload or re-score original payloads.

The primary RESULT hash was supplied by Lead and is checked by production code. The independent reviewer did not open that actual input or any compact result for this task. Tests use synthetic sealed documents and temporary paths only. Source/helper hashes were read directly for binding.

Derived outputs must be a new namespace outside the primary output tree and original run, checked before any mkdir/write. Original budgets and deadlines are read without being reset. The derived read allowance is added to actual primary logical reads and must stay below the original overall ceiling. The tool has its own bounded once-only invocation, error receipt and nonzero failure exit; it refuses existing output rather than silently re-running or overwriting. Systems/resource limits are a separate independent review responsibility.

## Review changes and executed tests

Two correctness clarifications were requested while the code was still work in progress and are included in the final version: explicit ratio-of-summed energies, and hash verification of the helper bytes actually executed. The final primary RESULT is also explicitly pinned. Historical design review `247a412c` is unchanged.

The original synthetic suite and additional independent checks exercise exact float32 restoration, correction=1 at both batch sizes, the one-ULP centering residual, zero and constant groups, tiny-standard-deviation flags, invalid compact shapes/support, event versus discriminative support, window boundaries, unique versus exposure counts, and the difference between pooled energy ratios and equal-update ratio means. An independently constructed resealed duplicate policy boundary is rejected even when document and shard checksums are internally valid. Test logs and commands are retained.

No actual dataset, saved reward payload, model, image, GT, metric cache, GPU or scorer was read or run. Approval is limited to this fixed derived descriptive analysis; numerical interpretation and any subsequent implementation or training decision remain separate.
