# Query repeat analysis

The protected QueryFramework in iqe/query_base.py inherits source DDPForesight.forward at 1493deda247107efe076b9294863a6753391298a. It encodes current observations once, repeats planner condition and ego target eight times, constructs Flow noise/time, and passes them to QueryActionBridge, which ignores noise/time. Its result is renamed ego_il. This is the historical training definition; it remains callable and unchanged.

The recorded real P0 has local image batch4, LLM sequence1930, repeated Query batch32, eight action-state tokens and sixteen compressed scene tokens. The historical approximately16ms number is precisely scoped below; it is not a complete optimizer update.

New `iqe.performance.native_query` installs an explicit native_repeat1 forward only on the newly constructed performance model. It retains all four objectives, original masks, normalizers, channel/time means, global denominators and auxiliary warmup. It performs neither Flow-noise sampling nor repeated Query evaluation. The source worktree and locked22k checkpoint are unchanged.

Classification: **T2**, a changed stochastic training estimator. Dropout/DropPath remain enabled in production. Repeating identical deterministic samples and averaging is equivalent in a dropout-disabled diagnostic; averaging eight independently masked Query evaluations is not the same sampled gradient as one evaluation. RNG consumption changes as well. The CPU behavioral tests use the real small QueryActionBridge and check deterministic repeated-mean loss/gradient, stochastic disagreement and empty-mask graph connectivity. These are distinct from the real-image eight-GPU screen.

The runtime identity records legacy_repeat8 or native_repeat1. Resume rejects a change of this field. Cross-definition runs are execution migrations, never exact continuation. Same-definition GPU resume evidence is reported separately. No scientific conclusion is inferred from32 updates.

Actual32-update native screen: same32 global samples/update and identical ordered IDs asP0;26 steady updates, mean2.74310s,p502.70902s,p952.92416s. P0 mean2.75587s: a0.46% difference in a sequential short screen is not compelling throughput evidence. All four losses/gradients remained finite. Rank0 IL mean over32 updates was0.132976 vsP0 0.130057; these noisy short-run training values do not establish better or worse planning quality.

First-step sampled gradient cosine0.84983 and relativeL2 0.54589 demonstrate the changed random estimator, rather than a lossless removal of unused Flow plumbing. T1 near-equivalence acceptance does not apply to this T2 comparison. No200-step extension or live migration is made for repeat1 because the screen gives no meaningful speed benefit; its production dropout is retained and scientific behavior remains UNTESTED.

The historical training16ms metric was audited back to `profile_query_base.py` and `reference/rank_0.json`: it measures QueryActionBridge.forward at local repeated batch32 (4×8), including Q-Former/ego, Decoder/heads and IL forward, excluding E0/backward/optimizer. Its rank0 median was15.512928ms over onlythree usable timings. The other historical16.33ms number measuresfive cached inference Decoders atbatch1, and must not be conflated with this training region.
