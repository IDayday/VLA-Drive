# Query repeat analysis

The protected QueryFramework in iqe/query_base.py inherits source DDPForesight.forward at 1493deda247107efe076b9294863a6753391298a. It encodes current observations once, repeats planner condition and ego target eight times, constructs Flow noise/time, and passes them to QueryActionBridge, which ignores noise/time. Its result is renamed ego_il. This is the historical training definition; it remains callable and unchanged.

The recorded real P0 has local image batch4, LLM sequence1930, repeated Query batch32, eight action-state tokens and sixteen compressed scene tokens. Therefore the old approximately16ms isolated Decoder forward is not the full action training cost (Q-Former, eight stochastic replicas, IL, backward and shared representation training).

New `iqe.performance.native_query` installs an explicit native_repeat1 forward only on the newly constructed performance model. It retains all four objectives, original masks, normalizers, channel/time means, global denominators and auxiliary warmup. It performs neither Flow-noise sampling nor repeated Query evaluation. The source worktree and locked22k checkpoint are unchanged.

Classification: **T2**, a changed stochastic training estimator. Dropout/DropPath remain enabled in production. Repeating identical deterministic samples and averaging is equivalent in a dropout-disabled diagnostic; averaging eight independently masked Query evaluations is not the same sampled gradient as one evaluation. RNG consumption changes as well. The CPU behavioral tests use the real small QueryActionBridge and check deterministic repeated-mean loss/gradient, stochastic disagreement and empty-mask graph connectivity. These are distinct from the real-image eight-GPU screen.

The runtime identity records legacy_repeat8 or native_repeat1. Resume rejects a change of this field. Cross-definition runs are execution migrations, never exact continuation. Same-definition GPU resume evidence is reported separately. No scientific conclusion is inferred from32 updates.
