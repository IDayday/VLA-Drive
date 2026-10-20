# Independent review correction: repair outcome observes both trajectories

The initial C `absolute_feedback` repair head read only the Base self-pair features, although repair labels change with the proposed modification. Independent G0-author review correctly identified that this made its pair-specific repair task unidentifiable.

Every mode now passes a signed two-branch feature difference to the same existing repair head. Absolute modes use after-minus-before single-trajectory features; paired modes use one half of forward-minus-reversed pair features. No parameter, initialization, context source or training budget is added. Tests keep the Base/context fixed and change only the proposed trajectory: C's repair logit and modified-trajectory input gradient must change. Pair swap reverses its directional logit relative to the learned identity bias.

Focused CPU rerun:30 passed, one existing pynvml deprecation warning,7.93s; full log `G1_REPAIR_HEAD_REVIEW_TEST.log`. This corrects the pre-training mathematical contract, with no actual trained S0/G1 checkpoint rewritten and no claim of real repair prediction accuracy.
