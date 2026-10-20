# Read-only verified-pair effect embedding

G1 forward now additionally returns `effect_embedding[B,hidden_dim]` (default256). D/E/F use one half of the ordered-pair forward-minus-reverse feature difference. Absolute controls use after-minus-before single-trajectory features. It directly exposes a feature difference already computed for the repair head; no parameters, initialization, scene inputs, model calls or existing output calculations change.

For D, projecting this embedding by `mean_head.weight` without its canceled bias is algebraically the predicted modified-minus-base feedback. Equivalent FP32 expressions have different reduction order; tests predeclare rtol1e-5/atol2e-7 for this diagnostic identity, while embedding zero/swap and unchanged old output comparisons are exact.

This feature is not itself a calibrated physical quantity. A downstream student may distill it only from the frozen teacher evaluated on actual verified simulated endpoint pairs. The teacher result must be detached. It must not be queried on arbitrary noisy Flow states or an unverified `epsilon+predicted_velocity` trajectory and then treated as trustworthy local physical supervision. The student receives only legal current frozen planning context, original noisy Flow state and time; no teacher embedding or true effect may become a deployment condition.

Focused CPU suite:32 passed, one existing pynvml warning,7.82s. A separate same-code/same-input reference comparison across all six controls confirms all pre-existing returned values and state_dict hashes remain bitwise identical; see `G1_EFFECT_EMBEDDING_EQUIVALENCE.json`. These are synthetic mathematical/compatibility checks, not evidence of a trained teacher's scientific effectiveness.
