# Independent review correction: ranking probability mass

Independent Lead review found `LossContribution.mean` incorrectly clamped every denominator to at least1. Valid-count tasks have integer positive denominators, but ranking uses a real total absolute-gap mass that can lie between0 and1. Clamping that mass changed the declared normalized distribution objective.

The reducer now replaces only a zero denominator with1. A single valid pair with true gap0.02 must have exactly the ordinary pair softplus loss and gradient, since it is the entire comparison distribution. The new analytical test checks both, in addition to the existing0.95/0.02 relative contribution and microbatch/Gloo tests. This preserves gap weighting across scenes while normalizing its distribution correctly.

Current focused rerun:38 passed, one existing pynvml deprecation warning,8.04s. Full output: `G1_A_REVIEW_FIX_TEST.log`. This is a CPU mathematical correction before scientific model training, not a claim about real S0/NAVSIM gain.
