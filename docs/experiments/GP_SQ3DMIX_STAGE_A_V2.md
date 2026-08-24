# GP-SQ3D-Mix Stage-A-v2

Status: corrective implementation in progress; no Stage-A-v2 result has been recorded yet.

This experiment preserves the GP-SQ3D-Mix architecture and changes only the causal test and selection contract:

- fixed, global, same-command hard donors with moderate action distance and geometry-far selection;
- deterministic within-view spatial feature derangement;
- paired FlowMatchingState and dropout streams for real/hard/spatial losses;
- ratio-of-means paired bootstrap statistics;
- matched `projected_residual` and `gated_residual` variants;
- immutable utility, causal-gap, residual-distribution, gradient, alpha, and retention gates.

Runtime results are written to the machine-local Stage-A-v2 evaluation root first. This file must not contain placeholder measurements; it is updated only after a completed run.
