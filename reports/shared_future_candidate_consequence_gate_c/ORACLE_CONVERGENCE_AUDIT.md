# Oracle Probe Convergence Audit

This calibration uses the fixed 2,000-scene, 40-log pilot store and all five
log-disjoint folds. It selects the training budget only; it does not decide Gate
C1 and is not mixed with the formal 45,378-scene result.

| Epochs | Group | MLP pairwise | MLP Top-1 regret | Linear pairwise |
|---:|---|---:|---:|---:|
| 8 | O3 static baseline | 0.8148 | 0.0390 | 0.7674 |
| 8 | O8 full dynamic | 0.8112 | 0.0304 | 0.7800 |
| 8 | O9 state + recomputed risk | 0.8127 | 0.0316 | 0.7797 |
| 15 | O3 static baseline | 0.8209 | 0.0418 | 0.7811 |
| 15 | O8 full dynamic | 0.8256 | 0.0322 | 0.7974 |
| 15 | O9 state + recomputed risk | 0.8255 | 0.0298 | 0.7973 |

At 15 epochs, O8−O3 is +0.0047 pairwise and −23.0% Top-1 regret; O9−O3
is +0.0046 pairwise and −28.7% regret. Fifteen epochs is therefore the locked
formal budget: the probe has converged enough to avoid an eight-epoch false
negative, while the predeclared +0.03 Gate C1 threshold remains unchanged.
