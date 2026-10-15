# Training versus unseen-log validation

| Policy | Scenes | Logs | PDMS /100 | Gain pp | 95% log CI pp | Regret pp | High→0 | New NC/DAC/TTC |
|---|---|---|---|---|---|---|---|---|
| M_scene fit | 13139 | 169 | 96.7722 | +2.5865 | [2.3409, 2.8269] | 0.6722 | 1 | 0/1/19 |
| M_scene internal val | 3442 | 42 | 93.6636 | -1.2222 | [-2.3328, -0.2080] | 4.2335 | 53 | 28/35/149 |
| M_scene overfit64 | 64 | 42 | 97.1872 | +5.0048 | [2.7161, 8.6442] | 0.1984 | 0 | 0/0/0 |
| M_set fit | 13139 | 169 | 95.3062 | +1.1206 | [0.8159, 1.3991] | 2.1381 | 37 | 6/34/114 |
| M_set internal val | 3442 | 42 | 94.5140 | -0.3718 | [-1.3778, 0.3969] | 3.3831 | 35 | 13/24/93 |
| M_set overfit64 | 64 | 42 | 96.6648 | +4.4824 | [2.2376, 8.1382] | 0.7208 | 0 | 0/0/0 |

Fit and internal-val numbers use the same internally selected model. The independent 64-scene models are implementation/fitting diagnostics only and never initialize either main model. New-model internal validation has 42 held-out logs; R00 saw those scenes and is only in-sample background there. Final refits see full D_S, so internal validation is no longer unseen for them. Final external checks are separate.
