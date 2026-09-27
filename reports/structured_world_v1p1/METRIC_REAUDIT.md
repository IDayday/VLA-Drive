# P0 measurement re-audit

All 12 original banks / 1696 scenes each were read without VLM inference, training or PDM changes. Full v6 GT (no slot truncation), strict 2m threshold. Original metrics remain legacy. Six targeted matching tests passed, including swapped classes and brute-force maximum-cardinality verification.

| Run | Legacy recall % | Geometric filtered recall % | Fixed-K recall % | Class-constrained recall % | Failures |
|---|---:|---:|---:|---:|---:|
| A2_seed42 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| B_seed42 | 0.6089 | 1.8722 | 2.0800 | 1.0556 | 0 |
| B_seed43 | 0.0000 | 0.0000 | 1.1552 | 0.0000 | 0 |
| C_seed42 | 0.4410 | 0.7739 | 1.2007 | 0.2447 | 0 |
| C_seed43 | 0.0085 | 0.4353 | 2.6775 | 0.0341 | 0 |
| C_support_v6_seed42 | 0.0000 | 0.0000 | 1.3743 | 0.0000 | 0 |
| D_seed42 | 0.1593 | 0.3983 | 1.2833 | 0.1878 | 0 |
| E_adapter_support_v6_seed42 | 0.0000 | 0.0000 | 2.1539 | 0.0000 | 0 |
| E_seed42 | 0.5179 | 1.1552 | 2.6348 | 0.8280 | 0 |
| E_support_v6_seed42 | 0.0313 | 0.1679 | 2.4698 | 0.0996 | 0 |

C_support_v6 and E_adapter_support_v6 each output 108544 supported proposals, zero objectness-positive predictions: their filtered zero TP/FP is **no-object collapse**, not ROI exclusion or solely incorrect matching. Raw fixed-K recall is only 1.3743% / 2.1539%. Centres concentrate around x=15.69/17.25m and y=0.309/0.125m (medians). Mean no-object probabilities .7842/.7406. Both detection and slot localisation require rehabilitation.

A1 minus A0: PDMS -3.3397pp, DAC -3.3608pp, NC -.0590pp, TTC -.3538pp, EP -2.9916pp. Of 39 scenes with A0 >=.9 and A1=0, 38 fail DAC, 1 NC, 2 TTC (overlap). This is factor decomposition, not a proven causal explanation.

Real-loader action roundtrip on 64 training scenes: max xy 2.6445e-6m, yaw 2.3581e-7rad. Raw-log poses/time/cameras independently verified using NAVSIM's actual Quaternion.yaw_pitch_roll convention. An initial audit used matrix ZYX yaw and failed (up to .001912rad); failure artifacts retained. Correcting the **audit reference**, not the training labels or tolerances, gives max yaw 4.44e-16rad and xy0. Sampling intervals .498269–.500563s. This evidence covers 64 scenes, not the entire training corpus.

Scene CSV and full raw distributions: external artifact root /mnt/project/structured-world-v1p1-artifacts/20260927/metric_reaudit_863ab99. No private scene artifacts in Git. Code at metric-run launch: 863ab99; reported process-finish SHA may include independent audit-script additions; matching implementation was unchanged during that run.
