# Prefix-aware Soft Contrastive Label Audit

Every horizon uses only trajectory and consequence prefixes at or before that horizon.  No later-tail waypoint enters a short-horizon label.

| horizon_s | scenes | mean_gt_weight | mean_entropy | mean_effective_positive_count | mean_false_negative_count | mean_consequence_offdiagonal_mass |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 500.0000 | 0.1543 | 2.2575 | 9.5693 | 6.5380 | 0.9006 |
| 1.0 | 500.0000 | 0.1587 | 2.2781 | 9.7748 | 6.7500 | 0.8721 |
| 2.0 | 500.0000 | 0.1664 | 2.2960 | 9.9442 | 7.5180 | 0.8114 |
| 4.0 | 500.0000 | 0.1924 | 2.3139 | 10.1258 | 8.7460 | 0.8051 |

Same-prefix/different-tail checks: short-prefix equality rate `1.0`; longer-horizon separation rate `1.0`.  A hard one-hot label would treat every non-GT candidate as equally negative; `mean_false_negative_count` reports how many non-GT candidates retain at least 0.050 factual probability and would therefore be false negatives under that rule.

The K×K consequence labels combine standardized environment relations with actor features matched by stable track hash, valid masks, and an unmatched-set penalty. Binary risks and continuous distances are normalized separately through fixed per-feature scales.
