# Stratified log-replay / IDM agreement

The binary agreement target is whether each candidate pair has a hard-consequence difference under the corresponding traffic assumption. Ranking correlation uses the three-way score order (-1/tie/+1) and Kendall tau-b, so ties are retained.
The divergent subset is reconstructed from replay-side hard/soft thresholds before identifiability confidence can relabel a conflicting pair as ambiguous.
The confidence-support count uses the union of hard-relation and ranking disagreements inside the declared critical subsets.

| Subset | Scenes | Pairs | Hard disagreements | Rank disagreements | Raw agreement | Positive agreement | Cohen κ | MCC | Kendall τ-b |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all_pairs | 128 | 13353 | 125 | 212 | 0.9906 | 0.9564 | 0.9512 | 0.9518 | 0.9829 |
| divergent_pairs | 118 | 3696 | 102 | 115 | 0.9724 | 0.9641 | 0.9417 | 0.9433 | 0.9716 |
| safety_boundary_pairs | 51 | 542 | 65 | 68 | 0.8801 | 0.9362 | 0.0000 | 0.0000 | 0.8966 |
| at_least_one_unsafe_pairs | 55 | 2421 | 110 | 113 | 0.9546 | 0.9614 | 0.9064 | 0.9104 | 0.9501 |
| dynamic_interaction_scenes | 120 | 12438 | 125 | 212 | 0.9900 | 0.9537 | 0.9480 | 0.9488 | 0.9814 |
| low_ttc_scenes | 3 | 345 | 44 | 44 | 0.8725 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| idm_reacted_scenes | 102 | 10442 | 125 | 212 | 0.9880 | 0.9431 | 0.9364 | 0.9375 | 0.9776 |

## Confidence-weighting decision

- Reactive scenes: 128.
- Critical disagreements: 218 / 10879 (2.00%).
- Confidence weighting meaningful for the primary Phase-6 matrix: **True**.
- Action: run confidence_aee as a required method.

These labels remain `log_replay` and `reactive_model`; neither is described as a ground-truth counterfactual.
