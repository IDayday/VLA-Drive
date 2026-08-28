# Frozen EpisodeDrive Proposal Oracle Audit

| Group | Pairwise | Worst fold | Top-1 regret | Collision AUROC | TTC AUROC |
|---|---:|---:|---:|---:|---:|
| O3 | 0.7536 | 0.7460 | 0.1068 | 0.6366 | 0.6676 |
| O4 | 0.7239 | 0.7179 | 0.0981 | 0.7462 | 0.6806 |
| O5 | 0.8417 | 0.8267 | 0.0505 | 0.9877 | 0.8391 |
| O8 | 0.7670 | 0.7530 | 0.0632 | 0.9351 | 0.7763 |
| O9 | 0.7699 | 0.7558 | 0.0639 | 0.9370 | 0.7758 |
| O10 | 0.7156 | 0.7122 | 0.1014 | 0.7166 | 0.6939 |
| O11 | 0.6920 | 0.6862 | 0.1207 | 0.5896 | 0.5788 |
| O12 | 0.6394 | 0.6298 | 0.1012 | 0.5775 | 0.5694 |
| O13 | 0.7581 | 0.7538 | 0.1515 | 0.5916 | 0.6267 |

- Scenes/logs/K: 2,378 / 1,192 / 16
- Dynamic O8−O3 pairwise gain: 0.0133, log-bootstrap 95% CI [0.0119, 0.0339]
- Raw state O4−O3 / direct risk O5−O3 / state+recomputed-risk O9−O3: -0.0297 / 0.0880 / 0.0162
- Largest shuffled/noise/repeated-static control gain: 0.0044
- State/recomputed-risk retention: 1.2171568143131533
- Baseline-selected / best-of-16 mean official score: 0.9626 / 0.9842
- Best-of-16 headroom: 0.0216
- O8 ranker selected mean score: 0.9210; it does not beat the original scorer
- Ground-truth proposal inserted: no

This is an offline upper-bound analysis on frozen proposals. Official outcomes
are labels only and are not available to a deployable scorer.
