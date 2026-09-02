# Scorer Navtrain-to-Navtest Domain-Shift Audit

This report is diagnostic only. Navtest factors are not used to tune or train a scorer.

## Candidate-bank and Base-scorer difficulty

| Split | Scenes | Logs | Base PDMS | Best-64 | Regret | Top-16 pairwise | Base oracle hit |
|---|---:|---:|---:|---:|---:|---:|---:|
| navtrain_train | 85109 | 101 | 0.955618 | 0.987749 | 0.032131 | 0.6670 | 0.0280 |
| navtrain_validation | 18179 | 61 | 0.931313 | 0.980300 | 0.048988 | 0.6352 | 0.0272 |
| navtest | 12146 | 136 | 0.911493 | 0.983129 | 0.071635 | 0.5961 | 0.0273 |

## Current-feature domain probes

A linear classifier is trained on current-observation scene/ego/scorer features and evaluated on held-out logs. AUROC near 0.5 means the domains are hard to distinguish; high AUROC indicates representation shift.

| Domains | Held-out-log AUROC | Accuracy | Train scenes | Test scenes |
|---|---:|---:|---:|---:|
| navtrain_train vs navtrain_validation | 0.7504 | 0.6900 | 18755 | 5245 |
| navtrain_train vs navtest | 0.6456 | 0.6169 | 19432 | 4568 |
| navtrain_validation vs navtest | 0.6461 | 0.5998 | 18290 | 5710 |

## Interpretation

Navtest Base scorer regret is 1.46x the Navtrain-validation regret, while the best-of-64 ceiling remains high. The held-out-log current-feature domain AUROC is 0.646. This separates candidate-generation headroom from scorer-domain generalization: new rankers must be selected by multi-domain/worst-fold Navtrain criteria rather than a single official validation split.
