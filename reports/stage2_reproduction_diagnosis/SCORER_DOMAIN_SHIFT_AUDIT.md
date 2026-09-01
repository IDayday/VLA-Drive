# Scorer Navtrain-to-Navtest Domain-Shift Audit

This report is diagnostic only. Navtest factors are not used to tune or train a scorer.

## Candidate-bank and Base-scorer difficulty

| Split | Scenes | Logs | Base PDMS | Best-64 | Regret | Top-16 pairwise | Base oracle hit |
|---|---:|---:|---:|---:|---:|---:|---:|
| navtrain_train | 85109 | 978 | 0.968072 | 0.990787 | 0.022715 | 0.7004 | 0.0258 |
| navtrain_validation | 18179 | 214 | 0.951612 | 0.988734 | 0.037122 | 0.6384 | 0.0199 |
| navtest | 12146 | 136 | 0.909594 | 0.984112 | 0.074518 | 0.6279 | 0.0271 |

## Current-feature domain probes

A linear classifier is trained on current-observation scene/ego/scorer features and evaluated on held-out logs. AUROC near 0.5 means the domains are hard to distinguish; high AUROC indicates representation shift.

| Domains | Held-out-log AUROC | Accuracy | Train scenes | Test scenes |
|---|---:|---:|---:|---:|
| navtrain_train vs navtrain_validation | 0.7771 | 0.7253 | 19092 | 4908 |
| navtrain_train vs navtest | 0.6775 | 0.6136 | 18514 | 5486 |
| navtrain_validation vs navtest | 0.4966 | 0.4850 | 19858 | 4142 |

## Interpretation

Navtest Base scorer regret is 2.01x the Navtrain-validation regret, while the best-of-64 ceiling remains high. The held-out-log current-feature domain AUROC is 0.497. This separates candidate-generation headroom from scorer-domain generalization: new rankers must be selected by multi-domain/worst-fold Navtrain criteria rather than a single official validation split.
