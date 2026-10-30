# Qwen-World IL checkpoints and later RL observations

Full12146 scenes, canonical NAVSIM v1.1, seed2101, single clean-x K10 shift5 ODE. IL observations do not change the fixed IL40k RL parent. Test is historically observed.

| Snapshot | Updates | PDMS | Change vs IL40k (points) | Change vs previous same RL (points) |
|---|---:|---:|---:|---:|
| OPTIMIZED_IL_050000 | 50000 | 87.032442 | -0.939940 | -0.272146 |
| TIME_RESIDUAL_IL_025000 | 25000 | 86.888757 | -1.083625 | +3.162683 |
| TIME_RESIDUAL_IL_027000 | 27000 | 86.247893 | -1.724489 | +2.521819 |
| HUBER_IL_013000 | 13000 | 81.092652 | -6.879731 | -2.261203 |

The previous-same-RL comparisons use full paired scenes/logs under the same protocol. A lower point estimate is an observed rollback; interpret its physical-log bootstrap interval separately. This is not multiple training-seed evidence. Different RL update budgets do not establish method superiority.

All metric components, zero-score and safety transfers, and paired95% intervals are in the per-model JSON. Raw worker costs and scene records are retained at /mnt/project/qwen-world-postcot-20261025/loss_later_NAVTEST_20261030T165924Z. No teacher/search/scorer chooses a deployed output; every observation is one candidate.
