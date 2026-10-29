# Qwen-World IL checkpoints and later RL observations

Full12146 scenes, canonical NAVSIM v1.1, seed2101, single clean-x K10 shift5 ODE. IL observations do not change the fixed IL40k RL parent. Test is historically observed.

| Snapshot | Updates | PDMS | Change vs IL40k (points) | Change vs previous same RL (points) |
|---|---:|---:|---:|---:|
| WORLD_IL_085000 | 85000 | 88.670582 | +0.698200 | +0.169773 |
| WORLD_IL_089000 | 89000 | 88.560053 | +0.587671 | -0.110529 |
| OPTIMIZED_IL_022000 | 22000 | 83.323871 | -4.648512 | -0.802403 |

The previous-same-RL comparisons use full paired scenes/logs under the same protocol. A lower point estimate is an observed rollback; interpret its physical-log bootstrap interval separately. This is not multiple training-seed evidence. Different RL update budgets do not establish method superiority.

All metric components, zero-score and safety transfers, and paired95% intervals are in the per-model JSON. Raw worker costs and scene records are retained at /mnt/project/qwen-world-postcot-20261025/new_checkpoints_NAVTEST_20261029T0825Z. No teacher/search/scorer chooses a deployed output; every observation is one candidate.
