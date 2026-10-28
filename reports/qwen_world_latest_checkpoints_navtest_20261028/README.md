# Qwen-World IL checkpoints and later RL observations

Full12146 scenes, canonical NAVSIM v1.1, seed2101, single clean-x K10 shift5 ODE. IL observations do not change the fixed IL40k RL parent. Test is historically observed.

| Snapshot | Updates | PDMS | Change vs IL40k (points) | Change vs previous same RL (points) |
|---|---:|---:|---:|---:|
| WORLD_IL_060000 | 60000 | 88.416776 | +0.444394 | -0.125490 |
| ATTENTION_RL_046100 | 46100 | 90.414274 | +2.441892 | -0.051903 |
| MEMORY_RL_033300 | 33300 | 90.741310 | +2.768928 | +0.266728 |
| TIME_RL_050000 | 50000 | 89.508000 | +1.535618 | -0.192919 |

The previous-same-RL comparisons use full paired scenes/logs under the same protocol. A lower point estimate is an observed rollback; interpret its physical-log bootstrap interval separately. This is not multiple training-seed evidence. Different RL update budgets do not establish method superiority.

All metric components, zero-score and safety transfers, and paired95% intervals are in the per-model JSON. Raw worker costs and scene records are retained at /mnt/project/qwen-world-postcot-20261025/latest_IL_RL_NAVTEST_20261028T075617Z. No teacher/search/scorer chooses a deployed output; every observation is one candidate.
