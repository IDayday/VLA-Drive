# Qwen-World IL checkpoints and later RL observations

Full12146 scenes, canonical NAVSIM v1.1, seed2101, single clean-x K10 shift5 ODE. IL observations do not change the fixed IL40k RL parent. Test is historically observed.

| Snapshot | Updates | PDMS | Change vs IL40k (points) | Change vs previous same RL (points) |
|---|---:|---:|---:|---:|
| WORLD_IL_100000 | 100000 | 88.620904 | +0.648522 | +0.002331 |
| WORLD_IL_098000 | 98000 | 88.633642 | +0.661260 | +0.015069 |
| OPTIMIZED_IL_035000 | 35000 | 86.704773 | -1.267609 | +1.257239 |
| OPTIMIZED_IL_030000 | 30000 | 85.682545 | -2.289837 | +0.235011 |
| HUBER_IL_014000 | 14000 | 83.353855 | -4.618527 | — |
| TIME_RESIDUAL_IL_016000 | 16000 | 84.159052 | -3.813330 | — |
| TIME_RESIDUAL_IL_015000 | 15000 | 82.753299 | -5.219083 | — |
| ATTENTION_RL_021000 | 21000 | 90.468800 | +2.496418 | -0.112093 |
| MEMORY_RL_034000 | 34000 | 90.580115 | +2.607733 | -0.161195 |
| TIME_RL_039000 | 39000 | 89.696944 | +1.724561 | -0.142024 |

The previous-same-RL comparisons use full paired scenes/logs under the same protocol. A lower point estimate is an observed rollback; interpret its physical-log bootstrap interval separately. This is not multiple training-seed evidence. Different RL update budgets do not establish method superiority.

All metric components, zero-score and safety transfers, and paired95% intervals are in the per-model JSON. Raw worker costs and scene records are retained at /mnt/project/qwen-world-postcot-20261025/latest_midpoints_NAVTEST_20261030T012613Z. No teacher/search/scorer chooses a deployed output; every observation is one candidate.
