# Qwen-World IL checkpoints and later RL observations

Full12146 scenes, canonical NAVSIM v1.1, seed2101, single clean-x K10 shift5 ODE. IL observations do not change the fixed IL40k RL parent. Test is historically observed.

| Snapshot | Updates | PDMS | Change vs IL40k (points) | Change vs previous same RL (points) |
|---|---:|---:|---:|---:|
| WORLD_IL_053000 | 53000 | 88.542266 | +0.569884 | — |
| WORLD_IL_050000 | 50000 | 87.953895 | -0.018488 | — |
| WORLD_IL_045000 | 45000 | 86.984807 | -0.987575 | — |
| WORLD_IL_035000 | 35000 | 87.016552 | -0.955831 | — |
| WORLD_IL_030000 | 30000 | 87.146897 | -0.825485 | — |
| ATTENTION_RL_032500 | 32500 | 90.466178 | +2.493795 | -0.101066 |
| MEMORY_RL_022100 | 22100 | 90.474582 | +2.502200 | -0.187076 |
| TIME_RL_043800 | 43800 | 89.700919 | +1.728537 | +0.331256 |

The previous-same-RL comparisons use full paired scenes/logs under the same protocol. A lower point estimate is an observed rollback; interpret its physical-log bootstrap interval separately. This is not multiple training-seed evidence. Different RL update budgets do not establish method superiority.

All metric components, zero-score and safety transfers, and paired95% intervals are in the per-model JSON. Raw worker costs and scene records are retained at /mnt/project/qwen-world-postcot-20261025/additional_IL_RL_NAVTEST_20261028T013728Z. No teacher/search/scorer chooses a deployed output; every observation is one candidate.
