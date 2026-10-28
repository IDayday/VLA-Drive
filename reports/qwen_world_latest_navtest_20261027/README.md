# World latest IL/RL Navtest snapshots

Canonical NAVSIM v1.1; full12146 scenes; single clean-x K10 shift5 ODE; seed2101. All three RL policies share IL40k as parent. IL52k is a separate IL continuation, not a replacement RL control.

| Snapshot | Updates | Full Navtest PDMS |
|---|---:|---:|
| WORLD_IL_052000 | 52000 | 87.656609 |
| ATTENTION_RL_026600 | 26600 | 90.567244 |
| MEMORY_RL_018400 | 18400 | 90.661658 |
| TIME_RL_035600 | 35600 | 89.369663 |

Paired IL40k comparison, native submetrics, new/recovered failures, G+/G−, log-cluster bootstrap and recorded worker costs are in the JSON files. This is one training/inference seed and different intermediate update budgets; it does not establish a matched endpoint ranking of methods. No candidate oracle, teacher or scorer selects the deployed trajectory. Test has been observed historically and these observations are not claimed blind.
