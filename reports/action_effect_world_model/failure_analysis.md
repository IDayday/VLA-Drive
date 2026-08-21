# Gate-3 failure and case analysis

Cases are selected deterministically from the full fixed test/IDM subsets, including unfavorable deltas; they are diagnostics rather than causal counterfactuals.

## Factual prediction with weak action sensitivity

| Scene | Candidate pair | Pair type | Safety boundary | Geometry | Consequence | Selection delta |
|---|---|---|---:|---:|---:|---:|
| `205eaba8a7f95a1a` | `654f39e835bf840c` / `ab8307cddc6438d7` | effect_divergent | False | 2.5550 | 1.7907 | 162097.6808 |
| `c2b4f95be2855a57` | `908a38a8b807d99a` / `ddff14c9048da9d7` | effect_divergent | False | 2.4748 | 1.9316 | 88982.8112 |
| `cb3d52c845ec589e` | `7b597e5516e55928` / `a48b1ebb58b2a541` | effect_divergent | False | 2.0504 | 1.8184 | 87761.6884 |
| `590a88cf27a85e4c` | `10202114fafdfc58` / `227476a832caee9f` | effect_divergent | False | 2.0715 | 1.5925 | 79329.2309 |
| `3af9c4278e835280` | `4bc94acc9c4e3f0d` / `71997c2286b305bd` | effect_divergent | False | 2.7636 | 2.3728 | 66150.6889 |

## Global separation over-separates equivalent actions

| Scene | Candidate pair | Pair type | Safety boundary | Geometry | Consequence | Selection delta |
|---|---|---|---:|---:|---:|---:|
| `3cac5230a7e45054` | `02a0c7e85322d681` / `723957b0ee1f9efb` | effect_equivalent | False | 0.9843 | 0.0820 | 0.3460 |
| `3cac5230a7e45054` | `02a0c7e85322d681` / `bfd195b951803d87` | effect_equivalent | False | 0.7382 | 0.0392 | 0.3274 |
| `3cac5230a7e45054` | `02a0c7e85322d681` / `a74c9b37bc1d8c93` | effect_equivalent | False | 0.7081 | 0.0398 | 0.3234 |
| `cd1c3b256dbb58a1` | `0103210ac90f17e0` / `abda296437c249fc` | effect_equivalent | False | 0.9843 | 0.0410 | 0.3095 |
| `d600098375e45a90` | `b8db1f197457829c` / `d92d37e274dea569` | effect_equivalent | False | 0.7429 | 0.0870 | 0.3082 |

## AEE safety-boundary diagnostics

| Scene | Candidate pair | Pair type | Safety boundary | Geometry | Consequence | Selection delta |
|---|---|---|---:|---:|---:|---:|
| `a0ab4777d8245e01` | `639887284383f1ae` / `f71215ca121df8dc` | effect_divergent | True | 0.5054 | 2.0950 | 0.6942 |
| `a0ab4777d8245e01` | `52572f1e15fc00b0` / `f71215ca121df8dc` | effect_divergent | True | 0.4974 | 2.0933 | 0.6808 |
| `a0ab4777d8245e01` | `f678a1033e9454b9` / `f71215ca121df8dc` | effect_divergent | True | 0.5646 | 2.1029 | 0.6623 |
| `a0ab4777d8245e01` | `06e7e74fce73ee7f` / `f71215ca121df8dc` | effect_divergent | True | 0.7139 | 2.0752 | 0.5872 |
| `a0ab4777d8245e01` | `6efd6ade41cfde52` / `f71215ca121df8dc` | effect_divergent | True | 0.7461 | 2.0985 | 0.5829 |

## Log-replay / reactive-model conflicts

| Scene | Candidate pair | Pair type | Safety boundary | Geometry | Consequence | Selection delta |
|---|---|---|---:|---:|---:|---:|
| `9b56207d416a5f74` | `516d0c1a89d87a28` / `f6a7a25ec5e55bce` | ambiguous | True | 0.0091 | 2.0752 | nan |
| `2295480487565083` | `28c4ef1673abf1a0` / `e3c53746222ca3f2` | ambiguous | True | 0.0136 | 2.0567 | nan |
| `b649db17afea5a36` | `294acddde33d7b12` / `587e2e56dc9e2e00` | ambiguous | False | 0.0184 | 0.1779 | nan |
| `7a46488aa2d05c51` | `7681d864807ffa91` / `b92225d3cd6dc3ce` | ambiguous | True | 0.0202 | 2.1849 | nan |
| `7a46488aa2d05c51` | `7681d864807ffa91` / `ba26f52775dde751` | ambiguous | True | 0.0205 | 2.1942 | nan |

## World metric improved but planning did not

Not evaluated by instruction: no world loss was connected to Qwen+DiT, no planning training was run, and no PDMS/EPDMS value is populated. Therefore this delivery makes no world-to-planning transfer or gradient-conflict claim.
