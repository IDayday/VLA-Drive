# Six-Factor Oracle Primitive Action-Effect Gate

本报告只评价 oracle replay-grounded primitive action-effect representation 的候选排序价值；没有训练 forward/inverse/VLA/WoTE/trajectory 模型。

## 实验契约

- Probe backbone：`matched_hybrid_v3`。
- matched-v3 的 A--L 变体均从相同 seed 的已验证 Direct scorer 初始化，并保持完全相同的可训练参数量。
- 评价集为预注册的 512-scene development slice；fresh Direct holdout 与 future-effect reserve 均未用于本 Gate。
- 本报告中的 six-factor score 是独立标签上的离线候选排序指标，不是 navtest PDMS。

## 表一：基础 scorer

| Model | Selected score | Regret | Rank | Oracle capture | False-safe |
| --- | ---: | ---: | ---: | ---: | ---: |
| WoTE base selector | 0.503262 | 0.383652 | 25.21 | 0.000000 | 0.082031 |
| Trajectory-only | 0.511395 | 0.375520 | 36.27 | -2.053728 | 0.158854 |
| Direct current | 0.614900 | 0.272015 | 17.19 | -1.099673 | 0.127604 |
| Ego kinematic | 0.609223 | 0.277691 | 18.17 | -0.976513 | 0.128906 |

## 表二：Action-effect 分解

| Model | Selected score | Delta vs Direct | Delta vs Static | Regret | Pairwise acc. |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static primitive | 0.621888 | 0.006988 | 0.000000 | 0.265026 | 0.870578 |
| Shared logged future | 0.615112 | 0.000212 | -0.006777 | 0.271803 | 0.870419 |
| Dynamic replay primitive | 0.635014 | 0.020114 | 0.013126 | 0.251901 | 0.873004 |
| Full primitive | 0.625714 | 0.010814 | 0.003826 | 0.261201 | 0.877475 |
| Full engineered | 0.626120 | 0.011220 | 0.004231 | 0.260795 | 0.875070 |

## 表三：现有 latent world model

| Model | Selected score | Delta vs Direct | Regret | Candidate-specific |
| --- | ---: | ---: | ---: | ---: |
| WoTE full future | 0.665140 | 0.050240 | 0.221775 | Yes |
| WoTE environment-only | 0.609661 | -0.005239 | 0.277253 | Yes |

## 表四：干预实验

| Intervention | Selected score | Drop vs Full | Regret increase |
| --- | ---: | ---: | ---: |
| Full primitive | 0.625714 | 0.000000 | 0.000000 |
| Full effect swap | 0.619430 | 0.006285 | 0.006285 |
| Actor-only swap | 0.619670 | 0.006044 | 0.006044 |
| Static-only swap | 0.627626 | -0.001912 | -0.001912 |
| Scene-mean effect | 0.618547 | 0.007167 | 0.007167 |
| No interaction mask | 0.640338 | -0.014624 | -0.014624 |
| Interaction mask only | 0.612653 | 0.013061 | 0.013061 |

## 表五：六因子 MAE

| Model | NC | DAC | DDC | EP | TTC | Comfort |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct current | 0.125102 | 0.068789 | 0.158051 | 0.081604 | 0.142565 | 0.011565 |
| Static primitive | 0.122389 | 0.066348 | 0.159198 | 0.079563 | 0.138805 | 0.014524 |
| Full primitive | 0.100998 | 0.063844 | 0.154830 | 0.078248 | 0.124078 | 0.011272 |

## 核心比较

| Comparison | Score delta | Regret reduction | 95% CI | Status |
| --- | ---: | ---: | --- | --- |
| Full vs Direct | 0.010814 | 0.039756 | [-0.000981, 0.022181] | FAIL |
| Full vs Static | 0.003826 | 0.014436 | [-0.008239, 0.015138] | FAIL |
| Full vs Shared | 0.010603 | 0.039008 | [-0.001191, 0.022519] | FAIL |
| Full vs Swap | 0.006285 | 0.023495 | [-0.003974, 0.016674] | FAIL |

## 诊断性观察

- Full primitive 相对 Direct 的三个 seed 增益为 `[0.018973, 0.021081, -0.007611]`；其中一个 seed 为负，且 regret reduction 未达到预注册的 10%。
- 去除 interaction mask 的正式 H checkpoint 相对 Direct 为 `0.025438`，95% CI `[0.013190, 0.037737]`；这是诊断信号，不替代必须由 G 通过的 primitive requirement。
- WoTE full-future 相对 Direct 为 `0.050240`，95% CI `[0.038418, 0.062656]`；environment-only 相对 Direct 为 `-0.005239`，95% CI `[-0.014132, 0.003504]`。
- Oracle capture 是逐 scene 比率；当 WoTE-to-oracle gap 很小时，该比率不受界，因此均值可能被少数负离群值主导。完整 mean/median/quantile 见 `oracle_capture.csv`。

## Interaction subset

| Subset | Scenes | Full | Static | Delta | CI |
| --- | ---: | ---: | ---: | ---: | --- |
| all | 512 | 0.625714 | 0.621888 | 0.003826 | [-0.008239, 0.015138] |
| interaction_rich | 479 | 0.631293 | 0.627469 | 0.003824 | [-0.007773, 0.015319] |
| non_interaction | 33 | 0.544741 | 0.540891 | 0.003850 | [-0.055276, 0.060614] |

## 自动判定

- `final_verdict`: `WOTE_LATENT_SIGNAL_ONLY`
- `scientific_hypothesis_status`: `PARTIAL`
- 下一实验（本任务未运行）：analyze useful WoTE latent information and action sensitivity

即使 verdict 为 `ORACLE_PRIMITIVE_ACTION_EFFECT_VIABLE`，其含义也仅限于：在单专家日志合法构造的 replay-grounded primitive effect 上，候选特定 action effect 含有超越相应控制组的规划信息，值得进入轻量 forward prediction 阶段；这不表示完整世界模型已经成立。

## 明确 NOT_RUN

`forward_effect`、`effect_predictor`、`inverse`、`trajectory_refinement`、`policy_distillation` 均未运行，因为任务边界要求停在 Oracle Effect Gate。
