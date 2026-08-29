# Six-Factor Oracle Primitive Action-Effect Gate

本报告只评价 oracle replay-grounded primitive action-effect representation 的候选排序价值；没有训练 forward/inverse/VLA/WoTE/trajectory 模型。

## 表一：基础 scorer

| Model | Selected score | Regret | Rank | Oracle capture | False-safe |
| --- | ---: | ---: | ---: | ---: | ---: |
| WoTE base selector | 0.503262 | 0.383652 | 25.21 | 0.000000 | 0.082031 |
| Trajectory-only | 0.431548 | 0.455367 | 46.37 | -2.226451 | 0.144531 |
| Direct current | 0.491391 | 0.395523 | 27.53 | -1.679306 | 0.125651 |
| Ego kinematic | 0.472073 | 0.414842 | 30.14 | -1.559029 | 0.167318 |

## 表二：Action-effect 分解

| Model | Selected score | Delta vs Direct | Delta vs Static | Regret | Pairwise acc. |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static primitive | 0.471182 | -0.020210 | 0.000000 | 0.415733 | 0.843820 |
| Shared logged future | 0.452857 | -0.038534 | -0.018325 | 0.434058 | 0.839245 |
| Dynamic replay primitive | 0.467108 | -0.024284 | -0.004074 | 0.419807 | 0.843303 |
| Full primitive | 0.478002 | -0.013389 | 0.006820 | 0.408913 | 0.851397 |
| Full engineered | 0.479288 | -0.012103 | 0.008106 | 0.407627 | 0.850965 |

## 表三：现有 latent world model

| Model | Selected score | Delta vs Direct | Regret | Candidate-specific |
| --- | ---: | ---: | ---: | ---: |
| WoTE full future | 0.547955 | 0.056563 | 0.338960 | Yes |
| WoTE environment-only | 0.473144 | -0.018247 | 0.413771 | Yes |

## 表四：干预实验

| Intervention | Selected score | Drop vs Full | Regret increase |
| --- | ---: | ---: | ---: |
| Full primitive | 0.478002 | 0.000000 | 0.000000 |
| Full effect swap | 0.433591 | 0.044411 | 0.044411 |
| Actor-only swap | 0.464549 | 0.013453 | 0.013453 |
| Static-only swap | 0.452646 | 0.025356 | 0.025356 |
| Scene-mean effect | 0.437966 | 0.040036 | 0.040036 |
| No interaction mask | 0.467606 | 0.010396 | 0.010396 |
| Interaction mask only | 0.471724 | 0.006278 | 0.006278 |

## 表五：六因子 MAE

| Model | NC | DAC | DDC | EP | TTC | Comfort |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct current | 0.147263 | 0.094771 | 0.171716 | 0.091030 | 0.167749 | 0.020737 |
| Static primitive | 0.163431 | 0.108237 | 0.177051 | 0.105973 | 0.191640 | 0.029394 |
| Full primitive | 0.137554 | 0.106336 | 0.178698 | 0.101363 | 0.169539 | 0.024838 |

## 核心比较

| Comparison | Score delta | Regret reduction | 95% CI | Status |
| --- | ---: | ---: | --- | --- |
| Full vs Direct | -0.013389 | -0.033852 | [-0.027701, 0.000683] | NOT_RUN |
| Full vs Static | 0.006820 | 0.016405 | [-0.006547, 0.020398] | NOT_RUN |
| Full vs Shared | 0.025145 | 0.057930 | [0.010942, 0.039189] | NOT_RUN |
| Full vs Swap | 0.044411 | 0.097968 | [0.028840, 0.059632] | NOT_RUN |

## Interaction subset

| Subset | Scenes | Full | Static | Delta | CI |
| --- | ---: | ---: | ---: | ---: | --- |
| all | 512 | 0.478002 | 0.471182 | 0.006820 | [-0.006547, 0.020398] |
| interaction_rich | 479 | 0.486479 | 0.479179 | 0.007300 | [-0.007141, 0.021547] |
| non_interaction | 33 | 0.354962 | 0.355106 | -0.000143 | [-0.037925, 0.032857] |

## 自动判定

- `final_verdict`: `DIRECT_BASELINE_UNDERFIT`
- `scientific_hypothesis_status`: `UNTESTED`
- 下一实验（本任务未运行）：repair the scorer before evaluating the direction

即使 verdict 为 `ORACLE_PRIMITIVE_ACTION_EFFECT_VIABLE`，其含义也仅限于：在单专家日志合法构造的 replay-grounded primitive effect 上，候选特定 action effect 含有超越相应控制组的规划信息，值得进入轻量 forward prediction 阶段；这不表示完整世界模型已经成立。

## 明确 NOT_RUN

`forward_effect`、`effect_predictor`、`inverse`、`trajectory_refinement`、`policy_distillation` 均未运行，因为任务边界要求停在 Oracle Effect Gate。
