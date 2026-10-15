# 共同终点的轨迹误差、修改和固定场景分组

24项开发局部行为评价均已完成，每项1696场景。本文只汇总既有输出，没有新推理、几何检查或canonical评分。完整312组行和原聚合收据见[GROUPS.csv](behavior_results/GROUPS.csv)、[SUMMARY.json](behavior_results/SUMMARY.json)。分组可重叠，不是额外独立样本。

下表ADE/FDE均相对日志GT（米）；因此MIN可能更符合保持目标却离GT更远。`q0`在RECOVERY表内指实际扰动plan_in，而非另一个干净答案。

| 分布 | 未修正ADE | 未修正FDE |
|---|---:|---:|
| NATURAL | 0.66509 | 1.40041 |
| RECOVERY | 1.13240 | 1.49575 |

| 配方/组/头seed | 输入 | 输出ADE | 输出FDE | 平均改动m | 新增已知约束违反 |
|---|---|---:|---:|---:|---:|
| primary/LOG_NONE/42 | NATURAL | 0.66240 | 1.40118 | 0.07369 | 1/1696 |
| primary/LOG_CV/42 | NATURAL | 0.66413 | 1.40014 | 0.08405 | 1/1696 |
| primary/LOG_FUT/42 | NATURAL | 0.66375 | 1.40023 | 0.08665 | 0/1696 |
| primary/MIN_NONE/42 | NATURAL | 0.66499 | 1.40050 | 0.00274 | 0/1696 |
| primary/MIN_CV/42 | NATURAL | 0.66489 | 1.40047 | 0.01022 | 0/1696 |
| primary/MIN_FUT/42 | NATURAL | 0.66476 | 1.40047 | 0.00946 | 0/1696 |
| primary/LOG_FUT/43 | NATURAL | 0.66483 | 1.40253 | 0.08807 | 3/1696 |
| primary/MIN_CV/43 | NATURAL | 0.66499 | 1.40068 | 0.00678 | 0/1696 |
| primary/MIN_FUT/43 | NATURAL | 0.66425 | 1.40030 | 0.00839 | 0/1696 |
| primary/LOG_NONE/42 | RECOVERY | 0.66688 | 1.40753 | 0.72403 | 7/1696 |
| primary/LOG_CV/42 | RECOVERY | 0.66924 | 1.40502 | 0.72051 | 11/1696 |
| primary/LOG_FUT/42 | RECOVERY | 0.66810 | 1.40484 | 0.72542 | 8/1696 |
| primary/MIN_NONE/42 | RECOVERY | 1.12968 | 1.49158 | 0.00552 | 0/1696 |
| primary/MIN_CV/42 | RECOVERY | 1.12868 | 1.49301 | 0.01421 | 2/1696 |
| primary/MIN_FUT/42 | RECOVERY | 1.12856 | 1.49247 | 0.01335 | 0/1696 |
| primary/LOG_FUT/43 | RECOVERY | 0.67032 | 1.40838 | 0.73750 | 7/1696 |
| primary/MIN_CV/43 | RECOVERY | 1.12712 | 1.49069 | 0.01349 | 1/1696 |
| primary/MIN_FUT/43 | RECOVERY | 1.12720 | 1.49078 | 0.01406 | 1/1696 |
| balanced/MIN_NONE/42 | NATURAL | 0.66403 | 1.40285 | 0.04794 | 1/1696 |
| balanced/MIN_CV/42 | NATURAL | 0.66661 | 1.40403 | 0.05359 | 1/1696 |
| balanced/MIN_FUT/42 | NATURAL | 0.66623 | 1.40343 | 0.05392 | 1/1696 |
| balanced/MIN_NONE/42 | RECOVERY | 1.11774 | 1.47383 | 0.06085 | 4/1696 |
| balanced/MIN_CV/42 | RECOVERY | 1.12264 | 1.48155 | 0.06442 | 7/1696 |
| balanced/MIN_FUT/42 | RECOVERY | 1.12490 | 1.48711 | 0.06196 | 7/1696 |

以下固定分组只列seed42 MIN_CV/FUT原配方与一次平衡对照的ΔPDMS（pp），用于检查未来关系比较的范围；全部模型、指标及各组已有聚类区间在JSON中。极小分组的较大点估计不能被当成稳定效果。没有按这些分组选择模型或在线回退。

| 输入 | 分组 | 场景数 | 原CV | 原FUT | 平衡CV | 平衡FUT |
|---|---|---:|---:|---:|---:|---:|
| NATURAL | local_geometry_KEEP_non_GT_ADE_ge_0p5 | 140 | -0.65806 | -0.65584 | -1.25573 | -0.71300 |
| NATURAL | original_high_ge_0p9 | 1242 | -0.14560 | -0.14147 | -0.35709 | -0.28163 |
| NATURAL | original_low_le_0p5 | 44 | -0.00021 | -0.00008 | +10.65074 | +9.25841 |
| NATURAL | dynamic_enter | 7 | -0.00153 | -0.00180 | -0.00392 | +4.16549 |
| NATURAL | space_release | 106 | -0.00247 | -0.00063 | -0.04882 | -0.03059 |
| NATURAL | road_boundary | 406 | -0.11072 | -0.10636 | +0.01286 | -0.27944 |
| NATURAL | static_annotated_obstacle_near_input | 226 | -0.00740 | -0.26128 | -0.17207 | +0.59708 |
| NATURAL | low_speed | 151 | -0.00381 | +0.00125 | -0.00158 | -0.00354 |
| NATURAL | label_support_below_half | 394 | -0.11466 | -0.10790 | -0.29862 | -0.02116 |
| NATURAL | unresolved | 1399 | -0.06505 | -0.10217 | +0.18977 | +0.20163 |
| RECOVERY | local_geometry_KEEP_non_GT_ADE_ge_0p5 | 209 | +0.40791 | +0.41025 | +0.42781 | -0.20632 |
| RECOVERY | original_high_ge_0p9 | 1033 | -0.06020 | -0.05677 | -0.20478 | -0.14202 |
| RECOVERY | original_low_le_0p5 | 195 | +3.19089 | +2.16559 | +19.83671 | +17.57975 |
| RECOVERY | dynamic_enter | 29 | +1.43456 | +1.43846 | +18.85130 | +11.69627 |
| RECOVERY | space_release | 106 | -0.00335 | -0.00140 | +5.21151 | +3.05123 |
| RECOVERY | road_boundary | 461 | +1.12839 | +0.69763 | +4.45289 | +4.16978 |
| RECOVERY | static_annotated_obstacle_near_input | 282 | +0.53614 | +0.53872 | +3.74330 | +3.02374 |
| RECOVERY | low_speed | 151 | -0.11377 | -0.11079 | +0.44100 | +0.16369 |
| RECOVERY | label_support_below_half | 378 | +0.29968 | +0.03992 | +2.85357 | +2.19357 |
| RECOVERY | unresolved | 1417 | +0.37542 | +0.23752 | +2.52946 | +2.07062 |

分组使用既有算子[evaluate_minimal_repair_behavior.py](../../../tools/decision_local/evaluate_minimal_repair_behavior.py)：原高/低分阈值0.9/0.5，合理非GT代理为局部KEEP且输入对GT ADE≥0.5m；道路边界按有效道路余量≤0.5m，低速为当前速度范数<0.5m/s。dynamic_enter/space_release是沿输入查询当前与未来端点标注交叠变化，不认证其他车辆对ego修改的反事实反应。静态对象组遵循原算子的当前速度与邻近定义。

`new_known_constraint_violation`来自执行后密集局部几何/运动学检查，不能冒充官方责任碰撞数；未知仍单列，不能因没有已知违反而声称安全。KEEP也不独立证明路线或交通规则正确。`new_stop`只检测终点位移跨1m阈值，本文以endpoint transition命名，不能用全零推出没有真实停车或进度退化。全量canonical评分始终保留不确定/无解场景。

新增汇总器[summarize_repair_behavior.py](../../../tools/decision_local/summarize_repair_behavior.py)在真实现有24项记录上执行成功；只额外读取两种共同输入各1696条既有标量记录补齐原汇总漏列的q0 FDE，没有读取新GT图片、地图或重新生成标签。
