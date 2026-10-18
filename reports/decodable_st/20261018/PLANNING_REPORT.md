# 规划：真实工程链已跑，效能尚未验证

`planner_engineering_v1`从空间tiny的第2步检查点开始，只训练reader两步。使用真实当前图像、原FM repeat8、独立原噪声/time分布；原10步Euler与canonical评分器不改。这个工程run不是1000步表征后的P3，更不是匹配预算正式结果。

读出源确为学生自身预测context：query/readout生成完整BEV，64个memory token被原action hidden读取，零初始化residual加回原条件，再交给原DiT。教师不在规划前向；没有GT轨迹reference、候选搜索或oracle择优。

独立导出后实际评分8个开发场景、1log：

| 指标 | 同模型无reader q0 | reader后的q_final |
|---|---:|---:|
| PDMS /100 | 88.4651 | 88.1999 |
| NC/DAC/TTC/Comfort/DDC | 均1 | 均1 |
| EP | .723163 | .716797 |
| 零分 | 0/8 | 0/8 |
| 异常/失败 | 0 | 0 |

零→非零0，非零→零0。单log不能建立有效log-cluster区间，且两步训练不足以解释方法优劣。这组数据只证明真实训练、纯当前部署、轨迹导出及官方仿真/评分链连通；下降也保留，没有筛场景或调evaluator。

正式共同终点、完整1696开发集、Navtest、稳定高分退化/动态进入/低进度分组、三训练seed均未运行。没有用Navtest选择阈值、权重或新方法。现有Navtest历史被反复观察的事实不变。

1000步RS另导出为**reader关闭**模型，用于表征训练后遗忘诊断；不能把它和上述两步reader模型混成“成熟空间表征已经改善规划”。当前不能声称规划增益或安全收益。
