# 修正后局部违反与支持状态的独立统计

执行源码：`f662f0dad7a14123a8519efd9832c80c9d326bca`。
本次扩展现有 `evaluate_minimal_repair_behavior.py`，复用原 PlanCurve、GEOS、
measured_vector_v2 当前速度和 LocalEvidence。不引入 solver、动作选择或在线安全门控。

原计划与最终计划在相同的真实物理时间上读出局部事实。每个约束分为已知正例、
已知负例、未知三个状态，保存全部九种转移及 scene/log/对象/时间信息：

- 正例→负例：该采样时刻的已知违反被确认消除。
- 正例→未知：修正后证据不足，不计作修复。
- 负例→正例：已知新引入的违反。
- 未知→正例：新读出的已知违反，不能断言原计划没有违反。
- 未知→负例：新获得负证据，不能断言原计划的问题被修复。

连续的已标注对象距离另按两端有效性比较；距离增加不会自动生成“整个车身区域
确定无冲突”的负标签。既有 strict KEEP/REPAIR、全部场景和 canonical 分数保留。
稠密几何检查仍是计划曲线在固定日志环境下的关系，不是模拟执行或责任碰撞结果。
完整目标的独立 canonical 执行复核继续使用另一项既有工具。

同时新增各输出时刻的位置、位移向量、距离及包角后的 heading 改变量，记录首次修改
和最大修改时刻。逐位相同与数值未变化分开；1e-6m/rad 只用于数值报告，不用于
KEEP 判定或部署回退。历史模式默认行为不变；新自然协议自动附加这些统计。

## 实际验证

8 个新增测试和 18 个既有 LocalEvidence 测试共 26 项通过，包括：
支持丢失不算修复、未知不算原负例、九种转移互斥、真实 GEOS 对象避让但空闲证据
不足、物理时间一致、重复采样不增大 scene/log 数、包角及六/八点输出诊断。

复用了此前 4 个真实训练场景的不可变 R_DIFF_FUTURE 导出、同一 canonical 分数、
局部标签和合法当前状态；小头此前仅训练 6 次更新，仍属于
`engineering_historical_anchor_only`。没有新模型前向或训练，也没有重新评分和求解。

0–4 秒、0.05 秒间隔，每场景 81 个采样位置：

| 关系状态变化 | 采样位置数 | 独立场景 | logs |
|---|---:|---:|---:|
| 道路负例→负例 | 300 | 4 | 4 |
| 道路未知→负例 | 1 | 1 | 1 |
| 道路未知→未知 | 23 | 1 | 1 |
| 对象重叠未知→未知 | 324 | 4 | 4 |
| 确认违反消除／已知新增违反 | 0 | 0 | 0 |

唯一的道路“未知→负例”发生在 2.0 秒：原因为既有边界模糊带，而不是已知道路
越界。不能把这一转移叫作修复。对象关系的 324 个未知位置也不能写成 324 个安全负例。
场景可能出现在多类转移中，采样点彼此相关，不作为独立统计样本。

四条轨迹均有数值变化，最大位置变化分别约 0.012、0.031、0.022、0.013m；这只是
输出轨迹改变量，不是几何标签达到厘米精度的证据。原有 canonical 零分转移保持
修复 0、新增 0、双方非零 4，不从这四个工程样本推导模型收益。

本次调用约 21.008 秒墙钟、18.569 CPU-process 秒；其中 worker 实际 CPU 约
8.180 秒，已包含在外层 waited-child 计量中，不能相加。GPU 成本为零。

## 已执行入口

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python \
  tools/decision_local/evaluate_minimal_repair_behavior.py \
  --predictions /mnt/project/natural-error-local-artifacts/20261015/engineering_cached_R_DIFF_v1 \
  --scores /mnt/project/natural-error-local-artifacts/20261015/engineering_cached_R_DIFF_scores_v1 \
  --labels /mnt/project/natural-error-local-artifacts/20261015/engineering_local_lineage_4_v1 \
  --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_diagnostic4096_v1 \
  --anchor /mnt/project/decision-local-consequence-artifacts/20261014/ANCHOR.json \
  --output /mnt/project/natural-error-local-artifacts/20261015/engineering_local_behavior_4_v1 \
  --workers 2 --scope engineering
```

实际有界调用由 `timeout 180` 和原 child-CPU 计量器包裹，完整 argv 在
`LOCAL_FACT_BEHAVIOR_COMMAND_v1.json`，结果和成本在同名前缀的收据中。
默认正式模式仍要求完整开发人口和相同底座，不能用 engineering 开关冒充正式结果。

本次没有持续监控训练。正式新基础策略的 N/P/R 对比和规划结论仍需后续实验；
这里交付的是已在真实数据上执行的必要评价能力。
