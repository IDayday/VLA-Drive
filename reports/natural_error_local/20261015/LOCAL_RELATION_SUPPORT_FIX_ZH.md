# 局部后果敏感性：未知不等于不敏感

本次修改只涉及关系评价，执行源码为
`51654686a85bbe78541052b86c52532c5fdae520`。
没有更新模型、训练 loss、标签、扰动阈值或 canonical 评分器。
没有重新运行训练、预测或目标求解，也没有轮询基础训练。

## 发现的问题和修正范围

旧 `scene_metrics` 在任意一个时刻、一个判据有共同标签时，就允许把没有检出
明确变化的查询对归入 insensitive。LocalEvidence 的分时刻、分任务有效性下，
这会把“别的时刻仍未知”混入“已确认无关键变化”。旧统计保留，未覆盖。

新 LocalEvidence 评价使用三类：

- sensitive：任一可靠共同标签确认了原登记的道路、对象距离或重叠变化。
  其他时刻未知不抹去这个局部正证据。
- insensitive：两个查询在所有定义该分组的采样时刻和判据上均有支持，且没有上述变化。
- unknown：没有已确认变化，但比较支持不完整。仍报告其中有效位置的连续误差，
  不把整个查询或场景删除。

原空间模糊带单列，距离、道路和间距阈值均保持原值。
这里的完整支持仅指八个采样位置上的比较判据，不是 dense 4 秒修复资格或安全认证。
旧协议默认沿用旧分组；新自然错误协议自动采用新定义。
比较工具拒绝混合两种分组定义，必须对两个已保存预测共同重算后再比较。

## 已执行的真实数据验证

使用此前独立保存的 P_ABS/P_DIFF 预测，两者都只有 6 次工程训练更新。
同一批 32 个真实场景、32 个 logs，每场景 6 个 anchor–variant 查询对。
它们属于 `engineering_historical_anchor_only`，不是新 S0_A 的研究结果。

| 分组 | 原统计查询对 | 修正后查询对 |
|---|---:|---:|
| 近且已确认敏感 | 9 | 9 |
| 远且已确认敏感 | 0 | 0 |
| 近且不敏感 | 104 | 0 |
| 远且不敏感 | 50 | 0 |
| 近且比较支持未知 | 未分开报告 | 123 |
| 远且比较支持未知 | 未分开报告 | 54 |
| 空间模糊 | 6 | 6 |

原 154 个 insensitive 查询对全部缺少至少一项定义所需的支持；另外 23 对原先
因没有共同有效判据而未进入四象限统计，现同样明确列为 unknown。
已确认敏感的 9 对来自 9 个场景/9 个 logs。
近 unknown 覆盖 32 个场景，远 unknown 覆盖 27 个场景；两者有交叠，不能相加当独立人口。

直接重用原 FP32 预测张量，在 CPU 上比较新旧统计：所有非分组指标逐值相同，
包括绝对误差、差异误差、方向正确计数和冲突计数。原 CUDA 统计与本次 CPU 重放
的计数精确一致，浮点归约最大绝对差为 `6.103515625e-05`，出现在累加统计；
通过执行前固定的 `rtol=2e-6, atol=2e-5`，没有声称跨设备逐位一致。

7 个新增针对性测试与 18 个既有 LocalEvidence 测试共 25 项通过。
实际运行的配对比较保留 32 个场景，空 insensitive 组返回 `null` 而非零误差；
unknown 组有独立有效分母。256 次 bootstrap 只验证工程统计流程，不用于方法优劣结论。
两次重算合计约 12.613 CPU-process 秒、12.875 秒顺序墙钟；比较成本另见收据。
GPU 成本为零，没有生成新权重或特征缓存。

这项证据说明原四象限表不能把部分标签下的“未检出变化”解释为“不敏感”。
它不证明 ABS/DIFF 的规划收益，也不替代新基础策略的完整自然纠偏实验。

## 可执行命令与收据

两个已执行的完整命令在 `LOCAL_RELATION_SUPPORT_COMMANDS_v1.json`，输入和输出均为
独立共享存储路径。工具为：

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python \
  tools/decision_local/rescore_local_relation_support.py \
  --saved-evaluation /mnt/project/natural-error-local-artifacts/20261015/engineering_relations_v1/P_ABS \
  --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_diagnostic4096_v1 \
  --labels /mnt/project/natural-error-local-artifacts/20261015/engineering_local_R_32_v1 \
  --output /mnt/project/natural-error-local-artifacts/20261015/engineering_relation_support_v2/P_ABS \
  --max-seconds 120
```

已有输出是不可变记录，重跑必须指定新的空输出目录。
对应 P_DIFF 命令只改变保存预测和输出目录；没有重新调用模型。
结果身份、源预测 SHA、标签身份、执行源码及 CPU waited-child 成本分别保存在
`LOCAL_RELATION_SUPPORT_{P_ABS,P_DIFF}_v2.json` 和对应 `*_CPU_v1.json`。
完整配对统计在 `LOCAL_RELATION_SUPPORT_COMPARISON_v1.json`，作用域仍为工程验证。

按用户“无需监控训练”的最新要求，停止人工持续轮询与主动训练进度播报；
保留原训练、原生保存、预算保护和已登记的科学评价。此次没有刷新实时训练状态。
