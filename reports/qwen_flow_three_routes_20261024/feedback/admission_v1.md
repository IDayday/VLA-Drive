# 发现、教师准入与吸收之间的覆盖损失

> **验收口径更新（2026-10-25）**：本页所有DEV/VAL、TRAIN、候选搜索或拟合结果仅作诊断，不作为方法有效性、性能提升或方法排名依据。S1/S2/S3已完成完整NAVTEST，当前效力判断见[完整测试报告](full_navtest_feedback_20261025.md)；本页历史诊断不承担效力判断。原实验日期、数值和历史记录保留；详见[当前验收口径](navtest_acceptance_20261025.md)。

本补充只读已封存的512个TRAIN场景、共同五seed参考分层和F2/F3搜索结果，
没有新增搜索、模型前向、评分或optimizer更新，没有改变teacher、β或训练人口。
F2与F3的以下计数完全相同。原始逐场景审计留在私有资产，公开文件仅含聚合。

## 两种reference不能混为一个门槛

发现主口径是：存在NC/DAC/TTC均为1的候选，score比独立五reference的均分高至少0.005。
教师准入则按已登记的另一口径：比共同candidate0高至少0.005，且score至少0.9。
两者都使用0—1原始分数；0.005是0.5个PDMS点。

| 固定参考分层 | 场景数 | 相对五ref均分有安全改善 | 相对candidate0有安全改善 | 被0.9门槛挡住的candidate0改善场景 | 准入teacher场景 | teacher条数 |
|---|---:|---:|---:|---:|---:|---:|
| stable_high | 388 | 185 | 184 | 0 | 184 | 607 |
| other_or_unstable | 120 | 110 | 107 | 55 | 52 | 137 |
| stable_zero | 3 | 1 | 1 | 0 | 1 | 4 |
| low_nonzero | 1 | 0 | 0 | 0 | 0 | 0 |
| 全部 | 512 | 296 | 292 | 55 | 237 | 748 |

相对五ref均分发现改善的296个场景中，70个没有teacher：51个由于0.9绝对质量门槛，
19个由于candidate0增益门槛。准入237个场景中有226个teacher同时满足五ref改善口径；
另11个只满足candidate0口径，不能将它们重标为相对五ref均分的明确收益。

以candidate0这一正式准入reference计算，292个有安全改善场景中55个被绝对0.9门槛排除，
约18.84%；这确实减少了teacher覆盖，但不能把所有发现到准入的损失都算给质量门槛。
首条合格teacher不会被ADE去重移除；ADE0.2m门槛只控制后续目标的区别程度。

## 安全修复有没有进入训练

三个stable-zero场景中，只有一个被F2/F3在有限K32中找到严格可行解，最佳score为1.0。
它的四条合格teacher**全部已经进入S2/S3固定教师库**。因此该已发现修复若未被学生复现，
不能解释为被0.9门槛挡掉。剩余两个没有发现可行解，不能制造成功标签或宣称其理论不可解。

stable_high由五个独立reference有限采样定义；同一场景的另一个candidate0 root仍可能失败。
因此stable_high分层中出现candidate0→teacher安全恢复，并不与固定分层矛盾。

## 教师监督主要在教什么

按每条teacher相对共同candidate0分类，三类互斥：

| 类型 | teacher条数 | 每场景softmax抽样的累计质量 |
|---|---:|---:|
| 参考与teacher均严格安全，只有EP原生子项变化 | 719 | 228.2663 |
| 参考有NC/DAC/TTC违规，teacher全部通过 | 23 | 7.0000 |
| 已安全但有其他原生质量分项变化 | 6 | 1.7337 |
| 合计 | 748 | 237.0000 |

EP-only占条数96.12%，占有teacher场景的抽样质量约96.31%。安全恢复分布在7个场景。
若按全部512个均匀训练场景计算，安全恢复的teacher质量为7/512；没有teacher的275个场景
仍完整参与GT项，teacher项为0，不通过改变分母提高其权重。

“EP-only”只表示保存的其他原生分项不变，不证明几何行为完全相同，也不说明这些进度收益没有价值。
但本阶段不能声称教师库已经提供了大量不同类型的安全修复监督。

## 对后续结果解释的限制

S2/S3仍沿用固定237场景、748目标与同一个β=2。β在一次TRAIN梯度校准中触及上限，
未截断值为2.79144；实际初始teacher/GT梯度比约0.71648，而非目标1.0。
不能因后来效果不佳就无记录改变β、质量门槛或teacher人口。

教师池511/512个场景完全相同，S2/S3比较本身对物理反馈增量的判别力有限。
需要同时看S2/S3相对原base和GT-only S1的单候选结果，再区分：发现不足、准入覆盖不足、
学生未复现已准入修复，或更新时新增退化。训练FM loss下降或搜索oracle更高都不能代替该评价。

数值与seal见 [聚合JSON](admission_v1_summary.json)。复现命令：

```bash
CUDA_VISIBLE_DEVICES='' python -m scripts.analysis.qwen_flow_exploration.feedback_analyze_admission \
  --protocol /mnt/project/qwen-flow-three-routes-20261024/common_protocol.json \
  --reference-strata /mnt/project/qwen-flow-three-routes-20261024/common_bank/reference_strata.json \
  --search-root /mnt/project/qwen-flow-three-routes-20261024/feedback/search \
  --output /mnt/project/qwen-flow-three-routes-20261024/feedback/analysis/admission_v1
```

输出目录必须不存在，以保留既有分析版本。该命令不运行策略或评分器。
