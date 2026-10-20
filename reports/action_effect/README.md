# 动作效果建模、显式修正与Flow内化：本轮结果

本轮完成了真实开发与有预算的训练、评分、部署接线验证。
**局部修复机会与训练集拟合成立，但尚未证明物理效果监督优于普通修正轨迹SFT。**
工程通过和研究命题得到支持是不同结论，逐项状态见[VERDICT.json](VERDICT.json)。

使用原始成熟S0（100k、DiT/Flow、Euler10），不是Query Base。
固定1024个TRAIN场景、1696个DEV场景/16个日志；未启动navtest，未更新ACTIVE，
未修改现有Base或其他实验锁定的权重。

| 方法 | DEV PDMS，百分制，三训练seed均值 |
|---|---:|
| 原S0，固定快照 | 91.388363 |
| A0：普通显式修正 | 91.389143 |
| A1：效果引导显式修正 | 91.224200 |
| B0：修正端点FM SFT | 91.408385 |
| B1-trajectory：minus-bridge离路径速度目标代理 | 91.357387 |
| B1-effect：同代理加真实物理效果监督 | 91.400210 |

这些微小均值变化不能代替配对日志置信区间和安全分析。A1每个seed新增4个高分转零分
场景；B1-effect未超过B0均值，且每个seed新增1个TTC失败。完整逐seed结果、匹配对照、
安全分项和不确定性见[FINAL_COMPARISON.md](FINAL_COMPARISON.md)。

G0找到7个经真实模拟确认的最小局部修复，只有6个可用于拟合、1个用于训练阶段留出。
没有原始碰撞失败正例，不能据此宣称掌握碰撞修复或广泛泛化。
G1确实学到改善/恶化的符号信息，但选择其最高预测改动仍造成真实质量损失；
绝对评分对照也没有被差异模型击败。见[G0](G0_REPAIR_OPPORTUNITY.md)、
[G1](G1_EFFECT_MODEL.md)、[路线A](G2_EXPLICIT_REFINEMENT.md)、
[路线B](G3_INTERNALIZED_FLOW.md)。

23个正式训练均完成1000步，主路线使用42/43/44三个独立学生训练seed，G1教师为固定seed42。
训练段合计4.2354 GPU小时，另计特征提取、严格加载、官方评分和验证。
31项官方评分完成，38368次场景评估对应38180个新增固定参考请求、0重试、0无效。
这些数字与整个历史项目累计开销不同，详见[训练开销](TRAINING_EXECUTION_MATRIX.csv)、
[评分开销](NATIVE_EVALUATION_COST_SUMMARY.json)。

当前CPU回归487通过；真实G1/B0单GPU中断恢复、双GPU损失归约、在线导出和标签独立性有
各自范围内的实测证据。没有把它们扩展成所有后端的bitwise保证。见
[恢复与测试](REGRESSION_AND_RESUME_REPORT.md)、[新增推理成本](LATENCY_AND_PARAMETER_SUPPLEMENT.md)。

入口、固定数据、实际checkpoint和恢复命令见[RUNBOOK.md](RUNBOOK.md)。
恢复原实验必须使用其实际执行的不可变源码；不能在新HEAD下静默替换旧运行身份。
NAVSIM v2、合法特权表征上界、匹配预算RL对照与ViGT/OCC扩展没有被标为通过。
下一步最有依据的是扩大跨日志的真实可修复监督支持，并先验证效果模型在干预支持域内的
排序、校准及提议效果；不应从当前DEV反复挑阈值或有利seed。
