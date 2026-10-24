# 已准入教师的抽样期望与原策略参考

本补充在[准入覆盖审计](admission_v1.md)基础上，只读同一批封存TRAIN候选，计算实际
`choose_teacher`使用的score/0.1 FP32 softmax的抽样期望。没有新增模型前向或评分，
没有改变candidate0准入规则、teacher库或已经运行的S2/S3。

必须区分：

- 准入reference：每个scene共用candidate0，teacher必须比它高至少0.005且score≥0.9。
- 发现和本表比较reference：固定5个独立root的score均分。
- E_softmax[R]：已经准入的最多4条teacher按训练中的场景内概率抽样的平均reward，
  不是best-of-K，也不是更新后模型的reward。

| 方法 | 已覆盖237场景E_softmax[R]（0–100） | 同237场景五ref均分 | 覆盖场景目标差（PDMS点） | 全512场景目标差，缺teacher按Δ=0 |
|---|---:|---:|---:|---:|
| F2_PDMS | 95.24488 | 92.93994 | +2.30494 | +1.06694 |
| F3_FEEDBACK | 95.24601 | 92.93994 | +2.30607 | +1.06746 |

两个方法的覆盖人口完全相同。F3只有1个scene中的3条teacher与F2不同，
因此E_softmax之差也很小。两个方法的全512场景五ref均分均为94.02257。

缺teacher的275场景是**零差值记账**，不是reward0：参考填充后的账面期望约95.09。
这个填充只用于说明目标质量与覆盖损失；实际训练在这些场景执行GT FM，而不是冻结reference，
因此不能由此保证模型保住它们，也不能把约+1.067点视为可兑现收益、理论上界或实际部署分数。
β=2也不意味着将这个reward差再乘2；它是监督loss系数。

## 最佳teacher和实际抽样目标不同

每个方法237个准入场景中：

- 226个至少有一条teacher比五ref均分高0.005；11个的最佳teacher也达不到该幅度。
- 按真正softmax抽样的期望计算，212个达到0.005，25个不足，其中4个期望略低于五ref均分。
- 所有teacher期望仍比各自candidate0至少高0.005064；上述4个负差来自reference定义不同，
  不是违反已登记准入合同。

F3全人口目标记账的正向贡献约+1.06779点、负向贡献−0.000334点，净+1.06746点；
F2分别为+1.06727、−0.000334、净+1.06694点。它们只是已评分目标的有限集合证据。
这说明本轮存在账面可学习的收益，但是否能吸收，以及新增退化是否抵消收益，仍须看统一原先验单候选评价。

此前719条EP-only、23条安全恢复、6条其他质量变化是teacher条数；
约96.31%是EP-only的抽样概率质量占比，不能将其误读为96.31%的梯度或reward净增益。

完整聚合、分层和seal见 [v2 JSON](admission_v2_summary.json)。v1原摘要保留。
本次实际执行入口仍为 `feedback_analyze_admission`，输出新版本
`/mnt/project/qwen-flow-three-routes-20261024/feedback/analysis/admission_v2`。
