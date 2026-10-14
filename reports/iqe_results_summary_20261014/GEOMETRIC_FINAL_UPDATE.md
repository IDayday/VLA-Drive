# 几何 Scorer 完整结果补充

归档时间：2026-10-14T23:50:30.083019+00:00。这是 [23:32 结果快照](README.md) 之后完成的同一已注册实验；保留此前快照和原始证据，不重写历史。完整数据、状态、选择端点和校验见 [来源清单](GEOMETRIC_FINAL_SOURCE_MANIFEST.json)、[机器可读表](GEOMETRIC_FINAL_RESULTS.csv) 与 [检查结果](GEOMETRIC_FINAL_VERIFICATION.json)。

**6000 步训练、固定开发评估、信息利用诊断，以及最终 4000 步权重的真实图像/新进程研究包检查均已完成。开发集选轨为 90.5824，相对同一 Base 只高 0.0523 分；95% CI [-1.7163,+1.5967]。尚未获得可靠泛化收益。**

本次没有延长预算、训练另一组 Scorer、启动新 Navtest 或更新 ACTIVE。报告分数均为 PDMS×100，差值为百分点。Base70k、三个专家、候选池、原视觉/语言精度及评分协议与前表完全相同。stage_val 在读取本次 dev 前选择 4000 步，选择 hash `d51fa3f46798d9ad59f9db6b7bbd6ffe8b23725058788a81fe5efd9e42998d69`。所有 4,898,311 个新增 Scorer 参数从随机初始化开始，没有外部评分权重或诊断模型 warm-start。

## 完整角色结果

|集合|场景 / logs|Base|Oracle@4|原始 argmax|相对 Base|Oracle 捕获|高分→零分|
|---|---:|---:|---:|---:|---:|---:|---:|
|D_S，拟合集|16581 / 211|94.3310|97.5383|96.4392|+2.1082|65.73%|59|
|stage_val，选端点集|1024 / 53|91.3915|96.5215|91.2401|-0.1514|-2.95%|11|
|dev_report，固定开发集|1696 / 16|90.5302|95.3517|90.5824|+0.0523|1.08%|18|
|dev，独立校准 margin|1696 / 16|90.5302|95.3517|90.5302|0.0000|0.00%|0|

拟合集表现不能当泛化证据。D_S 原 Base 训练见过，但三个新增专家没有用它专精训练。开发只有 16 个 logs、一个 seed，且旧开发结果在本轮开展前已经观察；本轮是探索性跟进，不称独立盲测。

固定开发集上的道路/交互/进度 alone 仍分别为 77.5541 / 85.1406 / 81.4423，最佳固定策略为 Base。几何 Scorer 对最佳固定策略的收益同样只有 +0.0523 分。原 S11 是 90.2695，R00 是 90.6960；这些是同池不同训练方案的点估计，不能用挑选 dev 中最高值替代既定 stage_val 选模。

## 收益被哪些错误抵消

开发集正确替换收益 **+3.0223 分**，错误替换损失 **2.9701 分**，净值 +0.0523 分。候选机会 G=4.8215 分、选择 regret R=4.7692 分，核验 `selected−Base=G−R`。切换 1330/1696 场景（78.42%），Base 回退 21.58%。不是“基本都选回 Base”造成原始 argmax 的小收益。

- 高分（Base≥0.8）转零分：18/1696；总零分 63，Base 为 55。
- 新增 NC / DAC / TTC 失败：20 / 9 / 82，分母均为1696。
- 实际总 NC 失败 39 vs Base22，DAC 27 vs37，TTC 147 vs71。DAC 总数改善仍不能掩盖新增失败与其他分项退化。
- regret>0.1：141/1696；regret>0.5：42/1696。
- 真实零分候选的平均预测为0.5446；其中 Base 的55条真实零分平均预测仍为0.8812。严重失败高估没有解决。

selector_cal 的固定 margin 网格0/0.01/0.02/0.05/0.1没有一个满足已声明的零新增 NC/DAC 与高分转零分预算，故附加规则回退 always-Base。它不记为 Scorer 成功，也不证明未知场景零风险。

## 空间信息诊断与局限

同一 stage_val 和同一权重：正确V的 PDMS91.2401；打乱V为91.6734；屏蔽V为91.4241。诊断只读，不重新训练、选 checkpoint 或修改部署选择规则。它没有提供“正确空间配对提高留出选择质量”的证据；不能仅因输出变化或训练64场景过拟合改善就宣称空间读取有效。也不能据此直接断言空间输入没有信息。

[投影覆盖审计](../iqe_geometric_readout_20261014/VISIBILITY_RISK.md)进一步显示：stage_val 85次错误替换中，33次的8个轨迹点都有至少一个画面内投影，仍占错误损失40.65%；这些场景出现5次高分转零分。画面内不是无遮挡或未来可见，但失败不能只归结为轨迹走出视野。

目前证据支持“训练可拟合，但留出严重误选和选择泛化不足”。不支持继续用同一配方简单延长训练，也不支持声称几何空间读取已解决选轨。专家池的Oracle机会仍存在；后续研究应先固定该池分析真实风险监督与表征是否足够，不把多个变化混成一次收益归因。

## 实际完成与成本

训练固定6000更新、global batch128；自然流576000场景呈现、比较流192000 pair呈现。恢复段完成5998更新，墙钟4061.0629秒、4卡分配时间16244.2517 GPU秒；它不含最初2步、缓存/评分构建、后评估和整项目开销。头部训练VLM调用0，Base与专家更新0。没有把这个数字叫作部署推理成本。

最终选中模型已经导出研究 bundle，并由两个独立进程在4个固定真实 stage_val 观测的原始 batch 布局上 capture/verify。校验结果 `PASS`；实际每batch visual=1、language=1，Base Decoder=1、三个专家 Decoder各1、上下文投影=1、空间读取=1。完整张量差值、观测/模型/包hash见 [SELECTED_BUNDLE_REPORT.json](evidence/geometric_final/SELECTED_BUNDLE_REPORT.json)。它是兼容性检查，不增加科学样本或发布权限。

源码新增的投影风险审计有2个解析行为测试通过；这里只引用其实际记录，不重跑/合并此前测试数量。报告归档只读既有产物，未启动额外训练、模型评测或Navtest。已运行的收尾进程自然完成，未发终止信号。

## 复现入口

在工作区 `/mnt/project/DriveVLA-M0-iqe-robust-scorer-20261014`，已完成入口可校验后幂等复用：

```bash
PY=/root/miniconda3/envs/ddp/bin/python
CFG=configs/iqe/geometric_readout/full.yaml
$PY -m iqe.geometric_readout.evaluation --config "$CFG" --device cuda:0
$PY -m iqe.geometric_readout.bundle --config "$CFG"
$PY -m iqe.geometric_readout.bundle_check capture --config "$CFG" --device cuda:0
$PY -m iqe.geometric_readout.bundle_check verify --config "$CFG" --device cuda:0
$PY -m iqe.geometric_readout.visibility_risk --config "$CFG"
```

这些命令只应在当前允许资源上运行，不重复启动同目录的活动任务。完整训练/恢复命令见 [FULL_TRIAL_RUNBOOK](../iqe_geometric_readout_20261014/FULL_TRIAL_RUNBOOK.md)，本次没有通过恢复追加第6001步。限时主机不再使用。MOPD保持停止，Base/G2未受本次整理影响。

已完成测试集仍只有 Base及专家Oracle：Base89.1006、Oracle@4 94.2039，12146场景。**本文件没有 Scorer 的 Navtest 分数**，不把开发集90.5824冒充测试集结果。
