# 官方事件时间监督：有限诊断

起点：`d97781a56dc55e879bedabc008fb7dd78d963964`。已有几何 Scorer 在 D_S 提升2.1082分、dev仅0.0523分，仍有18个高分转零分。所有既有矩阵已完成；本诊断不续训它们、不改变其规则或结果，也不自动启动另一个全量矩阵。

## 要区分的问题

目前整条轨迹的总分和分项监督没有告诉评分 Decoder 风险首次发生在哪个时刻。检验训练中的时间定位监督能否改善同一真实比较集的风险拟合和轨迹排序。小型拟合改善仍不能证明泛化；若不改善则不自动扩大训练。

[Designing Versatile Samples](https://arxiv.org/html/2609.01799v1)分别训练指标评分网络，用模拟后的真实结果监督，并强调边界候选；它没有证明我们的时间监督方案有效。[GTRS](https://arxiv.org/html/2506.06664v1)结合候选分布扩充和传感器增强；已有丰富银行不是其完整复现。当前不更换感知模型、不做不一致图像增强、不加载任何外部权重。

## 固定范围与数学边界

- 同一个Base70k、同一四候选部署池、同一原始丰富训练银行。
- 使用已在旧几何300步诊断中固定的64个D_S场景、相同顺序和采样seed。不是从dev或新结果挑选场景。
- 最多768个既有候选，4个CPU worker，每个请求最多2次尝试；无新增轨迹，逐候选重新加载官方参考上下文。
- 沿用原 NAVSIM v1.1 `pdm_score`。仿真完成后只读取碰撞首次事件、TTC首次违规时间、逐时刻不可行驶区域标记。全部总分、六分项及官方参考hash须与旧银行一致，预先固定容差1e-10。
- 每个原轨迹时刻监督“截至该时刻是否已有事件”。NC=0.5也有事件；TTC时间是官方TTC违规检测时间，不冒充物理碰撞时间；t=0事件计入首个时间段。原分项软标签保持不变。
- 这些仅是训练目标。context白名单不接纳事件、GT、分数或未来真值；不增加第二套编码器或额外VLM调用。

只增加同一评分Decoder时间token上的256→3线性辅助头，共771参数。原总分/分项头和推理选择规则不变，不把三个边际事件概率相乘冒充期望PDMS。

一次新训练、最多300更新；common参数随机初始化与已完成几何对照完全一致，不载入旧诊断权重。额外头使用独立seed=20261044，训练RNG和采样继承。保持global128、micro16、AdamW1e-4/wd0.01、warmup100后恒定lr、原value/components/rank损失，增加权重1的累计事件BCE。先scene内按有效候选/时刻归约，再scene均值；本诊断为原rich自然流阶段，没有额外pair加权。

单独输出真实拟合PDMS、候选MSE、事件Brier、固定0.5阈值的recall/false-positive，以及错配V的只读结果。无stage_val/dev/test选模或阈值搜索。训练端点不用于部署。数据错误或非有限立即停止并保留证据；没有自动加预算入口。

## 入口

工作区：`/mnt/project/DriveVLA-M0-iqe-robust-scorer-20261014`。只用永久授权资源，当前rl-zt2已只读核实空闲，限时主机禁用，Base/G2和其他任务保留。

```bash
PY=/root/miniconda3/envs/ddp/bin/python
OUT=/mnt/project/iqe-temporal-risk-20261014/diagnostic64_v1
$PY -m iqe.temporal_risk.data build \
  --source-campaign /mnt/project/iqe-specialist-scorer-study-20261014/base70k_v1 \
  --control-root /mnt/project/iqe-conditional-readout-20261014/v1 --output "$OUT"
$PY -m iqe.temporal_risk.diagnostic --output "$OUT" \
  --geometric-control /mnt/project/iqe-geometric-readout-20261014/v1/diagnostic --device cuda:0
# 仅对本实验实际中断点恢复，终态不追加更新：
$PY -m iqe.temporal_risk.diagnostic --output "$OUT" \
  --geometric-control /mnt/project/iqe-geometric-readout-20261014/v1/diagnostic --device cuda:0 --resume auto
```

当前6个行为单元测试通过：首事件时间、NC软失败、缺失/矛盾trace拒绝、有效分母、microbatch加和、空mask梯度、共用初始化/输出、单Decoder与标签白名单。真实标签提取与训练证据另写结果报告；此注册不预填PASS。
