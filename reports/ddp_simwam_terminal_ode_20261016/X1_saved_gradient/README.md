# X1：已保存批次的梯度与方向保持诊断

2026-10-16。使用已完成 X1 的5k/10k/15k/20k原生检查点与对应已保存
behavior，每个阶段8个TRAIN场景、G8。四组场景顺序和冻结condition已与
原Canonical核对相同；训练后的链、候选reward和优势不同，不能称逐位
匹配的同一behavior实验。另用Canonical20k完成一次分析脚本回归。

登记与数学检查：[registration.json](registration.json)、
[qualification.json](qualification.json)。完整结果：[summary.json](summary.json)。
本次没有新增轨迹、真实评分或optimizer更新。没有修改正在运行的
LAST_ODE，也没有改变任何已完成训练或部署结果。

## 先修正分析中的权重归属

原`diagnose_saved_batch.py`的梯度和总anchor均调用actor的真实权重，
但报告通道贡献时写死了`[.2,1,.5,.5]/2.2`。这个表达式对已有
SOURCE/CANONICAL成立，对X1的`[1,1,.5,.5]/3`不成立。

新诊断从actor实际的H×D权重矩阵计算各通道贡献，再聚合heading pair；
其和必须与实际anchor一致。没有修改actor、概率、loss或optimizer。
在运行X1诊断前完成8组FP64/FP32、两种通道权重与两种horizon权重检查。
真实Canonical20k回归的PG/anchor norm、cosine及clipping与旧记录完全
相同，通道贡献差异仅为求和舍入范围内。完整旧结果仍有效。

脚本也拒绝覆盖旧输出、检查原生checkpoint seal和run身份，并要求
Adam计数对应所请求collection完成四次inner后的边界。新增结果明确
标记：这里是更新后的参数对旧保存链的诊断，不是第一inner的ratio误差。

分析源码：`361ab35f6a1dee1d511ec34a61fc0875e2e2dee4`。

## 随机步骤9：梯度关系随批次变化

下表是每个阶段最后一份已保存全局batch。norm按原rank平均，再对相应
步骤求梯度；anchor列已乘beta50。完整7/8/9三步数据均保留在JSON中。

| X1检查点 | PG norm | beta50 anchor norm | cosine | PPO clip比例 |
|---|---:|---:|---:|---:|
| 5k | 0.036896 | 0.026731 | -0.3741 | 34.375% |
| 10k | 0.050214 | 0.013490 | -0.2931 | 21.875% |
| 15k | 0.034248 | 0.024397 | -0.1314 | 7.8125% |
| 20k | 0.024722 | 0.024122 | +0.2096 | 25.000% |

5k—15k这几批的末步存在冲突，20k则同向。不能用一个batch或两个loss
标量认定anchor一直“过强”或“无效”。这里的clipping发生在四次更新后，
与未更新时的数值ratio噪声不同；没有据此放宽clip或减小beta。

## 纵向保持实际起作用，但不等于整体分数更高

同一保存batch中，随机步骤9相对永久SFT reference的纵向转移均值RMS
偏移如下。单位是原归一化动作空间，不是最终米级行程，也不是ODE误差。

| 检查点 | 原Canonical | X1 |
|---|---:|---:|
| 5k | 0.013419 | 0.006556 |
| 10k | 0.012259 | 0.005628 |
| 15k | 0.013592 | 0.006762 |
| 20k | 0.017024 | 0.008154 |

四批都观察到纵向偏移减少。这支持X1的保持项确实改变了学习行为，
不能据此把全部安全或泛化变化归于该局部量。

全程40,000次scene曝光、320,000候选的训练随机策略汇总：

| 指标（0—100） | 原Canonical | X1 |
|---|---:|---:|
| PDMS | 82.7407 | 82.3545 |
| NC | 99.7716 | 99.7938 |
| DAC | 95.3309 | 95.5453 |
| TTC | 97.4997 | 97.9278 |
| EP | 67.9604 | 66.2370 |
| NC/DAC/TTC联合通过率 | 93.0656 | 93.6806 |
| 有PG的group比例 | 99.6350 | 99.6500 |

这是一种安全/进度权衡。最后5k更新也使用相同窗口重新聚合，PDMS为
83.0436/82.6006，联合通过率93.2513%/93.9025%；没有将旧5000更新窗口
与新500更新窗口直接比较。训练人口偏困难、低进度，随机G8均值不能当
作部署DEV或Navtest成绩。原ODE是否净获益，以完整单候选评价为准。

## 成本和执行边界

一个Canonical回归加四个X1诊断，合计5份保存全局batch、40次已保存
scene-group重算；0新candidate、0评分、0更新。累计共享GPU worker wall
约154.30秒，单worker峰值分配显存7,369,116,160 bytes。四个X1诊断在
当时仍授权且有实测余量的tsinghua-zt4 GPU0—3并行完成；没有抢占其他任务。
worker wall之和不等于端到端wall，也不是独占GPU计算时间。

真实入口（输出须为新文件）：

```bash
python -m scripts.analysis.simwam_recipe.diagnose_saved_batch \
  --run /mnt/project/simwam-x-anchor-20261015/runs/R_CANONICAL_X1_seed42_v1 \
  --checkpoint /mnt/project/simwam-x-anchor-20261015/runs/R_CANONICAL_X1_seed42_v1/checkpoints/update020000_inner0 \
  --collection 5000 --output "$NEW_X1_GRADIENT_REPORT"
```

5k/10k/15k使用对应checkpoint及collection1250/2500/3750。分析保留源
部分随机步骤及dimension-mean/clamp代理的理论边界，不将其称为整条
混合链的无偏梯度。现阶段继续既定LAST_ODE和固定终点评价，不自动
追加LR、noise、reward、anchor或seed矩阵。
