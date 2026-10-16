# A_DDP_TIME：5k监督训练后的完整固定评价

本表是 **SFT阶段** 的变化，尚未进行本轮正式RL。DEV有1696场景/16个完整log；TRAIN诊断有512场景/343个log，其中含预先定向选择的困难场景，不能作为完整TRAIN人口的无偏估计。两种ODE均为seed2101、每场景单候选、原始观测入口、canonical NAVSIM v1.1；无搜索、GT选轨或best-of-two。

## PDMS：自身起点与5k

单位：0—100，差值与区间为百分点。

| 人口 | 部署网格 | 原始SFT起点 | 新SFT 5k | 差值 | log-cluster 95%区间 |
|---|---|---:|---:|---:|---|
| 完整DEV | E_NATIVE | 93.619237 | 93.864513 | +0.245277 | [−0.091581, +0.468069] |
| 完整DEV | E_SHIFT5 | 93.574749 | 93.550878 | −0.023871 | [−0.416578, +0.405509] |
| 固定TRAIN诊断 | E_NATIVE | 80.331593 | 80.973722 | +0.642129 | [−1.779500, +3.139474] |
| 固定TRAIN诊断 | E_SHIFT5 | 79.109938 | 79.877487 | +0.767549 | [−1.653884, +3.164524] |

原生ODE出现早期正向均值变化，两个网格尚未呈现一致、稳定的提升。所有区间仍跨零。该bootstrap只刻画当前固定训练seed/推理seed下的log抽样不确定性，不包含训练重复或推理seed变化。

DEV本身的原SFT已经超过92，不能将这里的93.8645写成最终RL或Navtest达到92。没有据此选择检查点或改预算。

## 安全转移

以下是相对于同网格固定原SFT的场景数；NC=0.5仍视为未完全通过。不能将任意负分差的场景计数等同事故数。

| 人口/网格 | NC 新增/恢复 | DAC 新增/恢复 | TTC 新增/恢复 | 非零→零 / 零→非零 | 原联合安全场景新增违规 |
|---|---:|---:|---:|---:|---:|
| DEV native | 0 / 0 | 1 / 5 | 4 / 1 | 1 / 5 | 5 / 1683 |
| DEV shift5 | 0 / 0 | 4 / 6 | 8 / 0 | 4 / 6 | 12 / 1678 |
| TRAIN诊断 native | 0 / 0 | 17 / 22 | 9 / 3 | 17 / 22 | 25 / 434 |
| TRAIN诊断 shift5 | 0 / 0 | 18 / 22 | 5 / 4 | 18 / 22 | 22 / 425 |

DAC恢复多于新增违规，但TTC仍出现净退化。DEV native 的联合安全场景总数仍为1683，却有5个原安全场景变为违规；相同总数不能证明逐场景安全保持。shift5新增违规更多，不能因为它与训练时间网格一致就预先认定其部署更好。

DEV native 的总正向收益G+=7.401889，总负向退化G−=3.241999；shift5分别为8.274253、8.679107。这里是逐场景**原始0—1 score差值之和**，不是百分点，也不是平均分。native达到0.005原始分以上的改善/退化场景为163/50；任意正差/负差为808/284。

## 解释边界与后续执行

该结果说明新时间采样/时间加权的监督继续训练能够改变原生ODE输出，暂时没有出现全局无法学习的迹象。但A同时增加了160,000次监督场景曝光，不能将变化单独归因于时间权重，更不能据此说明Flow-GRPO已被修复。

继续已登记的A 20k SFT、B 100k SFT及固定里程碑，不根据5k结果改LR、噪声、anchor、训练人口或评价网格。各自SFT终点之后重新固定自己的TRAIN困难清单，再从新的SFT开始相同R_SOURCE，单独测量RL−自身SFT。

本次完整DEV实际完成3392条正式单候选读出，另8条原入口一致性检查；job wall约365.53秒，累计worker wall约2898.81秒。TRAIN诊断分别为1024+8条、185.59秒、1459.10秒。worker wall包含共享设备驻留和等待，不等于独占GPU计算时间。

完整机器可读结果：[DEV](A_SFT_update005000_DEV.json)、[TRAIN诊断](A_SFT_update005000_TRAIN_DIAGNOSTIC.json)。全部场景payload经过hash验证；A初始基线复用由全参数/配置相等、原始观测双网格输出一致及原始payload校验支持，没有改写历史评分。

## 衔接代码的实际状态

新增调度代码固定在 `225b9af660cb0210208e2c46fc4216d0f151275b`：

- `dispatch_screen`：完整SFT的101,592场景单候选TRAIN筛选，按当前授权/容量派发，固定scene绑定seed及batch1。底层筛选已做两组真实4场景及不新增推理的恢复验收。
- `qualify_rl_endpoint`：等待自己的SFT终点与完整困难清单后，执行真实8卡continuous4 vs inner2恢复、全部原生状态比较、非零LoRA完整ODE合并和原始观测部署；**这项终点资格尚未执行**，不会用早期资格权重冒充正式初始化。
- `dispatch_rl`：要求上述真实8卡资格、自己的完整双网格SFT基线及全部source hashes匹配；从SFT重新初始化LoRA，而非从资格权重续训。中途SFT和资格模型的拒绝入口已经执行。
- 15项针对性CPU检查通过。它们验证人口、恢复/发布边界与队列依赖，不证明驾驶效果。
- 单一owner评价队列已在空闲边界交接到新版本，收纳登记的SFT和RL里程碑以及自身基线配对分析；不创建训练或Navtest任务。旧controller源目录保留，正常SFT源`4e0ce1f`及评价源`45d2318d`未热修改。

可执行入口已实现并完成CLI smoke；正式终点流程仍须真实执行，不能把这些入口的存在列作训练完成：

```bash
python -m scripts.analysis.fm_transfer.dispatch_screen --source IMMUTABLE_SOURCE --sft-checkpoint OWN_SFT_ENDPOINT --output OWN_TRAIN_SCREEN --shards 24
python -m scripts.analysis.fm_transfer.build_hard_manifest --screen OWN_TRAIN_SCREEN --output OWN_HARD_MANIFEST
python -m scripts.analysis.fm_transfer.qualify_rl_endpoint --sft-checkpoint OWN_SFT_ENDPOINT --manifest OWN_HARD_MANIFEST --output NEW_ENDPOINT_QUALIFICATION
python -m scripts.analysis.fm_transfer.dispatch_rl --sft-checkpoint OWN_SFT_ENDPOINT --manifest OWN_HARD_MANIFEST --qualification-root NEW_ENDPOINT_QUALIFICATION --output OWN_R_SOURCE_RUN
# 合法原生边界，同一源码/同一run恢复：
python -m scripts.analysis.fm_transfer.dispatch_rl --sft-checkpoint OWN_SFT_ENDPOINT --manifest OWN_HARD_MANIFEST --qualification-root NEW_ENDPOINT_QUALIFICATION --output OWN_R_SOURCE_RUN --resume OWN_NATIVE_RL_CHECKPOINT
```
