# B组25k SFT：完整DEV均分上升，仍需检验后续RL

B_SOURCE_ACTION_DIT完成登记的25k中期检查点评价：完整DEV1696场景/16 logs、固定TRAIN诊断512场景/343 logs。本文均为当前原始观测、seed2101、单候选、canonical NAVSIM v1.1；native和shift5分别报告，无候选筛选。B继续固定100k SFT，尚未进入正式RL。

B使用冻结Qwen与独立SimWAM ActionDiT，原8×4动作；没有Wan/MoT权重。25k原入口导出SHA256为`250721a234e0eedafcfbea849747ebf92d3e39ddd5de161332019835b624839e`。

## 相对同一B路线10k的变化

| 人口/网格 | 10k PDMS | 25k PDMS | 差值，百分点 | 按log聚类95%区间，百分点 |
|---|---:|---:|---:|---|
| DEV native | 91.689072 | 92.867093 | +1.178021 | [-0.860928, +2.322581] |
| DEV shift5 | 91.687880 | 93.161568 | +1.473688 | [-0.674732, +2.664100] |
| TRAIN诊断 native | 74.932637 | 77.387590 | +2.454954 | [-0.613575, +5.674473] |
| TRAIN诊断 shift5 | 76.261385 | 79.627974 | +3.366590 | [+0.280617, +6.790572] |

DEV均分观察到上升，但两个区间均跨零；TRAIN诊断只有shift5的该区间未跨零。区间保留场景权重、按完整log重采样，未包含训练seed或推理seed不确定性。TRAIN切片有预选困难场景，不能当成完整TRAIN无偏估计，也不能由两种人口的均分差直接判定泛化。

原F-SFT的DEV native为93.619237，B25k仍低0.752144个百分点。两者初始化知识和监督预算不同，不能据此隔离架构优劣。B从10k到25k的进展属于额外SFT，不是Flow-GRPO增量。后续RL仍必须相对自己的100k SFT终点衡量。

## 安全、零分与退化

| 25k人口/网格 | NC | DAC | TTC | EP | Comfort | DDC | 零分 | NC/DAC/TTC联合通过 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DEV native | 100.000000 | 98.466981 | 99.233491 | 85.770174 | 99.764151 | 98.850236 | 26 | 1659/1696 |
| DEV shift5 | 100.000000 | 98.702830 | 99.233491 | 86.123139 | 99.823113 | 98.850236 | 22 | 1663/1696 |
| TRAIN诊断 native | 100.000000 | 85.156250 | 96.093750 | 70.378655 | 99.414063 | 98.828125 | 76 | 417/512 |
| TRAIN诊断 shift5 | 100.000000 | 87.695313 | 95.312500 | 72.786826 | 99.609375 | 98.828125 | 63 | 427/512 |

分项显示为0–100。完全通过仍要求每个场景对应原指标等于1；分项均值不是事故概率。下表全部相对B10k同网格，未重划该比较的reference-safe集合。

| 人口/网格 | NC新增/恢复 | DAC新增/恢复 | TTC新增/恢复 | 非零→零/零→非零 | 10k联合安全人口中新违规 |
|---|---|---|---|---|---|
| DEV native | 0/0 | 9/28 | 6/11 | 9/28 | 15/1635 |
| DEV shift5 | 0/0 | 9/32 | 6/6 | 9/32 | 15/1640 |
| TRAIN诊断 native | 0/5 | 24/32 | 4/15 | 24/34 | 28/403 |
| TRAIN诊断 shift5 | 0/5 | 28/42 | 5/13 | 28/43 | 31/410 |

DEV两个网格的零分都从45下降，DAC有净恢复；native的TTC有净恢复，shift5的TTC失败数量持平，但失败场景并不完全相同。两个网格均新增15个原联合安全场景的违规。native Comfort还由99.823113降至99.764151，增加1个不通过场景。因此不能将零分减少或均分上升写成安全全面改善。

DEV native的G+=33.809686、G−=13.830456；shift5分别为37.293467、12.299725。它们是逐场景原始0–1分差的和，除1696再乘100才是均分百分点变化。native有501个任意正差、600个任意负差；达到0.005原始score幅度的改善/退化为157/250个。shift5分别为686/409和288/153。mean上升不表示多数场景都上升，任意负差也不等于事故。

DEV native/shift5的5%分位由82.210444/82.028204升至83.357904/83.356953；1%分位仍为零。低尾改善和剩余失败同时存在。

## 网格和请求轨迹

同一25k权重、场景和root下，native减shift5为−0.294475个百分点，95%区间[-0.577006,+0.112136]，仍不能排除零差。DAC有4个场景仅native通过、8个仅shift5通过；TTC各有2个仅一侧通过。不能选择shift5较高的均分作为唯一部署结果。

DEV请求点的平均分段速度：native从4.5441降至4.5244 m/s，shift5从4.5353降至4.5317 m/s；mean EP同时上升。请求轨迹的离散差分jerk RMS由1.1102/1.4285降至0.6718/0.8876 m/s³。这些是请求轨迹描述统计，不能替代官方Comfort或证明具体机制；请求与模拟结果均分别保留。

## 成本和执行

| 人口 | 两网格逻辑单候选数量 | 实际推理含原入口检查 | job wall秒 | 累计worker wall秒 |
|---|---:|---:|---:|---:|
| DEV | 3392 | 3400 | 544.133 | 4308.000 |
| TRAIN诊断 | 1024 | 1032 | 279.980 | 2228.042 |

两任务在tsinghua-zt/rl-zt2并行，共16个worker；每个worker实际验证原入口与native输出一致。全部场景及payload hash通过既有analyzer检查，全部worker已退出。没有缺失场景被移出分母，没有新推理seed。累计worker wall不是独占GPU计算时间，也不能与job wall直接相加。配对10k和网格差分析只读取已有结果，新增模型推理和评分均为0。

并行调度去除了DEV完成后才启动TRAIN诊断的依赖；不同检查点的缓存、权重和系统状态不同，本次不将其与10k wall差异当作严格速度因果对照。

## 当前决定

继续原100k SFT及50k/75k/100k观察；不以25k已超过DEV92为由提前改RL起点，也不新增reward、sigma、anchor或seed矩阵。A自己的RL仍正常进行，首个5k部署结果另报。

B25k尚无Navtest，也没有正式RL更新，最终“RL/Navtest超过92、IL→RL有明显增益”的目标仍未验证。

机器可读证据：[DEV相对10k](B_SFT_update025000_DEV_vs10k.json)、[TRAIN诊断相对10k](B_SFT_update025000_TRAIN_DIAGNOSTIC_vs10k.json)、[同权重网格差](B_SFT25k_DEV_native_minus_shift5.json)、[执行与成本核查](B_SFT25k_evaluation_execution.json)。

实际新增分析命令：

```bash
python -m scripts.analysis.fm_transfer.analyze_evaluation --evaluation /mnt/project/fm-transfer-20261016/evaluations/B_SOURCE_ACTION_DIT_SFT_update025000_DEV --reference /mnt/project/fm-transfer-20261016/evaluations/B_SOURCE_ACTION_DIT_SFT_update010000_DEV --output /mnt/project/fm-transfer-20261016/analysis/B_SOURCE_ACTION_DIT_SFT_update025000_DEV_paired_10k.json
python -m scripts.analysis.fm_transfer.analyze_evaluation --evaluation /mnt/project/fm-transfer-20261016/evaluations/B_SOURCE_ACTION_DIT_SFT_update025000_TRAIN_DIAGNOSTIC --reference /mnt/project/fm-transfer-20261016/evaluations/B_SOURCE_ACTION_DIT_SFT_update010000_TRAIN_DIAGNOSTIC --output /mnt/project/fm-transfer-20261016/analysis/B_SOURCE_ACTION_DIT_SFT_update025000_TRAIN_DIAGNOSTIC_paired_10k.json
python -m scripts.analysis.fm_transfer.analyze_grid_gap --evaluation /mnt/project/fm-transfer-20261016/evaluations/B_SOURCE_ACTION_DIT_SFT_update025000_DEV --output /mnt/project/fm-transfer-20261016/analysis/B_SFT25k_DEV_native_minus_shift5.json
```

这些输出已经存在，入口有禁止覆盖检查；需要再次分析时使用新的输出文件名，保留旧摘要。
