# B组50k SFT：继续改善，尚无B组RL结论

B_SOURCE_ACTION_DIT的50k固定观察点完成完整DEV1696场景/16 logs与TRAIN诊断512场景/343 logs。冻结Qwen、独立SimWAM ActionDiT、原8×4动作，没有Wan/MoT。所有结果为原始观测、seed2101、单候选10步ODE及canonical NAVSIM v1.1。

**DEV原生PDMS为93.465822，shift5为93.602993；相对25k分别+0.598729/+0.441426个百分点，但两项log聚类配对区间仍跨零。** 零分场景继续减少，仍有新增安全违规。本组仍处于SFT，按原100k终点继续；后续RL必须相对自己的100k SFT衡量，不能将这些增量算作Flow-GRPO收益。

## 配对学习过程

| 人口/网格 | 10k | 25k | 50k | 50k−25k，百分点 | 95%区间，百分点 |
|---|---:|---:|---:|---:|---|
| DEV 原生 | 91.689072 | 92.867093 | **93.465822** | +0.598729 | [−0.431393,+1.396941] |
| DEV shift5 | 91.687880 | 93.161568 | **93.602993** | +0.441426 | [−0.473347,+1.024024] |
| TRAIN诊断 原生 | 74.932637 | 77.387590 | 82.171598 | +4.784007 | [+2.474333,+7.169827] |
| TRAIN诊断 shift5 | 76.261385 | 79.627974 | 82.077723 | +2.449748 | [+0.089624,+4.859884] |

区间按完整log重采样并保持场景权重，不包含训练或推理seed不确定性。TRAIN诊断含预选困难场景，不是完整训练人口无偏均分。TRAIN增量比DEV大，但两者人口不同，不能仅由该差距认定过拟合。

原F-SFT的DEV原生为93.619237；B50k仍低约0.153415个百分点。B的动作头从随机初始化训练，A从已有预训练动作头继续，两条路线的知识、参数量与监督预算不同；现有结果不能隔离架构因果效应。

同一50k权重的原生−shift5为−0.137171个百分点，区间[−0.665358,+0.259193]。两种读出均保留，不以shift5均分较高替代登记的原生主协议。

## 安全与低尾

| 50k人口/网格 | NC | DAC | TTC | EP | Comfort | DDC | 零分 | 联合安全 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DEV 原生 | 100.000000 | 98.997642 | 99.646226 | 86.169388 | 99.764151 | 98.791274 | 17 | 1673/1696 |
| DEV shift5 | 100.000000 | 99.056604 | 99.705189 | 86.357089 | 99.764151 | 98.879717 | 16 | 1675/1696 |
| TRAIN诊断 原生 | 100.000000 | 90.625000 | 95.117188 | 74.907147 | 99.804688 | 99.023438 | 48 | 441/512 |
| TRAIN诊断 shift5 | 99.804688 | 90.625000 | 95.117188 | 74.759972 | 99.804688 | 99.121094 | 49 | 441/512 |

分项用0–100表示；联合安全要求NC/DAC/TTC均完全通过，均分不等于事故概率。下表相对B25k同网格，reference-safe集合固定为25k人口，只用于这一配对，不冒充最终100k参考。

| 人口/网格 | NC新增/恢复 | DAC新增/恢复 | TTC新增/恢复 | 非零→零/零→非零 | 25k安全人口中新联合违规 |
|---|---|---|---|---|---|
| DEV 原生 | 0/0 | 6/15 | 3/10 | 6/15 | 9/1659 |
| DEV shift5 | 0/0 | 8/14 | 2/10 | 8/14 | 10/1663 |
| TRAIN诊断 原生 | 0/0 | 6/34 | 6/1 | 6/34 | 11/417 |
| TRAIN诊断 shift5 | 1/0 | 14/29 | 4/3 | 15/29 | 16/427 |

DEV零分从25k的26/22降到17/16，TTC有净恢复；原安全人口仍新增9/10个联合违规。TRAIN诊断原生TTC从96.093750降到95.117188，shift5出现1个新增NC违规。不能把DAC改善或零分下降解释成全部安全分项改善。

DEV原生G+=19.260229、G−=9.105782，shift5为18.073824/10.587243，单位为原始0–1分差的场景求和。原生任意改善/退化为483/607，达到±0.005幅度为120/143；shift5分别428/662和141/119。任意负差不是事故计数，均分上升也不表示多数场景上升。

1%分位从25k的零提高到47.281889/54.531406；5%分位为83.333333/83.516331。低尾改善与仍存在的17/16个零分场景同时成立。

## 轨迹、成本与身份

DEV请求点平均段速度原生4.5268、shift5 4.5343 m/s；请求点离散jerk RMS从25k的0.6718/0.8876降到0.5206/0.6471 m/s³。模拟终点前进量均值17.6151/17.6431米。请求点jerk是描述统计，不是官方Comfort或模拟车辆瞬时jerk，不能单凭该量判断规划安全。

50k原生SFT checkpoint的阶段为READY、world8、cursor1,600,000，表示50k更新对应160万次场景曝光，不是160万个独立场景。完整100k终点尚未达到，不能将50k导出提前用作正式RL起点。

原入口导出权重SHA256：`65a9066cfc6cf7078d05bca7178d035dbd7251353d17c98b0a79615f1687bb31`。训练源码仍为`4e0ce1f45b6778895648689228a5628432bd2616`，数值评价源码仍为`45d2318d2dec265583860a26f891d8b79bb6068a`。冻结前缀保持，恢复使用原生模型/Adam/RNG checkpoint。

| 评价人口 | 两网格逻辑候选 | 实际推理含入口检查 | job wall秒 | 累计worker wall秒 |
|---|---:|---:|---:|---:|
| DEV | 3392 | 3400 | 434.334 | 3408.625 |
| TRAIN诊断 | 1024 | 1032 | 289.238 | 2288.039 |

两项分别在tsinghua-zt/vla-zt并行。16个worker已逐PID核实退出，16项原入口检查通过，完整场景分母和成本账本通过既有汇总器核验。配对与网格分析仅读取已保存轨迹/score，没有新增模型推理或评分。共享worker累计时间不是独占GPU计算时间，两个job wall也不能相加冒充端到端耗时。

当前两路线固定评价累计完成22/36项。A继续固定20k RL，B继续100k SFT，再通过自己的全TRAIN筛选和真实八卡RL验收进入20k RL；未新增seed、算法或预算。新Navtest尚未开始，DEV达到93.6不代表最终RL超过92的目标已经完成。

## 证据和实际命令

[DEV相对25k](B_SFT_update050000_DEV_vs25k.json)、[TRAIN诊断相对25k](B_SFT_update050000_TRAIN_DIAGNOSTIC_vs25k.json)、[同权重网格差](B_SFT50k_DEV_native_minus_shift5.json)、[执行、退出和成本](B_SFT50k_evaluation_execution.json)。

```bash
python -m scripts.analysis.fm_transfer.analyze_evaluation --evaluation /mnt/project/fm-transfer-20261016/evaluations/B_SOURCE_ACTION_DIT_SFT_update050000_DEV --reference /mnt/project/fm-transfer-20261016/evaluations/B_SOURCE_ACTION_DIT_SFT_update025000_DEV --output /mnt/project/fm-transfer-20261016/analysis/B_SOURCE_ACTION_DIT_SFT_update050000_DEV_paired_25k.json
python -m scripts.analysis.fm_transfer.analyze_evaluation --evaluation /mnt/project/fm-transfer-20261016/evaluations/B_SOURCE_ACTION_DIT_SFT_update050000_TRAIN_DIAGNOSTIC --reference /mnt/project/fm-transfer-20261016/evaluations/B_SOURCE_ACTION_DIT_SFT_update025000_TRAIN_DIAGNOSTIC --output /mnt/project/fm-transfer-20261016/analysis/B_SOURCE_ACTION_DIT_SFT_update050000_TRAIN_DIAGNOSTIC_paired_25k.json
python -m scripts.analysis.fm_transfer.analyze_grid_gap --evaluation /mnt/project/fm-transfer-20261016/evaluations/B_SOURCE_ACTION_DIT_SFT_update050000_DEV --output /mnt/project/fm-transfer-20261016/analysis/B_SFT50k_DEV_native_minus_shift5.json
```

命令已实际执行；既有输出不覆盖。未上传权重、缓存、原始数据或私有场景图片。
