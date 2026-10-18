# 决策：证据支持唯一尺度候选的有限验证，不构成训练批准

**支持按已封存定义实现 `scene_center_rollout_std_v1`，交独立实现正确性审阅，并在另行批准后执行有界SMOKE/PROFILE。** 完整TRAIN证据已满足原提案§4的资料准入条件，不需要再加幅值/安全gain阈值、改epsilon或寻找第二候选。本报告没有批准profile派发或正式训练，也没有宣称归一化会提分。legacy center-only仍是唯一对照。

此结论绑定提案commit `b1a8cf49b98a718e1b509dc7000c7db3e3c8fb94`，MD SHA `a32d9e405316bdc257ce283343e23b6fb05d3ed54086d42a0e2243dcce82d69c`，JSON SHA `c8294fed768a040b1de43a4153f68e753acd6f5539c7a5a11d63f640129884e9`。提案未修改：原FP32 scene-center A，仅除同old-policy完整optimizer rollout的全DDP G4 std，correction=1、epsilon=1e−10，分子不再全局center，含零A，原BC/policy权重均1。

## 原准入条款的逐项接线

| 原提案§4条款 | 已完成证据 | 判断 |
|---|---|---|
| 完整消费人口与独立核验 | 4096边界、131,064 scene呈现、524,256候选；80,933非零组与训练原记录精确一致；主审计和派生分析各有独立正确性/资源审阅，运行source与RESULT hash匹配 | 满足这一证据条款 |
| 每update std/scale/零组，固定分位与真实N | 4096个update全部有非零组；4095个N=128，1个尾batch N=96，无其他N。std中位数0.0010993601；scale中位数909.6200、范围6.7930–2963.6787 | 满足，不能用该中位数选择一个固定倍数 |
| 每update数值与舍入边界 | 预登记所有numeric flags均0；无std=0、std≤epsilon、常数组非零A或非有限。所有update的 `N*δ²/sum(A²)` 最大5.0675940e−8，远未触发提案原有的≥1 UNRESOLVED条件 | 没有观察到该舍入上界能覆盖优势能量；这不是模型执行精度或所有误差源的证明 |
| 四固定窗口与安全/EP-only尺度分层 | 四窗口均1024个非零update，原C1非零组20,168/20,311/20,184/20,270；安全差异和EP-only均有支持，缩放差异已明确量化 | 满足资料条款，并确认必须保留的风险 |

完整人口为101,592唯一scene、1,176 log；29,472 scene重复两次。每update统计包括所有候选。非零组内共323,732候选，其中8项A恰为0，不能把“非零组内候选”与“逐项A非零”的323,724项混用。没有实际全零update，所以其未来运行路径仍需实现级合成/分布式验收；全零组本身不能因统计或训练筛选而丢弃。

## 主要风险已被证据确认，不应改称无损修复

| 预登记update层 | update数 | std中位数 | scale中位数 |
|---|---:|---:|---:|
| 全部 | 4096 | 0.00109936 | 909.62 |
| 安全事件为空且全部非零组EP-only | 2406 | 0.00093258 | 1072.29 |
| 至少一个D/C/T事件 | 1664 | 0.03677991 | 27.19 |
| 至少一组D/C/T终局指标可区分 | 1119 | 0.07305167 | 13.69 |

最后两行有包含关系，不能当作互斥组相加。EP-only层与安全可区分层的**scale中位数之比为78.33**；这是两个统计量的比，不是配对因果效应、逐项梯度放大比或安全收益比。相同趋势在四窗口均出现：EP-only层scale中位数866.87/904.30/1152.10/1370.13，对应安全可区分层13.46/13.60/13.69/14.01。

因此该适配会改变不同update的相对权重，可能相对增强大量小差值进度批次，削弱稀有大安全差值在全程中的相对重要性。它保留同batch相对gap，不保留全训练绝对误选代价。此风险正是有限对照需要检验的内容；既定提案未把一个任意gain比设置为拒绝阈值，不能看到78.33后临时加cap、换epsilon或改population。本结论只允许把问题推进到有限验证的审阅阶段。

安全区分本身仍稀有：D/C/T分别为791/38/520次scene呈现，覆盖764/37/485唯一scene及395/31/294 log。派生表中“含安全事件的update”的unique scene/log包含同batch其他scene，不能当成安全例数。尤其不能把38个C差异update中的1215个不同batch成员，误报成1215个碰撞区分scene。

79,586/80,933非零组为EP-only（98.34%），但带NC×DAC门控的优势值协方差份额为EP35.80%、TTC48.43%、Comfort15.78%。这些是带符号的代数分解，既不是各单项的独立因果贡献，也不是参数梯度份额。主审计和派生都未计算全TRAIN参数梯度；固定3584的policy/BC=0.0613169%仍只适用于那个EP-only batch。不能用支持数或这一单batch结果宣布全程policy不足、BC冲突或安全梯度被某一确定倍数抵消。

## 进入下一阶段仍缺哪些证据

缺的是**候选真实实现和运行证据**，不是再选一组幅值常数。独立设计数学8项通过及这些描述性结果不能替代以下原定审阅：

1. 实现保持legacy明确身份；全rank非collective本地准备后共同失败投票，再进入统计collectives；同rollout归约在微批前完成；实际N/tail、空mask、零组、MC50、恢复hash和原RNG/BC路径均通过独立验收。
2. 对最终整合SHA、固定配置、数据/精度/拓扑单独批准有界SMOKE/PROFILE，权重弃用，保留原合同≤256更新上限和完整窗口协议。实测分布式归约/错误退出/恢复、成本、policy与BC尺度、有限精度和gradient clipping行为。正常描述性std不能证明这些路径已经可靠。
3. 只在这些证据与资源审阅通过后，Lead才决定是否批准正式同预算legacy对照。共同终点保留完整PDMS及NC/DAC/TTC、EP/Comfort、合法零分/双向退化和停车低进度行为；安全事件分母与log相关性保持可见。不基于DEV选系数、epsilon、epoch或新variant。

old每optimizer boundary刷新、maximum_reuse=1，初始ratio约1，clip0.2通常尚未激活就执行唯一optimizer step。因此profile中的更新前ratio/clip统计不能担保step后偏移小，更不能担保安全。候选不增加reuse/epochs来改变这个合同。

## 来源与本次读取范围

主结果：`full_train_feedback_support_v1/RESULT.json`，327,728B，SHA `0a6a84d3652122aeb69760b5237a56c71fec269c16b4066e78a0ddb05876d8cc`，runtime `dd413aea` / source SHA `30bb3237019a463e68bb5c0686d902521d814ee99bec228a94de7b537f92f4ca`。派生结果：`rollout_scale_train_evidence_v1/RESULT.json`，1,109,160B，SHA `3c65efe7e228cccec1d211eec7e1652c29d729c8292b25e8baa706c1ebfeaff1`，runtime `5751d4d0` / source SHA `026f49fb34f22307439fc6c2672e0323758b8d65710c836ffbbff0b3017ff72e`。所有review绑定见 `REVIEW_BINDINGS.json`。

派生运行已有成本6.7875 CPU秒、15.7496 wall秒、峰值RSS98,402,304B；只读紧凑账本，未生成新reward/score/gradient或optimizer更新。该成本属于Lead已执行的描述性分析，本Agent没有重复执行它。本Agent一次读取上述派生RESULT并仅读其他完成小摘要/源码，4MiB限额内保守预留3,206,312B；大payload、逐ledger紧凑回执、模型/图像/cache、DEV/Navtest读取均0。本次也未运行训练或profile。

`PRIMARY_EVIDENCE.json` 与 `PENDING_PRIMARY_ONLY.md` 保留派生交付前的中间状态；最终结论以本文件和 `DECISION.json` 为准。原提案未改动。本决策是研究证据建议，不是批准文件。
