# 独立证据审阅：道路失败32人口

**结论：两例真实、局部的canonical DAC修复成立；只接受局部证据，不批准进入代理训练、
DFQL或新的方法阶段。** 固定32场景中仅2个方向有定义，负向修复2/2；已观察修复的请求
人口下限为2/32=6.25%。其余30个为未定义，不能改写为30次修复失败，也不能从2/2推断
全人口100%修复。主C1 scale正式4096训练继续，当前诊断不构成改变主线的依据。

## 实际来源与核验

执行源码 `864619b0130fb3b220b577dd34b993ffeac97e27`；注册文件SHA
`c9c76ffd513fcf3cf59e1753358bd552b0635ae6394399e509d15c09dd9cda0a`；原FRAME文件SHA
`5ea6524267bc4e02fde7bfd79389ba774bea3f77821898b2471b369f316b1841`保持不变。
实际绑定selection文件SHA为
`02755da1e6563a6d3eea10ffd8258d0eff82e8a28d86509aca751d45fe143d5c`，完整selected内容hash为
`23cca9e7af152ebc3dede70e9d7d06e79ae68b8d60ae78e923c96b3bb85fc1a2`。

本审阅读取实际`BOUND_SELECTION`及binding receipt、注册/admission/独立reviews、32份
`STARTED`/`SCENE_RESULT`/`NUMERICAL_PARITY`、2份`DIRECTIONS`、8份完整variant结果
（2份复用original＋6份新评分）和6份调用意图。逐项对齐原FRAME顺序、boundary/rank/slot、
原失败及安全兄弟身份、原label、context与reference；重新计算保存JSON中物理轨迹、执行
states和reference arrays的checksum，以及最终selected内容hash。没有重读原`.pt`、
metric cache、图像或模型，也没有执行proxy/scorer/有限差分。全32原记录的payload验证
来自已固定的实际binder/runner链；本审阅独立核对逐行绑定和回执，不冒称重新验证那批大payload。

实际32条绑定均为原DAC=0、同context的安全兄弟DAC=1、native_t0=false；原scene顺序无
替换。2条复用original的实际41点native flags也再次与DAC核对一致。全部32几何parity
通过原1e-9 gate，最大XY/corner误差为0，最大周期yaw误差8.88e-16rad。

8个registered源码文件hash均匹配；`road_margin_proxy.py` SHA
`baa33371d2b000ccefa9d507289001ee67f7ee123c90ee32f6bdba105cb59a7a`与原0d0d19完全相同。
两例保存的扰动严格满足完整[8,4] raw空间正/负梯度方向及L2=0.005；随机seed也与原
scene_id＋最终selection hash算法一致。0.005不是米，未冻结yaw。两例保存的两档有限
差分均通过、离散regime稳定；这里核验的是原执行产生的记录，没有重新求梯度。

## 精确分母

| 项目 | 结果 |
|---|---:|
| 固定请求/原DAC失败/安全兄弟已绑定/t0=false | 32/32/32/32 |
| COMPLETE，共同四variant有效 | 2 |
| UNDEFINED_DIRECTION | 30 |
| 实际新评分 | 6，均有独立持久化intent及匹配结果 |
| 负方向DAC修复 | 2/2；请求人口已观察下限2/32 |
| 正方向DAC修复 | 0/2；请求人口已观察下限0/32 |
| 固定随机方向DAC修复 | 1/2；请求人口已观察下限1/32 |

2/32是已观察成功下限，既不是未定义样本被判失败后的成功率，也不是TRAIN总体置信界。
32来自预先固定的D-discriminative历史TRAIN组，只有2个不同log/不同旧policy boundary
进入有效配对；这是由代理定义域筛出的条件子集。没有新cohort、新随机方向或模型比较。
`population_summary_complete=false`与`PARTIAL_OR_UNDEFINED`准确保留，worker正常退出
不等于科学全人口完成或方法PASS。

## 两例是多小的边界跨越

以下距离来自执行时已保存的自定义road-union签名余量；官方DAC恢复另外由原生41点
事件flag确认，两者分开核验。负方向消除了所有原生D flag，确为canonical修复，
并非只把自定义cost下降当作修复。

| scene（boundary） | 原命令最小余量 mm | 原执行最小余量 mm | 原生出界采样时刻 s | 负向执行最小余量 mm | 负向最大命令角点位移 mm |
|---|---:|---:|---|---:|---:|
| a8757b9d85935d3f（3936） | +7.466 | −0.099083 | 1.9、2.0 | +9.734164 | 8.228344 |
| 2b5ac45dfc6f5273（2001） | +117.460 | −1.577845 | 1.0 | +4.600659 | 12.124513 |

两例原命令本身的四角余量均为正，原真实执行后才出现很浅的出界。这正体现了
command-space代理与tracker实际执行的区别；本次没有tracker的可微梯度。负向建议
恰在这两例改善了实际执行余量，但不能由此保证深度出界、长时段出界或其他几何区域
可修复。原生时间点是离散采样结果，不推断连续时间的精确违规持续时长。

负方向执行后自定义cost相对正方向和随机方向均在2/2更低；平均差分别为
−0.004343046m、−0.001710036m。这里的m是自定义余量代价单位，不是行驶距离增量。

| scene | 原PDMS | 负向PDMS | 正向PDMS | 随机PDMS |
|---|---:|---:|---:|---:|
| a8757b9d85935d3f | 0 | 0.9107816869 | 0 | 0 |
| 2b5ac45dfc6f5273 | 0 | 0.9221386946 | 0 | 0.9221559078 |

因此不能说负向PDMS全面优于随机：第二例随机也修复，PDMS还略高约0.001721个百分点。
两例负向均值0.9164601908只是**共同有效两例**的均值，不能报告为整体91.65个百分点
规划收益，也不能与原固定DAC1人口直接合并计算研究胜率。

## EP跃升主要是合规门解除

绑定的原生 `pdm_scorer.py::_aggregate_scores`（SHA
`02ab075017d4edb1c8bba6476ef1674b84d43dbca0f599df87cfe052b2dd5211`）在归一化EP之前先以
NC×DAC乘原始进度；最终PDMS也乘该合规项。两例reference输入、reference执行states
及EP分母在所有variant间完全相同。实际保存记录为：

| scene | 原raw progress m | 负向raw progress m | 实际进度增量 m | 官方EP原→负 |
|---|---:|---:|---:|---|
| a875… | 15.065804636 | 15.066063204 | +0.000258568 | 0 → 0.785876049 |
| 2b5… | 28.782570903 | 28.785575146 | +0.003004243 | 0 → 0.813132867 |

原车并未停住；原DAC0把effective progress归零。约0.26mm/3.00mm的实际raw进度变化
不能解释为显著行驶效率提升，EP大跳变主要是解除二元合规门。官方分数合法且可复现的
保存身份得到核验；这种敏感性限制的是研究外推，而不是否定两例真实修复。

## 安全读数与30个未定义

有效两例原NC/TTC/comfort均为1，全部四variant仍为1，方向分项也均为1；未发现这两例
新增碰撞、TTC或舒适分项失败。原始进度无下降，负向执行最大速度/加速度和tracking读数
只有小变化。但安全风险的有效分母分别仅2/2/2，不是完整32。

完整请求人口原NC合规分母31、TTC30、comfort32；其中未定义人口分别留下29、28、30个
原合规场景没有干预结果。原DAC0使所有原PDMS均为0，所以“没有高分变零”或“PDMS未
下降”不能作为本人口安全保证。安全兄弟用于证明原context存在一条合规命令，没有被
改作训练target或新增对照评分。

30个逐scene错误均为`MOVING_CORNER_BOUNDARY_OR_NEAREST_TIE`；没有这30个的DIRECTIONS
或variant评分文件。旧0d源码在moving corner落边界**或**最近segment距离tie时保守拒绝，
这些样本未通过完整有限差分/扰动验证。当前回执不区分boundary与tie，也没有保存足够的
nearest-segment regime来定位两者。因此不能断言30个都存在数学上不可导的真实道路点，
不能说是零梯度，不能说是数值parity失败，更不能说已经证明30例修复失败。

当前定义域覆盖率只有2/32=6.25%，这是真实的使用限制；本审阅没有事后设一个覆盖率
阈值来宣告统计FAIL。现有证据尚未建立足够的覆盖或稳定性，无法支撑把这个代理用于
新训练更新。该限制不能靠放宽tie、公差、扩大人口或反复试随机方向消除。

## 决策与资源

本轮按原固定人口结束。接受两例局部canonical道路修复，保留30个未定义及全部分母；
剩余90次调用额度不是另选样本或重试未定义方向的理由。**不批准下一阶段代理/DFQL/
C5/G8训练，不扩大cohort，不变步长、不放宽tie/tolerance。** 当前没有必要为这份结果再
评分或重算几何。若之后另行提出非平凡代理/更新方法，须先完成原论文全文、官方源码
和适配边界的独立证据工作，再单独审阅；本报告不提出未核实的新方法。

实际诊断supervisor wall19.394s，worker wall14.958s/CPU10.599s，逻辑资产读139,626,807B，
聚合峰值RSS1,712,340,992B；6次新评分，worker exit0且owned退出确认，无monitor/cleanup
错误。它们是执行回执的成本，不是本次证据审阅的成本。

本次最终stdlib审阅读取3,156,685B，CPU0.114745s，wall0.215131s，`/proc`实际VmHWM
19,976,192B。另预留8MiB覆盖先前小文件检查、一次收紧审阅断言后的重跑及最终封存读取，
内容读取保守上界11,545,293B，小于32MiB。没有模型/cache新读、proxy调用、评分或GPU。
脚本的13,400项断言多为保存数组的数值/shape/hash一致性，不是13,400个科学样本。

可重读的最小证据为`AUDIT_RESULT.json`（逐例标量、全32状态、精确分母）、
`READ_RECEIPT.json`（实际来源path/size/SHA与审阅资源）、`audit_saved_results.py`。
原FRAME、原回执、原代理和主训练代码保持不变；本报告仅签Evidence解释，不签实现或系统PASS。
