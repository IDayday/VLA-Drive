# G3 90k完整Navtest与收尾判断

2026-10-12完成锁定NAVSIM v1 canonical Navtest：12,146场景、136 logs，q0与q_final均为0个评分失败。模型/推理源码仍为`0c54f1c49e7ed2a76b22826b654d6c72625f5b6d`，FP32 master严格恢复，FP32推理、TF32关闭、原10步FM、单候选、场景绑定seed42。来源是原生`latest_0090000`，没有改变标签、模型或评分器。

| 轨迹 | PDMS | 零分场景 |
|---|---:|---:|
| q0 | 88.67901950 | 599 |
| q_final（主结果） | **88.88441673** | 573 |

q_final分项：NC98.10637247、DAC97.06076074、TTC94.77194138、EP83.09272870、comfort100、DDC98.38630002。

| 比较：90k final减基线 | PDMS差 | 95% log-bootstrap区间 | 修复/新增零分 |
|---|---:|---|---:|
| 同模型90k q0 | +0.205397 | [0.086142, 0.328453] | 41 / 15 |
| 80k final | −0.086167 | [−0.230270, 0.068283] | 38 / 43 |
| 70k final | +0.014291 | [−0.178127, 0.204877] | 75 / 67 |
| 50k final | +0.364869 | [0.005695, 0.755822] | 165 / 113 |

相对80k的点估计下降，区间跨零，不能说已经证明退化；相对70k也几乎没有增益。G3后期曲线支持“不追加原定100k以外预算”的决定。当前最高观测G3依然是80k的88.97058401。其FP32推理权重继续保留，90k不额外保留推理副本。内部修正的收益从70k的0.631842、80k的0.355954降到90k的0.205397；修正仍有正收益，但这不是独立原DDP对照，也不能由此断定某一个辅助loss的因果效果。

全部比较按136个logs聚类bootstrap20,000次，保持真实场景权重，仅一个训练seed。区间没有为反复Navtest观察校正；这些中间测试是用户明确要求，不再是未触碰的最终留出集。完整配对与失败转移见[G3_90K_COMPARISONS.json](G3_90K_COMPARISONS.json)，全组曲线见[SVG](G_NAVTEST_CURVES.svg)和[CSV](LEARNING_CURVES.csv)。

执行与恢复：初始16个GPU进程分布在临时授权的vlawm-zt和vla-zt3，各8张不同物理A800。两主机四场景资格验证与原生FP32路径的q0/q_final/out_of_range逐位一致。vla-zt3在15:22断连，日志明确记录remote-host transport closure；根因未知，不能推断是谁终止。SSH修复后，原8个PID/start-time实例已不存在，物理GPU UUID一致。另8个vlawm-zt评估进程及本地控制器于约15:30被另一调度任务按其记录的用户指令暂停，优先完成IQE 60k。外部守护程序于15:46:31/32恢复原进程。暂停时已有7,569份有效预测及评分，未重跑这些已完成场景。本任务没有向正式训练或压力脚本发信号。

15:59:37，全部12,146份预测和两套评分记录已经存在；原controller按其设计记录8个成功launcher与8个SSH255，保留为unclassified失败并退出。CPU恢复工具`dca070e`只接受本次失败hash、attempt、来源hash、确切exit-code顺序和原始日志。它再次证明所有原worker/launcher/scorer退出，核验每个token/log、预测payload、canonical proposal、metric-cache hash及评分有效性，再调用同一canonical scorer完成汇总。**没有把8份中断report改成COMPLETE，也没有将255伪改成0，没有新增GPU推理或修补失败分数。** 原失败JSON及原始进程报告保留。5个CPU负向控制测试通过；[执行命令](G3_90K_TRANSPORT_CLOSURE_COMMAND.json)、[恢复核验](G3_90K_RECORD_VALIDATION.json)、[退出证明](G3_90K_WORKERS_EXITED.json)可追溯。

模型进程峰值allocated12.548GiB、reserved13.078GiB。16个进程的已报告scene-loop process-wall合计**6.73118小时下界**，其中8个断连进程只有0.86779小时的部分计时；8个正常退出进程的驻留时间6.06870小时，含0.20530小时加载及外部暂停等待。它不是独占GPU-hours、纯kernel时间或完整总成本。canonical恢复汇总额外311.19秒CPU墙钟；完整恢复还包含前置数据复核。详细成本与限制在[COMPLETE_RESULTS.json](COMPLETE_RESULTS.json)，不以未知时段为零或把未完成报告当完整成本。

本任务的临时资源预留已在退出核验后释放；压力脚本由用户外部机制处理，没有由本任务启动或恢复。原注册的50k以后每10k Navtest调度已使用同一冻结dispatcher恢复，不新增方法实验。九小时授权仍严格截止23:25:43 UTC，不因恢复重新计时。

G1 60k目前最高观测权重已用原生CPU FP32 master加载器保存为一个12.164GB推理文件，耗时218.60秒、无GPU任务，完整17份临时native硬链接已释放。该资产不含optimizer，不能冒充训练恢复点；正式训练从自己的完整latest继续。G3保留原80k最佳推理文件；旧结果、负结果和失败记录均不删除。
