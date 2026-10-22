# S0可解码空间／未来表征：2026-10-22进展

本页增量补充[前次记录](../20261018/CONTINUATION_V2_ZH.md)，保留此前配方、失败与负结果。新配方RS与Rfrozen已分别于2026-10-19完成共同1000更新。**训练结束后未自动接上终点评估；本次核对发现并补齐这一执行缺口。没有新规划收益或Navtest结果。**

## 实际完成了什么

两组仍从同一个原始GT S0 100k开始，保持当前三前视、合法ego、DINO与原DiT。共同使用1024空间queries、32×32 patch读出、同一训练集PCA准备权重与训练内梯度校准。RS更新Qwen后8层attention的rank32 LoRA；Rfrozen冻结Qwen。它们是固定train64上的学习性实验，不是全量表征预训练：1000次单场景更新，平均15.625次曝光。DiT没有训练，world reader仍关闭。

| 教师预测一致性指标 | 后8层LoRA RS | 冻结Qwen Rfrozen |
|---|---:|---:|
| 同一train64占用IoU | 60.745% | 61.406% |
| 固定独立dev16占用IoU | 44.296% | 42.950% |
| 教师占用漏检率 | 45.910% | 46.940% |
| 前向x<15m IoU | 40.690% | 39.647% |
| 教师低置信边界代理区域IoU | 28.656% | 26.867% |
| 训练教师均值控制的dev16 IoU | 37.751% | 37.751% |
| 场景置换后dev16 IoU | 36.846% | 37.796% |

RS−Rfrozen为**+1.346个百分点**；10000次按log配对bootstrap、汇总原始TP/FP/FN后计算IoU，95%区间为**[-0.027,+2.938]个百分点**。这是16个独立开发log、每log一个预先固定场景、一个新增头训练seed42上的探索性证据；不是1696场景完整开发评价，也不能代表跨训练seed稳定性。原S0底座仍只有原来的训练seed。

上轮RS/Rfrozen dev16为42.328%/41.836%；新版本方向上改善，但两轮同时改变了query/读出、初始化与损失权重，不能把跨轮差值单独归因于某一个修复。新版本内部的RS/Rfrozen对照共用配方。

所有数值见[RESULTS.json](RESULTS.json)，逐场景计数见[RS](RS_DEV16_DECODING.json)、[Rfrozen](Rfrozen_DEV16_DECODING.json)，配对区间见[原始bootstrap收据](DEV16_PAIRED_BOOTSTRAP.json)。

## 目前的结论

1. **原压缩结果不能概括容量上限。** 同容量PCA压缩的留出训练日志IoU已达99.633%。学生独立开发解码仍约44%，主要缺口已经落在图像表征向目标空间的迁移与学习上；仅凭压缩数字不能证明学生学会。
   两学生在训练场景本身也只有约61%，尚未达到充分tiny-overfit。训练拟合不足与训练到开发的下降同时存在，不能仅凭开发下降将问题全部归为过拟合，也不能直接断言当前视觉不可学习。
2. **学生读到了部分场景信息，但关键细节仍弱。** 两组都超过训练教师均值；置换场景后下降。固定开发场景可视化显示粗布局相近、孤立占用和细边界难以恢复，并有patch纹理。边界代理区域低IoU和高教师占用漏检率支持这一局限，不能将整体一致性提升直接解释为驾驶约束已可靠。
3. **LoRA有方向性增量，尚未建立稳定内化结论。** 组间区间跨零，评价样本小且只有一个新增头seed。RS无reader时在固定dev8的同噪声轨迹平均位移约0.012米，Rfrozen为0；DINO MSE约0.016962/0.016836。LoRA仍可能改变原动作路径，零初始化reader不构成不遗忘保证。
4. **规划价值还没有被检验。** 本次没有训练P3 reader/DiT，没有新PDMS、安全分项、零分转移或Navtest。此前8场景两步reader工程评分只证明链路可运行，不能充当这轮方法效果。
5. **时间路线仍未验收。** SGDrive默认数据协议约1秒已有时间戳证据，但发布checkpoint的环境覆盖未记录；原标签含GT未来轨迹走廊、缺点置空及坐标口径问题。没有T-driving适配、有效动态区域优于persistence的结果或时间学生训练。

这里的IoU始终相对原生ViGT预测；没有可靠observed-free/unknown，不能称独立GT占用准确率。所谓false_free字段实际为FN/(教师占用数)。区域unknown计数表示该区域之外的查询，不是被验证的未知空间。教师低置信边界是abs(logit)<2的代理，不是精确道路边界。

## 下一步的有限顺序

先对齐train64与dev16，区分训练未拟合、图像到空间的定位问题和泛化问题；结合实际梯度及optimizer更新证据，只做有具体假设的共同修复。固定16个开发场景已生成教师／冻结学生／LoRA学生／误差面板，索引见[私有可视化索引](PRIVATE_VISUALIZATION_INDEX.json)，RGB与面板留在受控共享存储。

随后补可靠的occupied/free/unknown与坐标证据。当前合并点云缺五个物理LiDAR发射原点／首回波来源；不能使用identity外参或缺点补零伪造free。空间路线可以独立推进；时间路线先完成单一明确horizon的教师适配与persistence对照。

表征验收后再做匹配预算R0/RS及原DiT reader实验，同时评价无reader模型、修复与新增失败。新patch版本的独立去教师导出尚未验证；旧版本导出PASS不自动继承。完整人口P2/P3、RT/RST、容量/解码控制与3训练seed仍未完成，不自动恢复已暂停的G、nuScenes或其他研究。

## 身份、成本与复现

- 分支：`research/decodable-spatiotemporal-internalization-20261018`。
- 原S0训练源码：`1493deda247107efe076b9294863a6753391298a`；当前1000步学生训练源码：`cd1cd48915df664710f8c9f7976a8a9aceebceee`。
- 本次解码源码：`e0a6561e0708ec4f53a974e2b3ffd4c10406dfc9`；配对分析／可视化源码：`4177f91335a6cb899cb0f5aa93b87368fe84744f`。报告提交不替代执行源码身份。
- 两组tiny连同各自2步预检查和恢复初始化，收据合计1.803 GPU任务小时。窗口计时中位数约RS 2.22秒／更新、Rfrozen1.70秒／更新，属于当时记录的训练窗口，不能替代本次共享GPU上的隔离性能对照。总体预算仍为16 GPU-hours。
- 本次只补评估和分析，没有新增训练、停止无关进程或使用新撤销的tsinghua-zt6。

以下命令已实际执行；`python`对应既有ddp环境（Torch2.5.1+cu124、Transformers4.57.0），不升级共享环境：

```bash
task_root=/mnt/project/decodable-st-20261018
cd "$task_root/execution/endpoint_v2_e0a"
python -m tools.decodable_st.evaluate_decoding \
  --run-root "$task_root/continuation_v2/runs/RS_patch_v2" \
  --output "$task_root/continuation_v2/evaluations/RS_dev16_endpoint1000" \
  --required-update 1000 --evaluation-config "$task_root/audits/eval_dev16_job.json"
```

Rfrozen使用相应同名run与output；train64命令省去evaluation-config，保留required-update1000。当前分支的配对命令为：

```bash
cd /mnt/project/VLA-Drive-decodable-st-20261018
python -m tools.decodable_st.compare_decoding_endpoints \
  --left "$task_root/continuation_v2/evaluations/RS_dev16_endpoint1000/DECODING_REPORT.json" \
  --right "$task_root/continuation_v2/evaluations/Rfrozen_dev16_endpoint1000/DECODING_REPORT.json" \
  --output "$task_root/continuation_v2/DEV16_PAIRED_BOOTSTRAP.json"
```

15项相关CPU回归通过。统一入口已登记新终点评估命令与完整receipt身份；独立教师能力未通过时，正式阶段继续保持BLOCKED，不将算子测试或教师一致性写成能力PASS。

终点Adam二阶矩的补充诊断没有支持“大面积梯度被eps吞掉”的猜测：queries中RMS低于eps=1e-8的比例约0.66%，patch投影权重约0.004%。这不排除其他优化问题，也不是因果消融，未据此改学习率或eps。统计见[真实optimizer诊断](OPTIMIZER_SCALE_DIAGNOSTIC.json)。过程中两次CPU统计失败（超大quantile、float32索引舍入）留在私有失败收据中，修复采用明确抽样与int64索引，没有修改训练或容差。

本次四项解码评估的收据合计0.461 GPU任务小时；固定dev16面板、配对分析与评估均已完成并退出。没有后台续训，没有停用或恢复任何压力脚本。
