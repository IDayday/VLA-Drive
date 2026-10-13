# G0 60k完整Navtest（2026-10-13）

G0_NO_FUTURE_LABEL的60,000步权重已完成全量Navtest：**q_final PDMS 89.0247**。06:49:44 UTC完成队列校验，12,146个唯一场景、136个log，q0和q_final均无评分失败。完整预测成员与载荷hash、canonical评分和worker退出已通过既有控制器核验。

| 轨迹 | PDMS | 零分场景 | 评分失败 |
|---|---:|---:|---:|
| q0 | 88.0481 | 668 | 0 |
| q_final | **89.0247** | 543 | 0 |

q_final分项（0–100）：NC 98.1887、DAC 97.2337、EP 82.9364、TTC 95.1013、舒适度100、DDC 98.4192。原始精度数值保存在SCORES_COMPLETE.json。

| 比较 | PDMS差值 | log聚类95%区间 |
|---|---:|---:|
| G0 60k−同模型q0 | +0.9767 | [0.7077, 1.2819] |
| G0 60k−G0 50k | +0.1348 | [-0.1628, 0.4362] |
| G1 60k−G0 60k | +0.2575 | [-0.0422, 0.5766] |

G0从50k的88.8899小幅提高到89.0247，零分场景588→543。修复154个旧零分，同时引入109个新零分；PDMS区间跨零，尚不足以认定继续这10k步带来稳定提升。

同一60k预算下，G1为89.2823，比G0高0.2575分，但区间仍跨零。G0保留共同几何分支、未来表示和FGTR，关闭未来标签监督；本观察点尚不足以认定未来标签已带来稳定规划收益。

同模型修正修复153个q0零分场景，同时引入28个新零分，最终零分668→543。内部q0/q_final差值只用于修正诊断，不能替代独立训练消融。

上述区间使用136个log聚类、20,000次bootstrap，保留真实场景权重。只有一个训练seed，场景重采样不能替代训练复跑。用户要求中间观察点直接使用Navtest，需披露该测试集已反复观察。100k共同终点仍未齐，本次没有修改或停止训练。

## 执行与来源

本次按实际资源选择11卡：training-vlawm-zt2的3卡、training-vlawm-zt4的8卡；32个CPU评分进程随预测同步工作。推理占比1.0，主动限速0秒。正式注册至完整校验47.790分钟，累计推理进程时间7.809 GPU-hours，峰值allocated 12.548 GiB、reserved 13.078 GiB。该成本不含此前native恢复/profile和CPU模型加载，也不代表独占GPU账单；共享训练开销另计。资源数与上次26卡不同，不作为固定卡数或隔离性能A/B。

模型与冻结评价源码仍为`0c54f1c49e7ed2a76b22826b654d6c72625f5b6d`。使用严格原生FP32 master恢复、FP32推理、TF32关闭、原10步FM、单候选及scene绑定seed42，采样和canonical评价未变。实际checkpoint文件hash见QUEUE_COMPLETE.json；当前路由源码hash、两台主机原始入口对照结果分别见EXECUTION.json和FROZEN_PARITY.json。

配对统计见ANALYSIS.json，完整学习曲线包含18个已完成权重、两套轨迹的全部分项，见LEARNING_CURVES.csv。上传内容只有汇总数据及身份记录，未包含逐场景标识、图片、缓存或权重。

在training-rl-zt2可用以下既有命令核验结果，无需加载模型或派发GPU：

```bash
/root/miniconda3/envs/navsim/bin/python -B /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_parallel_20261013T0406Z/frozen_runtime_reserved/run_navsim_navtest.py --verify-complete --request-key G0_NO_FUTURE_LABEL_0060000 --output /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225
```
