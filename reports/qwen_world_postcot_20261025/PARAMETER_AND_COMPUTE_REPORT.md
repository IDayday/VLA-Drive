# 实际参数与成本

以下来自完整真实GPU实例，不是理论预算。

| 模块 | 参数 |
|---|---:|
| Qwen vision主体 | 306,242,560 |
| Qwen language＋final/DeepStack mergers | 1,821,289,472 |
| spatial/global provider | 37,985,280 |
| predictive provider | 9,479,680 |
| clean-x actor trunk，含time/ego等embedding | 467,116,032 |
| clean-x action输出projection | 3,075 |
| DINO readout | 7,374,336 |
| future readout | 18,153,472 |
| current object readout | 2,385,168 |
| post head | 8,939,521 |
| 完整IL可训练总量（joint） | 2,678,968,596 |
| IL部署总量 | 2,642,116,099 |
| RL attention LoRA可训练量 | 5,242,880 |

动作本体两行合计467,119,107，含bias/norm的20个blocks为461,844,480。
Qwen视觉总量含mergers为406,957,056：主体306,242,560，final merger25,174,016，三个DeepStack mergers
75,540,480；后两项合计100,714,496归入language/merger学习率组，不重复计入总数。

RL为160个attention投影的320个A/B张量，不训练另一套编码器。
保留LoRA wrappers的RL部署总量为2,647,358,979；不把训练期永久reference的参数
算进部署，也不把冻结reference遗漏为“没有训练显存开销”。
冻结DINO教师303,079,424参数；冻结WA encoder+projector306,786,816参数。
教师不常驻规划训练GPU，不进入部署。

BF16准备K/V的逻辑大小：20层×192token×1024×K/V×2bytes=15MiB/scene，
batch4为60MiB；不等同于全部autograd激活或reserved显存。
完整真实梯度诊断峰值allocated25,214,817,792、reserved26,747,076,608bytes，
该单卡诊断不是正式8卡吞吐/显存结论。

100k×32=3,200,000次scene呈现，约90,890场景的35.21遍；1000接口更新包含在内。
正式RL50k更新、inner4意味着12500轮fresh collection，8卡合计约100000次fresh
scene曝光、G8约800000条新候选；不把inner更新重复计作fresh场景。

真实教师payload为3,344,400bytes/scene：FP32 DINO、原生BF16 WA、mask和时间。
完整TRAIN约304GB decimal，加DEV约5.7GB及索引/shard开销，不是TB级视觉缓存。
checkpoint与导出另占空间：每1000步恢复点、每5000步milestone、有限rolling；
不删除任何旧项目权重。最终wall/GPU成本应以完整运行journal计算，尚未完成不填预计为实测。

截至2026-10-25 17:40 UTC，本任务已登记的runtime/jobs GPU租约槽位累计20.11 GPU小时，
按物理UUID时间区间去重后19.72 GPU小时；共享作业不能重复算作不同物理GPU。
这包括对应已登记任务的加载、等待、保存及失败尝试，不等于CUDA kernel活跃时间。
早期未用该收据框架登记的诊断及初段RGB等不在此统计内，因此这是**部分成本快照**，
不能包装成全项目总成本。全量候选准备另有真实子进程CPU计时5.326核小时。
证据：`audits/COMPUTE_COST_SNAPSHOT.json`；正式IL/RL及最终评测总成本仍在累积。
