# Latency and training efficiency

GPU资源更新：初次资格检查时vlawm-zt/2/3均不可用；23:39 UTC复查发现原允许主机`training-vla-zt2`的8张A800空闲，后续GPU/NCCL、真实框架和吞吐验证已在该机执行。没有停止其他科学任务或未知进程。三台vlawm的历史阻塞证据保留，不代表最终所有GPU仍阻塞。

真实A800 80GB测量：batch1、FP32参数/原VLM BF16 autocast、warmup3、重复10次，每段测量前后CUDA synchronize。在线路径从实际三张图像文件读取/验证、原预处理到选中物理轨迹CPU输出。K=1/2/3使用双轮smoke真实权重；K=5补了2个临时未训练S0副本，仅测容量，未写registry或银行。各K计时时全部5个头驻留，峰值显存不能解释成只装载K个头的内存需求。完整samples在`evidence/GPU_LATENCY.json`。

| K | image-to-selected p50 ms | p95 ms | 全部Decoder缓存路径 p50 ms | Scorer p50 ms | GPU峰值 GiB |
|---|---:|---:|---:|---:|---:|
| 1 | 570.78 | 572.10 | 3.55 | 2.50 | 8.63 |
| 2 | 577.41 | 580.48 | 6.80 | 2.61 | 8.63 |
| 3 | 579.52 | 580.67 | 9.96 | 2.61 | 8.63 |
| 5，2个临时容量副本 | 588.58 | 591.89 | 16.33 | 2.64 | 8.63 |

K=1实际选轨路径直接Base，不调用Scorer；表中其Scorer一列是独立额外计时。预处理/文件验证p50=119.36ms，共享VLM（含processor/tokenizer/visual）441.43ms，Q-Former/ego=2.24ms，选择/CPU copy约0.094ms。每个Decoder独立明细见JSON。各段独立测量，不能要求其统计分位数严格加和。smoke校准规则为always-Base；不是科学收益下的正式生产负载。

CPU对照：Intel Xeon Platinum8378A、8线程、batch1、warmup1/重复2次，K1/2/3/5总时延p50分别31731.95/31460.50/31844.27/31589.70ms。CPU轮次1只有2个已训练候选，K3/5使用显式临时副本；测量期间本任务有少量并行CPU审计，因此仅为诊断，不能给出可靠尾时延。`evidence/CPU_LATENCY.json`保留在线总时延与缓存Decoder成本，二者不混称。

GPU训练首轮实际吞吐：8张A800、global batch32、microbatch4/rank、全部ego IL/current-DINO/future-clip/interaction目标，完成8 optimizer updates。排除前3步后p50=3.365s、p95=3.873s，预先30s/update预算通过。完整CLI时间113.39秒（含初始化和约22GB完整checkpoint）；训练器计GPU秒621.62，另计启动开销，不能只报计算时间。最终冻结代码另执行独立profile版本，结果见运行记录和最终训练报告。

首次8卡预跑在更新前触发原语言模型reentrant checkpointing与DDP unused-parameter检查冲突，失败日志保留。修复仅将已启用checkpointing的语言层改为non-reentrant，仍保存/恢复RNG和激活重计算；未删loss、未冻结额外参数、未改输入或global batch。带dropout的前向/梯度逐tensor完全一致测试通过，随后真实8卡backward通过。原source文件未修改。

无损I/O优化：完整冻结FeatureBundle复用与按场景分片；候选/分数按单候选hash增量缓存；独立官方worker有界并发；Scorer验证按场景分块；checkpoint流式写与读取权重时mmap。全量数据import使用16线程有序map，仍核验每张实际图像、GT和metric context；384条真实记录的顺序/全部字段/hash与串行版完全一致。并发专家验证与原串行官方验证JSON也完全相同。没有通过减相机、降分辨率、减候选、改目标或改有效batch提升速度。

正式基础训练每1000步保存完整约22GB checkpoint，100k更新约2.2TB；专家/Scorer仍每200步验证/保存。此项降低I/O和存储成本，不删历史权重、不改变优化步骤。基础profile的checkpoint完整保存，其耗时没有从总成本中隐去。

旧LoRA-DiT集合尚未绑定可运行、输入/预算对齐的基线：UNMEASURED，没有给出架构加速倍数。Router真实预路由独立计时与训练监督见对应证据；不能用先跑全部Decoder的离线对照冒充预路由。

Router独立A800实测（同一真实K=2池、warmup3/repeat10）：网络p50=0.411ms，真正预路由缓存轨迹p50=4.794ms，完整图像到轨迹p50/p95=575.162/581.819ms。所用独立校准规则回退Base；该结果只反映执行成本，不代表Router有质量收益。见GPU_ROUTER_LATENCY.json。

最终代码334c000的独立profile_v3完成8步，稳态p50/p95=3.249/3.892秒，30秒预算通过，CLI总耗时约115.3秒、训练器GPU秒647.32。最终8卡正式训练使用同一模型/损失/批量配置，独立长schedule。全量启动前发现并修复来源校验反复构建9万ID集合的O(N²)开销，改为一次构建集合；原正式启动在0次更新时终止，未丢失任何optimizer进度。修正后的代码重新做完8卡profile再启动正式作业，未把失败/重启成本隐藏为正常步时间。
