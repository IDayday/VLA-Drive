# Qwen-Flow：单机、多机与恢复后的效率实测（2026-10-24）

正式IL/RL的actor、动作编码、采样、reward、优势和loss保持不变。这里是隔离执行副本的性能与数值资格测试，不是新训练方法，也没有将测试权重用于正式初始化。汇总证据见[完整数据](PERFORMANCE_SCALING_SUMMARY.json)。

## 结论

原IL的13.64秒/更新已回到约6秒，但恢复发生在受控保存/重启**之前**，不能归因于重启。空闲新主机的八卡副本为4.58秒/更新，原主机冷启动副本为6.19秒。单卡可以容纳IL；保持global batch32时没有八卡快。当前跨机走TCP Socket，不能按机器标称200Gb InfiniBand推算收益。RL全G8、随机步骤分片的数值等价通过，但实际共享16卡比专用8卡慢，本轮没有将正式训练切换成16卡。

## 相同与不同预算

|测量|GPU|global scene batch|平均关键路径秒/更新|scene曝光/秒|
|---|---:|---:|---:|---:|
|单卡副本|1|4|4.0555|0.9863|
|新空闲主机副本|8|32|4.5814|6.9847|
|原IL主机冷启动副本|8|32|6.1856|5.1733|

单卡一次处理4个scene，不能将4.06与八卡4.58直接当作相同batch。单卡维持32scene粗略需约32秒，这只是线性曝光估计，未运行32次累积的正式单卡计划。八卡两次独立完整模型profile采用相同53k模型副本、microbatch1×累积4，2次warmup和5次计时更新。profile的Adam是新建副本，不用于替代原正式Adam，也不宣称等同于带成熟Adam的长期训练吞吐。

单卡按CUDA事件分段的平均耗时：数据/processor0.300秒，Qwen0.549，condition provider0.099，ActionDiT/FM0.580，backward2.284，梯度检查及裁剪0.048，AdamW0.171。八卡总backward还含通信及重计算。CPU数据、宿主机负载、GPU共享和通信同时存在，当前实验没有将这些因素完全解耦。

## IL慢速的定位边界

八个rank逐更新取最大耗时，再对固定窗口求均值：59400–59449为13.6409秒；59450–59499为9.6045；59500–59549为6.0272；之后至59787均约5.98–6.01秒。恢复后的59788–59887为6.2310秒，排除前10步为6.1936秒。因此重启没有构成额外提速证据。

此前监测到八卡均有约1.6GiB未归属当前可见IL进程的占用，后来消失，GPU利用率及主机负载也变化。具体所有者未确定，不将未知进程当成压力脚本；没有终止这些未知任务。py-spy短采样主要可见backward，但大量native栈无法归属，不作为进一步精确定位的证据。

在59787完成合法边界保存、退出、运行冷profile，然后使用同一d4d0012源码、八卡、global32、原Adam、scheduler、RNG和数据游标恢复。八个rank首个恢复token及LR检查通过。全过程有约11分钟诊断空窗，不应从总成本中删去。[恢复证据](IL_RUNTIME_REBUILD_EVIDENCE.json)。

## 多机通信

|连接/负载|64MiB all-reduce中位关键路径|算法有效带宽GB/s|
|---|---:|---:|
|vla-zt3本机两卡|0.618毫秒|108.67|
|vla-zt3与SOURCE主机两rank|50.329毫秒|1.33|
|vla-zt3与CANONICAL主机两rank|43.018毫秒|1.56|

容器内没有/dev/infiniband，日志为NET/Socket，libibverbs不可用；/sys可见200Gb设备不等于实际用上RDMA。IL完整FP32梯度约13.39GB，当前未做正式双机IL训练，不能把两rank带宽当作完整16卡实测，也没有据此声称多机永远不可能受益。RL的LoRA梯度约63MB，小得多，但整条pipeline还受rollout、评分和共享资源影响。

## RL真实保存链的分片对照

使用同一20k保存模型/Adam副本、collection5000的真实链、old概率、优势和永久reference；G8整组不拆。16rank将每个逻辑rank的三个随机步骤分给两个物理rank，并按world/logical_world×local_K/3恢复原损失归约。

- 梯度相对L2误差9.10e-8，最大绝对误差1.60e-10。
- 四次Adam更新后adapter最大绝对差1.49e-8。
- 专用八卡后三次更新平均1.2952秒；八张辅助卡＋与正式SOURCE共享的八卡为1.6107秒。
- 共享16卡约慢24.4%；该结果混有共享、主机和拓扑差异，不能宣称专用16卡也一定更慢。
- 没有新增rollout或评分，没有更改正式模型或有效训练预算。

[分片数值证据](RL_STEP_SHARDING_EQUIVALENCE.json)。旧CLI的`--run`被torchrun缩写解析拒绝的失败已保留，改成`--native-run`后另起资格attempt；共享GPU测试显式限制allocator容量，不把这些执行配置混入正式配方。

## 可执行入口与源身份

实际入口：`scripts.analysis.qwen_sourceflow.benchmark_collectives`、`profile_il_distributed`、`profile_rl_step_sharding`、`validate_temporary_capacity`。实际完整命令和每rank原始记录保留在`/mnt/project/qwen-sourceflow-native-20261018/performance_20261024/jobs/`及各profile目录。这里只上传聚合证据，不上传模型、optimizer、梯度、私有环境或原始scene数据。

这些入口只允许当前明确授权的主机；临时vla-zt3原授权在两组旧RL达到50k时关闭。其后用户单独授权vlawm-zt和tsinghua-zt6的1–7号卡用于Navtest，不应据此扩大性能实验或训练授权。正式IL仍为d4d0012，RL仍为434cba8；本性能分支独立。
