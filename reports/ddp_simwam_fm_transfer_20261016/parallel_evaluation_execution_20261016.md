# 已执行：新两组里程碑评价并行

2026-10-16 20:40 UTC完成CPU评价控制器交接。旧owner在没有活动GPU评价时按`STOP_AFTER_TICK`退出，新owner经过进程/PID-start、旧锁释放、36项任务清单一致性检查后启动。未向A/B训练进程发送信号，也未修改它们的不可变源码。

新控制器使用`d58485b08078a6319b2835793d8c79ffbbd3e9d5`。实际推理worker继续使用`45d2318d2dec265583860a26f891d8b79bb6068a`，八分片派发入口仍为`225b9af660cb0210208e2c46fc4216d0f151275b`。新增的只是独立里程碑的主机路由和资源核查，不改变原始观测、scene绑定seed2101、每scene单候选、native/shift5网格或canonical评分。

资源核查确认tsinghua-zt与rl-zt2共16个不同GPU UUID，两个时刻都有足够显存且无未知所有者。两台均有128个可见CPU affinity，读取时load约11/7；CPU评分worker仍各使用一个数值线程。没有停止Base/G或清理未知进程。

B的25k SFT原生checkpoint随后封存并导出：权重SHA256为`250721a234e0eedafcfbea849747ebf92d3e39ddd5de161332019835b624839e`。20:43 UTC实际确认：

| 已登记任务 | 主机 | 实际派发 | 当时进程状态 |
|---|---|---:|---|
| B25k完整DEV1696场景 | tsinghua-zt | 8个独立shard | 8个PID/start全部存活，加载模型中 |
| B25k固定TRAIN诊断512场景 | rl-zt2 | 8个独立shard | 8个PID/start全部存活，加载模型中 |

这不是第三条实验。两任务并行后只等待各自完整分母及已有analyzer输出，不从未完成的部分场景推断均分。B继续100k SFT，A继续自己的20k RL。部署评价仍分别报告两种网格，不选最高网格。

新增4项里程碑路由测试与30项终点准入/派发测试全部通过。实际跨主机共享文件锁和已结束worker的远端PID检查也通过。真实CLI在最终LOCK不存在时拒绝派发，未生成新Navtest候选；本轮最终Navtest仍关闭。

这些测试验证调度、去重、重试和证据边界，不验证RL一定提升。只在两组完整终点及固定开发评价完成后发布最终锁。新最终派发入口已经实现并做资格检查，但没有提前执行终点推理。

机器可读执行证据：[parallel_evaluation_execution_20261016.json](parallel_evaluation_execution_20261016.json)。该快照记录的是当时实际进程状态，最终B25k分数会由独立完整评价报告提供。
