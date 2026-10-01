# 训练占卡资源恢复

C4 达到 50k 暂停后，原 `tools/foresight/run_allocated.py` 恢复了本任务的 8 个压力父进程，每卡 64GiB。原 allocator 通过 NVML PID 查容器 `/proc`，查不到时直接忽略该 GPU 进程；续训因此漏释放。归属已由 allocation ledger、完整 argv、GPU 环境、stdout 文件、UID 和进程组逐项核对。这是本任务调度错误，不是其他人的训练任务。

2026-10-01 11:11 UTC 释放这 8 个已确认压力进程后，C4 每步从约 4.9 秒恢复至约 1.87 秒；显存从每卡约 80GiB 回落至约 15–16GiB。0 个训练 / 无关进程被发信号。证据在外部 `navtest_milestones_20261001/resource_recovery_C4_20261001/`，保留原 ledger 和日志。

新增独立 CPU `tools.full_foresight.training_pressure_guard`，每 30 秒检查三台已授权主机，只有正式 run 身份 / 源码 / 主机一致且状态 RUNNING、30 秒内更新时才释放**完整验证归属**的占卡父进程。PAUSED、COMPLETE、FAILED、陈旧或其他身份状态不会触发释放。不启动 GPU 工作，不改训练源码或已有 Navtest 观察器，既有空闲时占卡政策继续由原 allocator 执行。

35 个针对性 CPU 测试通过，其中 28 个原自动评测测试及 7 个训练状态 / 身份反例。该 guard 使用干净、固定的新源码目录启动；训练仍为 d1d4085，Navtest 观察器仍为 693a1a9。真实启动 argv / PID / source SHA 和逐主机动作记录放在外部 `navtest_milestones_20261001/training_pressure_guard/`。预算结束或 STOP 仅停止 guard 检查，不停止训练或评测。
