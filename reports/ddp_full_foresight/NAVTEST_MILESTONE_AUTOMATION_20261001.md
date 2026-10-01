# 70k / 80k / 90k / 100k 后台 Navtest 队列

用户要求 C0、C1、C4 在四个固定 update 进行完整 Navtest。新增独立观察器 `tools/full_foresight/navtest_milestones.py` 和检查点 / 资产契约 `navtest_schedule.py`，不改训练算法、已锁定训练源码 d1d40854299b9599b2accc382bcfc4b676dd7623 或现有控制器。

2026-10-01 代码提交前验证：28 项 CPU 测试通过（0.52 秒）；CLI help 实际运行。覆盖完整 exact-step 快照及滚动清理、原子恢复、任务去重、未知配置拒绝、只释放精确归属的压力进程，以及对历史 C0 66200 完整 12146 行 / 136 logs 官方结果的只读再验证。测试产生 0 optimizer updates、0 CUDA 工作负载、0 新 benchmark 结果。

未来任务使用 FP32 / TF32-off / seed42 / 10 步 / 单 ego，原完整精度 NAVSIM v1 metric cache。保留全量分母与失败，推理只读当前相机；GPU / CPU 异步，一台服务器最多一个任务。72 小时观察器和 150 GPU-process-hours 新任务调度软额度，原 6000GPUh campaign 上限继续执行。约 419GiB 的完整快照保留上限已检查，共享盘约 6558GiB 空闲。

代码发布后，从固定干净 detached worktree 建立真实不可变注册并启动后台。实际 source SHA、远端 SHA、注册 identity、三主机 preflight、PID / argv 和启动状态将追加为单独部署证据。此刻尚未到目标 update，不宣称未来 12 项评分已完成。此前 Navtest 结果完整保留。

使用及恢复见 `docs/NAVTEST_MILESTONE_AUTOMATION.md`。
