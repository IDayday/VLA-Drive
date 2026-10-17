# B-15k只读分析衔接

2026-10-17 12:26 UTC，为既定15k观察点增加有限CPU后续任务。正式B训练、两个数值评价worker组、保存批次诊断和成本捕获各由原任务负责；这个任务只等待它们的完整证据，不启动训练或重复推理。

触发条件包括：15k world8/READY原生检查点及清单hash、精确成本与进度快照、已完成保存批次诊断、两个完整人口的评价和对自身SFT配对摘要、16份关闭的worker调用账本、按PID和启动时间确认评价进程退出。未满足条件只等待，不给缺失结果填值或自动重试其他任务。

触发后顺序执行九项已有分析：训练前缀、人口/源epoch流核对、DEV与TRAIN的安全转移、DEV与TRAIN相对10k比较、固定权重双网格差、DEV学习曲线、原生optimizer及完成状态核验。前缀使用e935快照，其余已有分析使用8dbd；数值推理仍为45d，正式训练仍为225b。分析输出已有则拒绝覆盖，失败尝试保留且不自动重试。绘图前拒绝已混入20k结果的情况，防止将更长曲线误标为through15k。

已执行9项针对性检查及6个现成CLI的真实`--help`：包括已完成10k证据接受、尚未生成的15k等待、所有辅助脚本编译、成功执行账本、既有输出保护、非零退出保留、失败尝试拒绝复用、实际`--once`检查。测试只创建临时CPU输出，没有模型推理、评分或optimizer更新。这些是衔接验收，不代表真实15k分析已经完成。

CPU任务已在授权vla-zt启动，启动时128 CPU affinity、load约9.79、MemAvailable约858GiB；进程523312、启动ticks1367860889，现场确认正在等待。4小时截止只约束这个分析等待器，超时不会被解释为训练停止，也不会停止训练。其脚本、资格验证和资源回执分别见[可执行记录](reproduce_B_RL15k_analysis_handoff.py)与[验证记录](B_RL15k_analysis_handoff_validation.json)。

以下命令已实际执行；当前已有唯一owner，不应再次启动普通入口。只读`--once`可检查当前前置证据：

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=/mnt/project/fm-transfer-20261016/source/8dbd2bb2c82026d3a8a0b4b96c9b0ba211248f81 \
  /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  /mnt/project/fm-transfer-20261016/resources/B_RL_15k_analysis_handoff_20261017/run.py --once
```

已完成10k与此前结果保持原hash，未增加训练预算、seed、采样器或科学比较。真正15k结果产生后仍需核对数值、查看图表并发布报告；最终Navtest门槛不变。
