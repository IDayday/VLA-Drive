# C框架按用户要求终止

2026-11-02，用户要求停止C框架实验、将资源用于其他实验。本次为实验生命周期调整，不修改任何算法、训练目标或评价协议。

- C IL在完整optimizer边界保存到`update045228`，原生worker返回码为0，checkpoint具有`COMPLETE`标记。没有完成原定100k预算。
- 已核对清华zt3的实际进程，C训练worker和八个训练rank均已退出。
- run-local `USER_TERMINATION.json`及全局生命周期hold禁止自动恢复和新评测。主campaign只管理A/B/D；替换CPU调度owner没有向B/D的GPU进程发送信号。
- 终止时15项已登记C周期Navtest均已完成，没有待运行或运行中的C周期评测。已有评分、轨迹、权重和日志保留。
- 清华zt3仍有三个既有RL任务共享八卡：`IMPROVEMENT_VERIFIED_RETENTION_G8`、`IMPROVEMENT_MIXED_NOISE_G8`、`IMPROVEMENT_FP32_OUTPUT_G8`，各有8个实际GPU进程。释放的C资源供这些现有任务使用，没有重复启动训练或干扰B/D。

实际停止与GPU核验收据：

```text
/mnt/project/qwen-world-four-frameworks-20261029/operations/C_user_termination_20261102/
  STOP_REQUEST.json
  CONTROLLER_BEFORE.json
  CONTROLLER_ABD_LAUNCH.json
  TSH3_GPU_EXIT_AND_REUSE.json
  STOP_VERIFIED.json
```

最终checkpoint：

```text
/mnt/project/qwen-world-four-frameworks-20261029/runs/ARCH_C/IL_seed42/checkpoints/update045228
```

动态IL报告收到行政刷新请求。C必须显示为“用户已终止”，不能因原生任务返回码为0被表述为完成100k，也不能因旧报告包含100k计划而再次启动。

本次提交仅保存终止记录与小型核验摘要；不包含权重、缓存或原始数据。
