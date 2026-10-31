# D组迁移至空闲八卡：实际完成

用户要求将D组迁至空闲八卡提高效率。2026-10-31实际完成：`training-vlawm-tsinghua-zt5` → **`training-vla-zt`，8×A800**。目标节点两次NVML/可见进程/物理UUID/CPU/租约检查通过，原来为空闲整机。新节点当前GPU进程仅D；旧节点保留原PPO_SAFETY_EP，没有停止其他训练。

## 恢复与科学身份

- D在有效optimizer update **10,671**、cursor341,472合作保存后退出，未回滚到10k。八个优化器分片与rank RNG/builder状态完整；原worker、torchrun及rank退出均验证。
- 原`formal_fdaff80`科学源码与完整run identity不变；global batch32、八卡ZeRO2、seed42、损失、数据、scheduler及100k终点均不变。
- 原生严格恢复模型、Adam、scheduler、RNG；八个rank首个新更新均为10,672，数据cursor和前后实际LR衔接一致、optimizer计数连续、跳步0、梯度有限。实测窗口共同推进至**10,703**。
- 仅短暂接替CPU campaign owner，原A/B/C和全部RL learner保持运行；唯一campaign已接管D的新invocation。

## 吞吐

采用迁移前最近64步与迁移后32步，各rank中位耗时的最大值：

|窗口|秒/有效更新|
|---|---:|
|迁移前共享节点|9.6003|
|迁移后空闲节点|7.2142|

观测吞吐约 **1.33倍**，单步耗时减少 **24.9%**。这是不同时间窗口和场景下的实测，不是同batch独占硬件因果基准；不将短窗口精确外推为全部剩余训练ETA。此前即时读数约10.9秒来自更早的窗口，以此封存的完整前后窗口为准。D的原生Qwen共享动作计算开销仍存在，迁移没有缩模型、减token或改任务。

## 保持独立运行

D worker持有primary GPU leases；CPU-only guardian接替启动测速期间的shared/research槽预留，避免项目评测或RL再次与D同卡。它不创建CUDA context、不向模型发信号；随本次精确invocation退出、授权撤销或`RELEASE_RESERVATION`请求释放。不会抢占任何未知进程。其他空闲节点仍可正常回填任务。

## 实际执行与复核

已执行入口（不要重复迁移；已有NEW_INVOCATION会被拒绝）：

```bash
/mnt/project/flow-safe-improvement-20261013/venv/bin/python -u -m tools.il_report.rehost_arch_d
```

运行于本分支工作树。主代码只复用原生stop/save/restore与Campaign._launch；未改运行中的formal/ops快照。guardian的原始执行脚本位于`/mnt/project/qwen-world-four-frameworks-20261029/resources/D_to_idle_vla_20261031/hold_secondary_slots.py`，本目录同名文件是逐字节归档，依赖原资产目录中的收据，不能直接从归档位置启动。

真实运行收据根：`/mnt/project/qwen-world-four-frameworks-20261029/resources/D_to_idle_vla_20261031`。包含PREDRAIN、NATIVE_STOP_COMMITTED、NEW_INVOCATION、POST_PLACEMENT、CONTINUITY_VALIDATION、COMPLETE、RESERVATION_OWNER。当前状态以运行目录为准，[此处验证快照](validation.json)不代表后续update不再变化。32项既有针对性测试通过；迁移验收包含真实八卡续训和前后吞吐，不以“已启动”当完成。
