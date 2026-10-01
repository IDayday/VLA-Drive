# 自动评测 70k / 80k / 90k / 100k

入口：`python -m tools.full_foresight.navtest_milestones`。这是独立的后台观察器，不修改或重启训练器。

当前队列为 C0、C1、C4，各评测 70000、80000、90000、100000 个 **optimizer updates**，共 12 项。观察器每 10 秒检查 COMPLETE 检查点，先在外部目录硬链接保存完整模型、8 个 optimizer 分片和 8 个 RNG 文件，再交给独立 worker。原 periodic 检查点被训练器清理后，评测快照仍可用。找不到精确步数时记录 `MISSED_CHECKPOINT`，不拿更晚权重替代。

每台服务器同时最多一个评测任务，8 个推理进程共享正在训练的 8 张卡。CPU 评分仅在已审计的本机 NAVSIM 环境运行，每组 16 workers，GPU 推理和 CPU 评分异步进行。训练暂停进行原定开发集导出时，新的 Navtest 推理等待原控制器交接完成。

评分固定为完整 Navtest 12146 场景 / 136 logs，FP32 master 恢复、FP32 推理、TF32 关闭、10 步 FM、单 ego 候选、sampling seed 42，无 scorer。部署保留 W，去除辅助头。沿用原始完整精度 metric cache、完整官方环境和审计过的 v1 评分，逐场景核对原 cache hash、有限性和 PDMS 公式。该固定 checkpoint 测量不是最终 5-run 评价，也不用于测试集调参。原开发集 50k（C0/C1）与 25k（C4）明确标为同 run 的此前证据。

使用已提交且干净的独立源码目录。配置所有路径、主机和预算均在外部 JSON 中登记；注册后 config 身份、源码和资产 hash 不可变。启动示例（本机真实路径）：

```bash
cd /mnt/project/VLA-Drive-navtest-milestones-run-SOURCE_SHA
/usr/bin/python3 -m tools.full_foresight.navtest_milestones register \
  --config /mnt/project/ddp-full-foresight-study-artifacts/20260929/navtest_milestones_20261001/config.json \
  --output /mnt/project/ddp-full-foresight-study-artifacts/20260929/navtest_milestones_20261001/registration.json
/usr/bin/python3 -m tools.full_foresight.navtest_milestones export-group \
  --registration /mnt/project/ddp-full-foresight-study-artifacts/20260929/navtest_milestones_20261001/registration.json \
  --task C0_070000 --preflight
/usr/bin/python3 -u -m tools.full_foresight.navtest_milestones watch \
  --registration /mnt/project/ddp-full-foresight-study-artifacts/20260929/navtest_milestones_20261001/registration.json
```

`SOURCE_SHA` 在部署报告中换成实际短 SHA；后台的实际 argv、PID 和完整工作目录记录于外部 `launch.json`。断线或重启观察器后，用同一条 `watch` 命令恢复。flock、独立 run ID 和进程 argv 校验防止重复观察器或同组重叠任务；已经完成的结果不重算。`--once` 仅执行一次调度扫描。

输出均在外部 artifact 根目录：

- `status.json` / `TASKS.json`：12 项任务的最新状态。
- `SUMMARY.csv`：完成项的 PDMS、全部分项、零分、失败数、ADE/FDE。
- `jobs/C0_070000/complete.csv` 等：全量逐场景分数、分项与轨迹误差。
- 每项 `lock.json`、`snapshot.json`、预测/评分身份、日志及 `progression/`：检查点来源和同 run 逐场景 / log 配对差值。

错误保留全部日志和已完成行，标 `FAILED`；不会删除失败样本后报均值。预算或运行时间导致未完成则标 `PAUSED`，保留快照。人工检查原因后，可通过相同注册的 `task --task C0_070000` 重试；先确认该任务没有仍在运行的 worker / GPU group / CPU scorer。重试使用新的 attempt ID，并复用合法已完成预测和评分。

创建 artifact 根目录内 `STOP_SCHEDULING` 可停止新任务，已有任务继续完成，训练不受影响。要恢复调度，明确删除这个本任务标记后重启 `watch`。停止后的观察器不再持续捕获未来快照，不能假定这些 periodic 快照仍在。

本次预计 12 项约 93 GPU-process-hours，登记 150 GPU-process-hours 为**新任务调度软上限**，在跑任务不因该软额度而丢失；原 campaign 6000 GPUh 硬上限仍由 exporter 在场景边界执行。该共享 GPU 进程计时不代表新增独占 GPU 租用成本。CPU / 推理时限均记录；当前注册观察器最长 72 小时。保留 12 个完整恢复快照最多约 419 GiB，另计预测与 CSV。仅对 ledger、完整 argv、GPU 环境、stdout 路径、UID 和进程组一致的已授权压力脚本发送 SIGINT；训练结束后才在确认为空闲的卡恢复原压力命令。

验证命令：

```bash
/root/miniconda3/envs/ddp/bin/python -m pytest tests/full_foresight/test_navtest_milestones.py -q
```

CPU 验证包括精确快照、rolling GC、原子发布恢复、分片完整性、错误身份拒绝、重复调度隔离、开发集交接、真实历史完整评分核验，以及 CPU dummy 进程的压力脚本归属校验；这些测试不产生新的模型评分或训练更新。
