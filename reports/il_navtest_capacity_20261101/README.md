# 检查点评测扩容：保护B/D并实际增加并发

2026-11-01 UTC。用户要求增加资源、尽快完成ckpt评测；保持之前B/D IL吞吐保护和Huber终止指令。

## 真实瓶颈与处理

1. 旧调度只尝试shared槽。实际两次GPU/owner采样和四namespace锁检查显示，很多显存空闲的设备shared槽在占用，但primary槽空闲。原有**已验证worker本来就支持primary+相同shared-headroom检查**，本次只让调度器使用它，不更改worker或夺取已有锁。
2. 每主机同时冷加载模型的上限为2。B26k两个末尾分片实际model_load分别约595/476秒，前向/评分分别约164/30和151/32秒；冷启动是明显开销。新burst默认每主机最多8个加载，并按真实可用RAM扣除128GiB余量、每额外加载预留32GiB再次限流。CPU/RAM紧张时自然少派。
3. TIME_RESIDUAL66k的World资源接口返回`available_memory_bytes`，且缺少total/used显存；原调度错误按另一schema读取。现通过**同一任务授权的remote接口**补采真实GPU/进程/RAM字段，未填0或伪造容量。修复后实际有限探测51条候选route、无probe错误，并排除B/D。
4. 原max3模型并发改为max4，仍采用每点32分片、单场景batch1。新CPU导出任务也优先离开B/D节点。

每次派发仍检查主机权限、物理UUID、已知/可见owner、显存余量和RAM，并由原worker在所选lease下再次验证。World保留其原shared worker；四架构使用原生支持的shared/primary两槽。没有新增第三种GPU执行槽，也没有改actor、精度、seed、分片人口或评分。

## 实际部署与执行证据

新持久CPU owner **2989441/start1494249184**，源码`d8a8dd52a7236a41b51d46933313d5ee163587c8`，工作区`/mnt/project/VLA-Drive-navtest-throughput-20261101`。沿用原`navtest_cadence_20261031/owner.lock`，没有第二套队列。

只替换原CPU dispatcher及C30k CPU burst owner；所有原GPU子进程继续，没有训练/GPU信号。C30k逐项接管原source/SHA/command/export/parity/已完成和在途分片，不改变world32，不重跑完整场景。

**部署后实际核对：C30k有32个RUNNING worker**，分布：

| 主机 | C30k worker数 |
|---|---:|
| vlawm-zt | 8 |
| vla-zt3 | 5 |
| tsinghua-zt2 | 8 |
| tsinghua-zt3 | 4 |
| vlawm-zt2 | 6 |
| rl-zt2 | 1 |

18个使用新primary路径、4个新shared、12个旧路径receipt（包括已退出export/parity）；32个GPU worker的RUNNING表示实际child已启动，部分仍在冷加载，不能当作32卡已满负载或32倍加速。

TIME_RESIDUAL66k已恢复原export，首个实际rank0/parity worker在vlawm-zt2运行，完成原入口native/export对照后再展开其他分片。Huber保持终止，未恢复其46k或重新构造已删检查点。

## 本次核对时的最新完整ABCD结果

| IL组 | checkpoint | 完整canonical v1.1 Navtest PDMS |
|---|---:|---:|
| A | 32k | 86.0692264051 |
| B | 26k | 86.7687640347 |
| C | 28k | 82.1780196449 |
| D | 20k | 87.5039121347 |

每行12146场景、单候选K10 shift5 clean-x ODE、seed2101、invalid0。C30k和TIME66k运行中，未发布部分人口均分。不同update不构成匹配架构排名。RL主线已有独立max64-worker评价owner，当前20个评价完成、低LR endpoint1000一个评价运行中；本操作没有重复其队列或恢复已终止ABCD RL。

## 检查、成本与可恢复记录

13项针对性CPU检查通过：空闲primary复用不抢shared锁、双槽皆忙时拒绝、旧probe真实字段补采、旧attempt接管/source拒绝、任务权限probe、逐2k保护和场景分片覆盖等；另有实际World资源探测和C32子进程启动核对。没有重做模型数学或全仓测试。

开发期间，导入检查捕获一次文件尾NUL padding，以及一次World源没有ABCD工具模块的错误引用；二者均在部署前修复，证据保留，不用失败版本启动GPU。成功部署源码为d8a8dd5。新probe/schema修复仅作用于调度观测，不改变任何合法trajectory或score。

资产：`/mnt/project/qwen-world-il-research-20261030/maintenance/eval_acceleration_20261101`中的RESOURCE_INVENTORY、QUALIFIED_CAPACITY、WORLD_CAPACITY_PASS、STATE_BEFORE、LAUNCH及实际执行后续快照。所有旧attempt/原GPU计时保留；旧分片复用计为逻辑完成，不能消除其历史物理计算成本。

```bash
# 已执行
/usr/bin/python3 -m unittest \
  tests.il_report.test_burst_recovery \
  tests.il_report.test_navtest_cadence -q

# 当前单owner入口；重启前先核对OWNER和owner.lock
cd /mnt/project/VLA-Drive-navtest-throughput-20261101
/usr/bin/python3 -u -m tools.il_report.navtest_cadence \
  --interval-seconds 600 --max-concurrent-models 4
```

B/D节点`training-vlawm-tsinghua-zt`、`training-vla-zt`在probe及route均显式排除；其原24锁保护保持。未抢占其他作业或新增主机授权。现有HTML owner收到刷新请求，训练和科学代码不热改。后续2k检查点继续自动保护、入队、完整评价及汇总。
