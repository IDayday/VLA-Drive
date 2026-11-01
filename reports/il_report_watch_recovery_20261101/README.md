# IL评测正常、报告监听退出：实际修复记录

2026-11-01。用户询问长期没有新IL ckpt结果，核对实际训练、完整SUMMARY、队列及HTML发布链后发现：**训练和评测在继续，单一IL报告owner退出导致新完成结果未呈现。** 不能把旧HTML作为当前无评测的证据。

## 故障与修复

旧报告owner2659330在新增TIME66k评价目录的`inotify_add_watch`收到`OSError ENOSPC`后退出。此错误来自文件监听容量，不代表磁盘空间不足或模型推理失败。没有修改系统inotify上限，也没有停止其他监听器来争资源。

新`ILStateEvents`将自身目录watch数限制为512。监听/实例容量不足时保留现有可用监听，并降级为**每60秒仅查看报告refresh request与评价队列STATE**，不轮询训练日志或逐场景记录。常规完整报告仍1800秒，事件不重置常规时钟；新完成分数由原cadence发布refresh request触发更新。实际恢复时本机仍无可用inotify watch，已验证0个watch下报告正常运行。

新单一presentation owner **3022551/start1494793497**，report source `633ee52ec0b9736d9835804261bd79bc853dce8e`，目录`/mnt/project/VLA-Drive-il-report-watch-recovery-20261101`。复用原`tools.qwen_world_tasks.report --watch-il-report`扩展、原watch.lock、原HTML地址和T1/T2/T3区块。未重建网页或改变模型/训练/评分源码。旧owner已退出，没有启动重复watcher。

实际启动后生成`2026-11-01 05:02:06 UTC`报告，**143个完整结果、warnings=[]**。包含先前未呈现的C30k与TIME66/68/70/72k。实际owner为RUNNING，backend为`linux_inotify_with_metadata_fallback`，training_journal_polling=false。

## 最新完整结果

| IL组 | checkpoint | canonical v1.1 Navtest PDMS |
|---|---:|---:|
| A | 32k | 86.0692264051 |
| B | 26k | 86.7687640347 |
| C | 30k | 86.6836711864 |
| D | 20k | 87.5039121347 |
| 时间分布加权 | 72k | 88.3303794102 |

均为原单候选K10 shift5 ODE、seed2101、完整12146场景。表按各组最新检查点列出，不是各组历史最高分，也不是匹配训练预算的架构排名。没有因本次报告修复重新推理或重评分。

查验时训练仍推进：A约34.0k、B27.7k、C31.6k、D21.6k、TIME72.6k，rank0/7的update一致且日志新鲜。A34k刚封存，已唤醒原评价CPU owner2989441并确认入队、实际RUNNING；B28k/C32k/D22k当时尚未封存。**每2k更新评价不意味着每半小时必定有一个新权重。** 2k原生update的生成时间与评测时间要分别计算。

HUBER保持用户终止、已清理原生checkpoint，不恢复评测。B/D新增GPU及重CPU任务保护继续，报告修复不向任何训练或评价进程发送停止信号。

## 已执行检查

- 注入watch ENOSPC，验证报告监听对象存活且refresh request仍能触发事件。
- 注入inotify实例EMFILE，验证timer/wakeup/关闭路径可用。
- 20个历史目录、最大4个watch的缩小fixture验证不会无限增加watch。
- 实际本机容量已耗尽，因此原“正常inotify事件”测试显式skip；本机12项中11项执行通过/1项环境skip。
- 同一源码在已授权tsinghua-zt2做有限纯CPU测试：**12项全部通过**，包含正常事件、日志不触发事件、去重、固定时钟及新容量异常测试。没有占用GPU。
- 实际HTML/快照刷新、原task-suite区块保留、143个完整结果及新owner运行状态复核。

复现：

```bash
/usr/bin/python3 -m unittest \
  tests.il_report.test_event_capacity tests.il_report.test_report -q

# 仅在旧owner退出且watch.lock空闲时，复用原扩展启动单一新报告owner
cd /mnt/project/VLA-Drive-qwen-world-task-suites-20261031
/usr/bin/python3 -u -m tools.qwen_world_tasks.report --watch-il-report \
  --il-source /mnt/project/VLA-Drive-il-report-watch-recovery-20261101 \
  --il-root /mnt/project/qwen-world-il-research-20261030
```

旧OWNER/REPORT_CONFIG/EVENT_STATE、实际LAUNCH、恢复后的核对记录保留在`/mnt/project/qwen-world-il-research-20261030/maintenance/report_watch_recovery_20261101`。本目录[verified_results.json](verified_results.json)是有限时点快照，实时状态以原报告和OWNER为准。
