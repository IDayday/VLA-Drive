# ABCD IL每2k完整Navtest评测恢复

2026-11-01 UTC。用户要求恢复ABCD IL每2k有效optimizer update的持续评测。训练、导出器、原始观测推理、轨迹解码与canonical v1.1 scorer没有修改。

## 中断原因

周期队列CPU owner仍存活，但A24k、C22k的burst owner在worker连续四次资源准入拒绝后退出。D18k随后遇到同类问题。队列将资源等待标为`FAILED_REVIEW_REQUIRED`，并禁止该方法继续登记新点，导致A/C后续评测停滞。全部已检查的非零退出都是`REFUSED_RESOURCE / child_not_started=true / returncode=75`，不是模型/评分失败。

另一个问题是burst调度层沿用了旧任务的probe/remote权限入口，既可能把较宽调度规则识别为已知的cohost分配给较严格的旧worker，也漏用了本次任务已授权的主机。我们保留旧worker的严格准入条件，改为在调度前直接执行其只读owner检查，并使用本任务的资源入口；没有扩大主机授权或放宽未知进程检查。

## 修复内容

- 支持四架构burst的精确分片接管：核对source路径、SHA、输出目录和attempt身份。复用原export/parity、已完成分片和仍工作的分片，只有缺失工作继续派发。
- 只有确定未启动child的资源拒绝进入有退避的资源等待，保留全部旧attempt。实际推理/评分失败仍明确停止，不自动吞掉或转为reward0。
- 资源探测通过本任务delegate，并复用实际旧worker的`_known_owners`检查。GPU UUID、显存余量、RAM及lease检查保持，未知或不可见owner仍排除，不驱逐任何作业。
- 已登记系列按最早可用的后续2k边界依次入队，避免跳到最新点。ABCD有旧点等待时仍保护新边界，防止训练滚动清理造成后续漏测。
- 每个新burst使用按脚本hash命名的冻结副本；旧脚本和失败证据保留。调度并发上限显式配置，本次设置3个模型，以先处理三个中断点。

## 实际恢复状态

恢复CPU owner：`2857207 / start_ticks=1493044809`。启动时执行源码`5ef0a627ec51d284ec3913f72f9d29dfda7af52d`，目录`/mnt/project/VLA-Drive-il-navtest-cadence-recovery-20261101`。旧CPU owners经pidfd和start_ticks核对后退出；训练/GPU进程收到信号数为0。接管过程中已运行的远程推理分片持续执行。

| IL组 | 本次续跑 | 已保护待测 | 下一个尚未保存的2k边界 |
|---|---|---|---|
| A | 24k，原23个完整分片复用 | 28k | 30k |
| B | 22k已完成 | 等24k保存后自动登记 | 24k |
| C | 22k，原9个完整分片复用 | 26k | 28k |
| D | 18k，接管原有在途分片 | 由本次续跑继续补齐 | 20k |

本次已确认新A/C worker具有实际`RUNNING.json`和child PID；C新worker已进入`training-rl-zt2`及`training-vlawm-zt2`，不只是启动CPU控制器。D新burst owner已启动并接管旧分片。表中是恢复快照，不是这些评测已经全部完成。

**A26k、C24k已被原生滚动保留规则清理，本次未找到可供严格评测的完整对应权重。** 不用25k或邻近权重冒充它们。该缺口保留记录；从当前恢复点开始保护每2k边界。

每点仍是完整12146场景、固定seed2101、单候选clean-x K10 shift5 ODE、同一canonical NAVSIM v1.1。32分片只改变任务分工，不改变单场景batch1或noise seed。只有全人口完成才发布PDMS，部分结果不冒充最终分数。

## 验证和复现入口

已执行11项CPU针对性检查，包含跨source/输出接管拒绝、最新attempt接管、真实worker错误不当作资源等待、任务probe归属、已有排队点不丢下一2k边界、native硬链接保留、分片覆盖和去重。实际接管三个原评测，检查新GPU child启动；没有改模型数学，也未重跑全仓验证。

```bash
# 已执行的针对性测试
/usr/bin/python3 -m unittest \
  tests.il_report.test_burst_recovery \
  tests.il_report.test_navtest_cadence -q

# 当前持久调度命令；仅在现owner退出后重启，owner.lock拒绝重复启动
cd /mnt/project/VLA-Drive-il-navtest-cadence-recovery-20261101
/usr/bin/python3 -u -m tools.il_report.navtest_cadence \
  --interval-seconds 600 --max-concurrent-models 3
```

控制状态：`/mnt/project/qwen-world-il-research-20261030/navtest_cadence_20261031/{OWNER,STATE,REGISTRATION}.json`。600秒检查的是检查点/任务元数据，不是高频训练日志监控。阶段汇报仍沿用原HTML owner及固定周期。

完整恢复收据、之前的STATE/OWNER/失败attempt引用、两次实际资源探测位于：
`/mnt/project/qwen-world-il-research-20261030/maintenance/navtest_cadence_recovery_20261101`。

未上传checkpoint、候选缓存、图像或场景原始数据。本分支只修改调度/恢复工具及测试，不更改四组IL配方或已有结果。
