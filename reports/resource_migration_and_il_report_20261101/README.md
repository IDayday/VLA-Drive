# 2026-11-01：资源撤回、原生恢复与 IL 报告更正

本记录区分已经执行的恢复和仍在等待截止时间的退出守护。没有修改 actor、损失、随机采样、评分版本、训练预算或参数精度。

## IL 最高分显示更正

用户指出首页最高 IL 已变化。检查发现报告仍在定时生成，但首页选最高分时排除了 `ARCH_*`；完整结果表中已经包含 ABCD。因此这是展示统计范围错误，不是重新评分得到的新成绩。

截至 2026-11-01 16:02 UTC 的已发布页面包含 161 个去重后的完整 IL Navtest 观察点。只读重新发现结果也得到 161 个。

| 模型 | 更新数 | canonical v1.1 PDMS |
|---|---:|---:|
| 架构 D | 26,000 | **89.0477481734** |
| 原 Qwen-World / 原始 L1 | 70,000 | 88.7929099070 |

上述分数均为 12,146 场景、invalid=0、seed2101、单候选 K10/shift5 ODE。差值为 +0.2548382664 分；这是不同架构/预算的已测最高观察值，不是匹配预算的因果结论。D28k 为 87.9830815164，不能称为单调提升。

修复已独立推送：`fix/il-report-all-architectures-best-20261101`，提交 `058dd96546c502317f29ef1b64c9e5f877d42a1d`。运行中的报告源码固定在 `/mnt/project/VLA-Drive-il-report-all-architectures-best-20261101`；原页面仍是 `/mnt/project/qwen-world-il-research-20261030/report.html`。继续复用唯一报告 owner 和原来的半小时/重要事件更新机制。

实际完整分数证据：

```text
/mnt/project/qwen-world-four-frameworks-20261029/periodic_NAVTEST_ARCH_D_026000_20261101T113921Z/D_IL_026000/SUMMARY.json
/mnt/project/qwen-world-il-research-20261030/maintenance/headline_best_il_20261101/PUBLISHED.json
```

## 已执行的资源与恢复操作

最新撤回的 `vla-zt3`、`vlawm-zt2`、`vlawm-zt3`、`vlawm-zt5` 已在全局派发层禁止新任务和失败重试，`tsinghua-zt4` 继续停用。用户随后授予的截止时间不包含这四台。

- `vlawm-zt2`、`vlawm-zt3` 的行政退出检查未发现可见 GPU 进程。
- `vla-zt3` SSH 认证失败、`vlawm-zt5` 连接拒绝，不能声称已通过物理退出检查。失联主机的旧 RUNNING 文件不是当前进程存活证据。
- 时间分布加权 IL 已从合法完整 `update079000` 原生恢复至固定白名单 `tsinghua-zt5`，原科学源码保持不变。旧日志未保存的 779 步归档后重做，不能算作零重算迁移。16:13 UTC 的八个 rank 均已到 79,576。
- Endpoint RL 已从合法 `update013000_inner0` 原生恢复至 `tsinghua-zt5`；127 个未提交更新重做。16:12 UTC 的八个 rank 均到 13,886，未更改其独立低 LR 配方。
- F/G 评测的 7 个正常场景边界暂停分片和 8 个已释放输出锁的失联分片已分别审查、重新排队。保持原命令、world、seed、场景缓存和候选身份；失联分片没有伪造正常退出凭证。

原生日志、旧尝试和失败证据保留，未上传 checkpoint、原始数据或大型日志。

时间加权恢复证据：

```text
/mnt/project/qwen-world-postcot-20261025/fresh_loss_IL_20261029/TIME_RESIDUAL/operations/recovery_20261101/
/mnt/project/qwen-world-postcot-20261025/fresh_loss_IL_20261029/TIME_RESIDUAL/jobs/full_IL_attempt006/
```

评测审查证据分别位于 F/G 的 `operations/WITHDRAWN_EVAL_RECOVERY_20261101/`。之后其他 RL 操作负责人继续了有限授权 CPU 调度兼容升级；本记录中的旧 CPU PID 不得作为重启依据，以现行 OWNER 和 `/proc` 为准。交接文件为 `operations/FINITE_RESOURCE_CPU_ADOPTION_20261101/IL_RESOURCE_OPERATOR_HANDOFF.json`。

## 截止守护：已启动，尚未发生

剩余已授权临时节点仅为 `rl-zt2`、`tsinghua-zt2`、`tsinghua-zt6`、`vlawm-zt`、`vlawm-zt4`；不是新增其他主机许可。

| 事件 | UTC | 北京时间 |
|---|---|---|
| 禁止新增并请求原生保存退出 | 11月1日22:15 | 11月2日06:15 |
| 对仍存活的本任务进程做最后退出处置 | 11月1日22:55 | 11月2日06:55 |
| 用户授权截止 | 11月1日23:00 | 11月2日07:00 |

独立守护已处于 ARMED，副本 `runtime_guard_v2.py` 固定，不热改。它只识别具有受信 STOP 路径、已登记模块和进程身份的本任务进程，不停止未知进程。强制退出只允许依赖最后合法 checkpoint，不能标为无损完成。

A IL 原生迁移队列在保存后只考虑未受 B/D 保护的固定 `tsinghua-zt3/zt5`，保持八卡、原生 optimizer/RNG 和预算。若没有足够容量则等待，不能占用 B/D 保护资源。队列已实现并随守护安排，但尚未到时实际迁移，不标记完成。

```text
/mnt/project/qwen-world-four-frameworks-20261029/resources/temporary_deadline_20261101T230000Z/PLAN.json
/mnt/project/qwen-world-four-frameworks-20261029/resources/temporary_deadline_20261101T230000Z/OWNER.json
```

## 已执行验收与恢复入口

本分支仅增加资源准入、行政守护和一次性明确审查入口。准入选择性 6 项、截止守护选择性 3 项 CPU 测试通过；它们不代表未来所有主机均能 SSH 退出或一定有足够固定资源。独立 F/G ops 分支另外验证正常 inner/scene 边界暂停的合法性。

```bash
/usr/bin/python3 -m unittest discover -s tests -p test_c_recovery_owner_admission.py -v
/usr/bin/python3 -m unittest discover -s tests -p test_temporary_deadline_selectivity.py -v
```

下面两个操作入口已实际执行，是一次性操作，遇到已有完成凭证会拒绝重复启动；正常任务不应再次执行。读取它们及对应 LAUNCH/REVIEW 文件可重建审查过程。

```bash
/usr/bin/python3 -u -m tools.il_report.recover_time_il_20261101
/usr/bin/python3 -u -m tools.il_report.rehydrate_withdrawn_evaluations
```

正常报告更新仍由现有 owner 完成，不新建监控进程。已有暂停/终止实验不恢复，原科学源码不热改，结果不重评分。
