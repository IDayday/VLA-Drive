# B / D IL 每 2K Navtest：到期分片恢复与及时发布

UTC 2026-11-03。只修改评测调度、恢复和报告发布；B/D IL、actor、precision、scorer、seed2101、B1 K10/shift5、12,146 场景和 44 个物理 log 均保持。

## 查明的问题

1. D50k 的 7 个分片在资源到期的合法场景边界退出，worker 的 `RESULT.json` 为 EXIT0。原 `InterimRequest.done()` 只看返回码，旧 burst 将它们当作全量分片结束；官方汇总随后正确拒绝缺失人口，留下真实 failure。
2. 恢复不限时授权不会重新激活已进入 `FAILED_REVIEW_REQUIRED` 的评测。该点已有 25 个完整分片及 7 个部分分片，共 10,971 个合法 scene score，缺 1,175 个；无需重跑完成部分。
3. 原失败 `summary.log` 用独占创建，恢复后再次创建会被同名文件阻塞。
4. 本地动态 HTML 在正常更新，但仓库静态快照没有与当前每个完整 B/D 点绑定自动发布。不能用一次性复制报告替代持续更新。

## 修复和边界

- `rankN` 的 EXIT0 还需匹配完整 rank receipt：status COMPLETE、rank/world、identity、该分片 `new+reused` 精确数量。export/parity 的短资格过程不套用全量 rank 语义。
- 自动恢复只接受严格绑定原 spec/command/source/export/seed、原生 `STOPPED_AT_SCENE_BOUNDARY`、temporary/migration、guard/deadline 及 STOP 文件的正常 EXIT0。强制退出、算法/评分异常、任意提前人为 stop 保持人工审核。
- evaluator 和 worker 独立检查同一 guard：evaluator 可以先完成，worker 的轮询标志和时间戳可能稍后写入，也可能尚未写入。以共同 guard/deadline、退出证据和原生身份确认，而不虚构轮询先后关系。
- 旧 RESULT、rank receipt、评分及 failure 证据保留。新尝试复用原 scene score；不伪造 COMPLETE，不回填分数，不改变分母。
- 恢复汇总采用新日志名，原失败日志保留。完整点完成后主动请求原单一 HTML owner 刷新。
- 新发布器只读已封存结果和队列元数据，每 60 秒检查一次；独立 checkout 的新报告分支只提交结果目录。校验完整人口、协议、身份、原生 parity 和 worker closure；不发布部分均值，不轮询训练 journal，不运行模型或 scorer。

## 实际执行

原 cadence CPU 3860297 正常停止后，仅替换这一 CPU owner。D50 原失败 owner 已退出；对 7 个分片完成显式审核后恢复。D52 原 owner 4128439 保持运行；B/D learner 与其 GPU rank 未收到信号。

- D50 新 burst CPU：4141896。
- 新 cadence CPU：4141897，运行不可变 `maintenance/bd_completion_20261103/runtime_v3@8cae1ae0419852edc324010ddd4d455d005218f2`，未来仍每 2K 登记、32 shard、最多 4 个模型。
- 七个真实 `rank*_attempt002` 已实际启动，25 个完整分片没有重新执行。
- scientist 仍为 `/mnt/project/VLA-Drive-qwen-world-matched-il-navtest-20261030@0357fd2948a9dcc60c36b3d3f577f0222ed86ecb`。
- 新 resource worker 与前一已验收版本逐字节一致；没有放宽未知 owner、GPU headroom、主机授权或 B/D IL 保护。

本地审核与部署收据：`/mnt/project/qwen-world-il-research-20261030/maintenance/bd_completion_20261103/`。旧失败在 D50 的 `operator_attempts/4029956/` 中保留，原 `summary.log` 未删除。

## 已执行验收

```bash
env PYTHONPATH=. /mnt/project/flow-safe-improvement-20261013/venv/bin/python -m pytest \
  tests/il_report/test_bd_publication.py \
  tests/il_report/test_native_boundary_return.py \
  tests/il_report/test_burst_recovery.py \
  tests/il_report/test_navtest_cadence.py -q
```

32 项针对性测试通过。覆盖 EXIT0 部分分片、合法到期恢复、两个独立 timer 的先后情况、错误 identity/world/count、强制退出、非资源停止、旧日志保留、封存人口发布门槛与无关 Git 修改拒绝。

## 动态完整分数

最新表持续更新在 [独立 B/D 结果分支](https://github.com/IDayday/VLA-Drive/blob/reports/il-bd-navtest-live-20261103/reports/il_bd_navtest_updates_20261103/README.md)，包含逐点分项与 hash。固定科学观察协议，不把 Navtest 标签回流训练。

发布入口实际为 `python -m tools.il_report.publish_bd_periodic --checkout <独立报告checkout> --state-root <单一发布owner目录> --watch --interval 60`。发布器从固定运行源码执行；报告 checkout 的 HEAD 更新不触碰训练或评测 runtime。

## 恢复完成与发布验收

D50k 七个 `attempt002` 实际完成，官方全量汇总通过，原有 25 个完整分片未重执行。D52k 也由原 owner 完成；两点均完整 12,146 场景、44 个物理 log、invalid=0。

| 架构 | checkpoint | PDMS |
|---|---:|---:|
| B | 60k | 88.1991 |
| B | 62k | 87.7696 |
| B | 64k | 88.1494 |
| D | 46k | 89.4955 |
| D | 48k | 89.2318 |
| D | 50k | 89.2877 |
| D | 52k | 89.2018 |

B 当前已完成最高点仍是 60k，D 为 46k；后续点有波动，不据此调整已登记训练预算。

实际检查时 B 八个 rank 同为 65,651 次更新、D 八个 rank 同为 52,474，日志最大年龄约 1.2 秒；最新封存 checkpoint 分别 65k/52k。下一固定评测点 66k/54k，保持每 2K，不为补报而插入新评价 seed 或改变人口。

原单一 HTML owner 3244218 保持，本地 HTML 于 UTC 02:40 已含两点。新发布 owner 4146505，固定源码 `publisher_runtime_v1@3e33b42164e05dea0b6a6001bf3f22f443fa079b`，独立报告 checkout `/mnt/project/VLA-Drive-il-bd-navtest-live-20261103`。首发 49 个完整 B/D 点、待评测点 0，远端与本地 SHA 同为 `84c7aa5966d7a8e7d1e68f6c5531e43c24a65aac`；该报告分支后续由同一 owner 持续更新。

实际部署与全量汇总证据见 [evidence](evidence/)。不提交 checkpoint、轨迹缓存、图片或密钥。
