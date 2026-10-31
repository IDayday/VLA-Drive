# 阶段报告：增量资源审阅

用户 2026-10-31 要求已固化为[长期规则](../../docs/PHASE_RESOURCE_REBALANCING.md)和[机器可读配置](../../configs/operations/phase_resource_review.json)，并写入共享 `/mnt/project/AGENTS.md` 及实际 dispatch policy 的 `phase_resource_review` 元数据。主机授权、科学源码、实验预算均未改变。

## 实际行为

- 每次 IL 阶段报告检查授权与实际 GPU/CPU/RAM/进程/租约；RL 阶段和 RL 报告更新也触发同一中央审阅。相同状态三分钟内可复用最近核查，保留原观测时间。
- 结果加入现有 IL HTML，机器收据同时发布给 RL。优先回填已登记队列，不修改训练数学或增加新实验。
- 可唤醒已有评测 CPU owner；并发上限与原生评测入口不变。不向 learner 发送停止/迁移信号。单任务迁移需要单独的成本/收益与恢复边界证据。

## 本次真实检查与动作

02:05 UTC 对当前允许的 11 个节点各取两次实际快照，全部访问成功。后续评测完成后释放出 vla-zt 的部分卡及 vlawm-zt2 的七张卡；后者 GPU6 的不明占用仍排除，rl-zt2 的不可见 owner 不当作空闲。具体卡数随当前任务改变，以带时间戳的运行收据为准。

完成的评测未重复启动；现有 cadence 正常接续 Huber16k 全量 Navtest。核对其他线的封存权重与间隔后，没有另外符合既定条件、等待分配的新 checkpoint。因此本次 **0 个 native learner 暂停、0 个迁移**，不为了占满 GPU 重复测点或复活筛除任务。下一次队列回填仍由现有 controller 的授权和租约检查决定。

只接替两个 CPU 元数据 owner（报告、评测队列），保留既有 worker 和评测进度，未重启任何 learner。报告原半小时 deadline 保留。真实 SIGUSR1 唤醒资格验证通过：队列 owner 存活，STATE 时间前进，没有向 native worker 发信号。

## 验证与收据

执行 `python -m unittest discover -s tests/il_report`：**32 项通过**，包含 9 项新增资源审阅测试。覆盖已知低利用率进程、不明显存、不可见 owner、租约、CPU/RAM 压力、物理 UUID 去重、无待执行任务、授权变化及不触发自身 HTML 事件。

运行证据根目录真实存在：`/mnt/project/qwen-world-il-research-20261030/resource_reviews_20261031/`，包括 `REGISTRATION.json`、`review_1793412343844495740.json`、`CPU_OWNER_ADOPTION.json`、`CPU_WAKE_VALIDATION.json` 和持续更新的 `LATEST.json`。这些是运行收据，不是新训练结果；没有新增 PDMS 结论。

可执行入口：

```bash
/usr/bin/python3 -m tools.il_report.resource_review --probe-only
/usr/bin/python3 -m tools.il_report.render --once
/usr/bin/python3 -m unittest discover -s tests/il_report
```

在本分支工作目录执行；持续报告 owner 已采用新代码，无需另启动一个 watch。
