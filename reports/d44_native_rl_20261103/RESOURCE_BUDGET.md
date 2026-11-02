# Resources, cost and migration

最新六主机许可截止 **北京时间2026-11-03 08:00 = UTC2026-11-03 00:00**：wm4、rl2、Tsh2、wmzt、Tsh3、wm2（完整SSH名称见运行授权收据）。23:15UTC进入原生保存退出guard；不把临时许可自动扩成永久许可。

常驻B/D IL不迁动。保护节点vla-zt/Tshzt仅在全部GPU无owner时允许新任务独占，绝不与其B/D IL共享。其他明确授权节点可与已确认身份的WorldPerformance任务共享，启动两次核验GPU UUID、owner ancestry、lease与至少41GiB空闲；不能以报告旧快照宣称节点空闲。

wm2 GPU6有约15GiB未解释占用，即使NVML未列PID也不使用。rl2有不可见PID的owner时拒绝共享。不kill未知进程。

## 已测成本

原G16策略重算前后向：source2.324s，all_off1.395s；峰值约25.55GiB。该优化保持输出和完整LoRA梯度一致。

最新真实八卡B64/G16资格（各10updates）：

| 每rank中位数 | D0 | D2（强制从资格起点启用rescue） |
|---|---:|---:|
| optimizer更新，8scene累计 | 10.95s | 13.26s |
| fresh collection主生成，8scene累计 | 19.00s | 19.16s |
| 主候选评分等待 | 1.56s | 1.59s |
| 峰值显存 | 25.69GiB | 31.35GiB |

D2表中主生成时间未包含额外B1/教师挖掘的全部成本；不得直接用它估算D2总费用。短资格样本不是稳态ETA。粗略D0每collection约42s，单epoch约19小时、4epoch约76小时（8卡约600共享GPU-worker-hours量级）；以正式100updates后实测更正。这不是今晚能够完成的预算，必须有跨许可窗口原生恢复与排队。

新invocation账本单列candidate requests、实际生成候选、scorer候选请求、生成wall、评分等待wall、保存wall。失败重算仍计物理成本；有效scene/PG candidate只在全局collection提交后增长。共享worker wall不是独占GPU计算时间，scorer等待不是CPU核时。

deadline CPU guardian收据位于`/mnt/project/d44-native-rl-20261103/operations/resources_until_20261103T000000Z/OWNER.json`。到期任务应原生保存，再由同一campaign恢复到当前仍授权、有容量的节点；没有合法容量则明确WAITING，不能冒用撤销节点。GPU训练进程、CPU job worker与辅助进程须一起核查。
