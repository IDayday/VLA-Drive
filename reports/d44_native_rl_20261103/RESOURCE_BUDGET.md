# Resources, cost and migration

最新六主机许可截止 **北京时间2026-11-03 08:00 = UTC2026-11-03 00:00**：wm4、rl2、Tsh2、wmzt、Tsh3、wm2（完整SSH名称见运行授权收据）。23:15UTC进入原生保存退出guard；不把临时许可自动扩成永久许可。

常驻B/D IL不迁动。保护节点vla-zt/Tshzt仅在全部GPU无owner时允许新任务独占，绝不与其B/D IL共享。其他明确授权节点可与已确认身份的WorldPerformance任务共享，启动两次核验GPU UUID、owner ancestry、lease与至少41,000MiB空闲；不能以报告旧快照宣称节点空闲。

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


## 2026-11-02 20:39 UTC运行边界

原节点在100步暂停后被其他已授权任务使用，旧调度还错误优先尝试了不可连接的替代节点；保留全部pre-child拒绝/SSH失败记录。新控制器优先上一健康节点、区分子进程未启动的失败、使用同一native检查点恢复。没有改P0的955289f科学快照，没有刷新old chain或Adam。

标准准入保留最多两个已确认CUDA owner。另提供**仅D0既有B64/G16/r16/all_off范围的恢复**可显式使用第三个独立lease：实际profile峰值25.69GiB，要求再次核验至少40GiB空闲、全部owner可见且来源核实；未知owner、第四个context、D6或扩大scope不能使用此特例。它是已授权共享GPU的运维适配，不改变算法、参数精度或batch。实际启动收据记录最终取得的slot，不由配置假定。

D1/FPO/AWM/D4的B32真实资格、D4/D5部署对照、完整packing与D6阶段恢复由独立有限资格队列管理。资源拒绝只计为准入失败，不冒充GPU数学失败或已运行样本。队列每10分钟检查实际容量，启动时仍由worker复核；有限节点到guard后禁止新派发。没有可用容量就等待，不挤占B/D IL或其他项目。

当前训练节点：D0=training-vlawm-zt4，D2=training-vlawm-tsinghua-zt2，D5=training-vlawm-tsinghua-zt3；均属于本次到期六节点，不能据此延长许可。永久队列仍受真实容量约束；vla-zt/Tshzt的B/D IL保持保护，wm5连接失败不能作为可用GPU容量。


资格队列额外共享规则：仅在既有cohost均为已确认、封存export的只读B1评测、至少48GiB实际空闲时，允许已测范围的D1/FPO/AWM/D4两卡B32/G16短资格（最多16updates）或D4/D5部署对照占用第三槽。D4既有实测峰值28.85GiB；新scope/G32/FiLM扩大长训不继承此许可。所有原子lease与双重UUID/进程检查保留；新GPU任务不向B/D IL节点派发。qualification exit0还必须有PASS证据/完整更新及冻结hash；到期短停按合法checkpoint继续剩余更新。源码/参数精度/主训练budget不受此运维规则影响。
