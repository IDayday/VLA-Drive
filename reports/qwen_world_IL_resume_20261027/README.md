# World IL：从40k完整恢复，继续原100k预算

用户2026-10-27明确要求“world IL 启动”。仅继续现有World IL；原始Qwen-Flow与三帧DINO
仍保持用户停止/暂停状态，已有三个World IL40k-parent RL任务保持原父模型和训练计划。

**已实际恢复训练。** 八rank均完成40,001后的联合IL更新，原生optimizer global step连续、
无跳步；恢复的首个batch、候选库和学习率与原定40k之后的数据/调度一致。
实际进程为rl-zt2八个GPU rank，不是仅启动一个等待中的控制器。
五项损失及梯度范数有限，恢复验收未创建独立资格权重或额外训练预算。
15:40 UTC八rank共同完成40,391更新，恢复后新增391更新；最新封存恢复点仍为40k，
下一个正常保存点41k。40k候选刷新导出已完成，调度状态为等待空闲GPU；
训练继续使用已锁定的合法候选库，不声称新的40k候选库已经构建完成。

## 训练合同保持

| 字段 | 恢复后设置 |
|---|---|
| 原生完整恢复点 | `runs/IL_full_seed42/checkpoints/update040000` |
| 恢复update / scene cursor | 40,000 / 1,280,000 |
| 总预算 / 剩余更新 | 原100,000 / 60,000；没有重开100k新预算 |
| 执行 | rl-zt2，8×A800，micro4、accum1、global scene batch32 |
| 数据与随机种子 | 原World TRAIN、原teacher/candidate合同、seed42 |
| 损失 | clean-x L1、当前DINO、未来、对象、post五项，权重10/1/0.5/0.1/0.2 |
| 学习范围 | 原联合IL阶段，Qwen、memory、actor与辅助头；不是只训练LoRA |
| 学习率 | 原三组LR和100k cosine时间线，继承完整Adam及scheduler |
| 数值 | 原ZeRO-2、FP32参数/optimizer状态、BF16前向及FP32损失 |
| 候选刷新 | 原每10k产出、每5k边界接纳；异步生成，不以旧30k库替代全部后续更新 |

实际训练仍在不可变`/mnt/project/qwen-world-postcot-20261025/snapshots/formal_main_v2`运行。
恢复前将其中全部训练source identity、resolved配置与40k检查点逐项比对，全部相同。
本分支只添加IL-only调度与日志回滚处理，不修改该快照、模型、优化器或训练损失。

## 完整恢复与存储验收

40k完整文件库存42,864,439,332 bytes，全部重新校验hash；原生完整模型、八个ZeRO optimizer
分片、八份rank RNG和builder状态均存在。40k identity仍为
`d14f1f4de212452ac479e65eb85602a6a0ea4b7f195371f2d0a77a9107f0a805`。
检查点的`qwen_optimizer_initialized=true`保留，不再次初始化Qwen Adam。

同一共享存储下实际完成64MiB写入、fsync、原子rename；可用空间约5TiB。
派发前核实rl-zt2实际八卡空闲，原生worker在UUID锁内再次检查，未抢占其它任务。
这证明启动时能写入；后续恢复点仍按原1000步保存、保留两个非里程碑滚动点，原子发布合同不变。

## 原故障与进度归属

原训练因配额错误在约40,031更新处退出，最近完整恢复点是40,000。
八个rank的40k后日志全部以原始字节保留，实际回滚31次未封存更新，不作为额外有效进度。
可观测的废弃计算成本至少1642.497个累计worker秒；这是各rank时间之和，不是独占GPU时间或总wall。

rank4在废弃的40,021记录内存在NUL损坏，保存边界以前的40,000行全部完整。
准备过程最初在该行拒绝继续；失败证据保留。修正只允许在**已验证保存边界之后**归档损坏字节，
不会跳过边界以前的损坏或制造缺失数据。前四rank已完成的准备在精确验证其当前文件等于归档合法前缀后复用。
所有rank现有有效journal只保留1至40,000的原字节，原完整journal、failure和旧progress单独归档。
恢复后从40,001继续追加，避免重复update、损失旧成本或把失败重算算作新增有效数据。

新增11项定向CPU测试通过：6项覆盖精确前缀保留、原始尾部证据不受新日志追加影响、非法保存前缀拒绝、
边界之后的损坏字节保留、部分准备的精确幂等重用、错误历史归档拒绝。
另外5例检查CPU控制器重新接管现有worker时的host/source/command/GPU绑定，拒绝错配。
没有重跑与本次无关的模型审计或训练消融。

原40k导出还有一个配额失败遗留的`IL_update040000.building`目录，导致第一次导出重试
在建立staging时拒绝；这没有影响正在运行的IL。已将原残留目录原子改名归档，所有字节保留。
只更换CPU调度owner，由新owner严格接管同一个`formal_IL_attempt002`原生worker，
没有停止八个训练rank、没有再次加载40k、没有回退新产生的更新。原导出/刷新代码继续执行。
清理阻塞后的第三次原生40k导出已实际退出0并完成发布；旧失败目录和两个失败收据均保留。

## 调度边界和实际命令

新IL-only owner复用原campaign的dispatch、资源门禁、候选刷新和DEV诊断函数，
保留原registry与固定bank boundary；它不调用原完整campaign的选parent、RL启动或多seed终测路径。
已有三组RL不会因IL继续训练而更换reference或父模型。
原campaign owner锁由新owner独占，避免两个控制器重复派发。

实际训练argv为原`formal_IL_attempt001`的完整命令，仅增加：

```text
--resume /mnt/project/qwen-world-postcot-20261025/runs/IL_full_seed42/checkpoints/update040000
```

实际CPU owner命令：

```bash
CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=/mnt/project/qwen-world-postcot-20261025/snapshots/formal_main_v2 \
/mnt/project/flow-safe-improvement-20261013/venv/bin/python -u \
  /mnt/project/qwen-world-postcot-20261025/il_resume_user20261027T1507Z/controller_attempt002/operator/resume_il_owner.py \
  --registration /mnt/project/qwen-world-postcot-20261025/il_resume_user20261027T1507Z/controller_attempt002/REGISTRATION.json
```

命令已执行。读取`ACTIVE_IL_OWNER.json`核实当前owner，不重复执行仍在运行的启动命令。
启动、原生恢复与实际更新的验收快照见 [resume_receipt.json](resume_receipt.json)。
本次启动不是新Navtest结果；World40k已测87.972382不能贴到后续权重上。
