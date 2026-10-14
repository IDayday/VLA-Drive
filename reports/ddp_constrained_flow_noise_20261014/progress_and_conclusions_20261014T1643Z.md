# 2026-10-14 16:43 UTC：强化学习结果与运行状态

两条正式训练均未完成。目前已经出现原单候选 ODE 的平均分提升，但尚未得到满足安全选择条件的检查点，也没有证明可学习噪声优于固定 SDE。

本次检查同时核对远端真实进程、八个 rank 的日志、失败记录和检查点；不再用最后一条训练日志推断进程仍然存活。此前 README 在 16:17 写“两组均继续运行”不准确：C_SDE 已于 16:09 退出。旧日志和失败证据均保留。

| 任务 | 已完成工作 | 当前状态 |
| --- | --- | --- |
| C_SDE | 1,850 次更新，对应 23,184 次原始场景曝光；随后失败的 collection 已采样到 23,208 | 原训练进程退出；隔离的八卡恢复正在进行 |
| N_FLOW_NOISE | 2,294 次更新，28,638 次原始场景曝光，占两遍预算 203,184 的 14.095% | 八个 rank 均正常推进，尚未完成 |
| 冻结 SFT 全 TRAIN G16 统计 | 28,601 / 101,592 场景，457,616 条候选 | 32 个实际 worker 全部存活；未完成全量 |

[远端进程核查](runtime_process_check_20261014T1642Z.json)保存本次快照。
累计曝光比例使用两遍原始人口预算，不把被动态筛选接受的场景数当总预算，也不把恢复重放计作新训练曝光。

## 原单候选 ODE：已有可分析结果

完整 RL_DEV 为 1,696 场景、16 logs。以下均使用原始观测入口、原 10 步 ODE、单候选 seed2101、canonical v1.1，部署时没有 noise head、搜索或评分器选择。训练 seed42；分数转成 0—100。

| 模型 | PDMS | EP | DAC 违规场景 | TTC 违规场景 | 零分场景 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原 F-SFT | 93.6192 | 85.8065 | 13 | 1 | 13 |
| C_SDE，update1,024 | 94.4840 | 87.4573 | 5 | 4 | 5 |
| N_FLOW_NOISE，update1,024 | 94.4822 | 87.9720 | 7 | 10 | 7 |

两个 update1,024 观察点分别消耗 12,760 / 12,744 原始场景，差 16；这是接近匹配的早期观察，不冒充严格相同曝光。相对 SFT，C/N 的 PDMS 分别提高 0.8647 / 0.8630 分。按 log 聚类的配对 95% 区间分别为 [0.5819, 1.0700] / [0.6396, 1.0957]。N−C 为 −0.0017，区间 [−0.1577, 0.2086]，不能认定可学习噪声带来了额外部署收益。完整证据见[匹配早期分析](interim_update1024_analysis.md)。

N 的第一个正式观察点 raw25,398 / update2,030，PDMS 为 **94.5460**，比 SFT 高 **0.9268**，区间 [0.5060, 1.1726]。DAC 违规 7，TTC 违规 4，零分 7；NC 全部通过。相较早期 N，TTC 从 10 降到 4，但仍高于 SFT 的 1。固定 SFT 安全人口新增联合违规 5、恢复 8。该点不满足安全选择规则，尚未触发预登记的严重退化暂停规则。[完整正式 DEV 结果](N_FLOW_NOISE_DEV_raw025398.json)。

C 因中断尚无相同 raw25,398 观察点，不能拿 N 的后期点与 C 的早期点作方法排名。尚无本轮锁定终点的 Navtest 结论；DEV 的 SFT 本身已高于 92，不能把 DEV 超过 92 宣称为 RL 达成 Navtest 目标。置信区间不包含训练或推理 seed 的不确定性。

## 有效探索与更新效率：尚未解决

N 的固定 128 场景 TRAIN 监测集使用正常 G16 随机链，共 2,048 候选；监测数据不进入 PG、dual、校准或训练游标。初始与 raw25,398 使用相同人口及根噪声协议。

| 随机策略监测量 | 初始化 | raw25,398 |
| --- | ---: | ---: |
| 候选 PDMS 均值 | 91.8595 | 93.2412 |
| NC/DAC/TTC 联合通过候选比例 | 97.6074% | 98.2422% |
| PDMS 完全等分组 | 37 / 128 | 40 / 128 |
| 最终优势全零组 | 23 / 128 | 28 / 128 |
| 官方安全事件成本有组内差异 | 12 / 128 | 8 / 128 |
| 存在安全且高于组均值 0.001 的候选 | 90 / 128 | 85 / 128 |
| Pairwise ADE | 0.1800 m | 0.1750 m |

随机策略的平均成绩和安全通过率提高，但零优势比例没有下降，也没有发现更多上述组内安全机会。lambda 同时变化，最终优势全零组的变化不能单独归因于探索噪声。sigma 上下边界饱和比例均为 0；真实噪声参数和原动作头梯度、更新已通过验收，不能把现象解释成 noise head 未训练。

该监测集与 DEV 人口不同，不能用其随机均值减去 DEV ODE 均值推断 sampler gap。候选最佳值也不是部署成绩。[初始与当前完整监测记录](N_FLOW_NOISE_monitor_raw025398_comparison.json)。

前 512 个真实 behavior collections 的分析显示，两组只有约 64% 原始组进入 PG，约 25% 最终优势全零，约 10% 为补样后未使用的有效余量。部分等分组仍用于未筛选成本统计和 reference 保持，但没有产生该组的 PG 更新。[原始效率证据](interim_update1024_signal_efficiency.json)。

## 全训练集统计继续

当前冻结 SFT＋C_SDE 的已完成 28,601 场景中：全 1 分 9,728 组，全 0 分 12 组，其他等分 13 组，非等分 18,848 组。最终优势全零 6,520 组，约 22.8%；此处固定初始 lambda，不能与训练中变化的 lambda 混用。

等分组中的满分比例约 99.74%，不表示整个训练集 99.74% 已满分。已观察人口中全满分约 34.0%，全零仅约 0.042%。目前算力利用问题主要来自满分饱和及 progress-only 排序，不是大量全零组。全量尚未完成，不把这些比例称为全 TRAIN 最终统计。[当前增量汇总](train_population_progress_20261014T1642Z.json)；更早的稀有等分原因逐条核查见[17,455 场景分析](train_population_snapshot_20261014T1615Z.md)。

三个临时主机仍受 23:07:18 UTC 的硬截止约束；22:37 停止新任务接纳，22:47 开始完成当前原子记录后退出，兜底清理只针对本任务登记进程。保存已完成结果，剩余人口可在永久授权主机继续，不延长临时授权。

## C_SDE 数值故障和恢复

rank3 在第 1,851 次更新前触发 `Fresh behavior/current ratio exceeded qualified layout tolerance`，阈值为 1e−4；整个进程组随后退出。失败状态包含 collection925 的真实链、成本和游标，但不包含 update1,850 的完整 actor/Adam，不能将这个失败文件当成可恢复 checkpoint。

最近完整原生检查点是 update1,024。当前独立目录的恢复程序加载该检查点，重放已经保存且身份一致的历史 behavior collections，沿用原 trainer、Adam、RNG、dual 和数据顺序。逐更新、逐 rank 比较 PG、KL、replay、总损失、ratio、clip、梯度范数、LR、lambda 和成本更新等原始记录；定期重新生成同噪声链并检查 chain/old mean/std/logprob 逐位相等。没有新增评分，没有将新候选配上旧概率。

截至本快照，八个 rank 均验证到 update1,116，即重放 92 / 826 次待恢复更新；第一组链逐位对照通过。恢复目标严格限制在 update1,850，到达后原生保存并停止。记录一致性不是对已丢失全部中间权重的逐位证明；完成前不宣称已经恢复，也不自动续跑故障批次。

只读探针已在 update1,024 完整模型上执行：同一保存状态下 flat 与逐转移布局的均值最多相差 3.34e−6，对应 mean-ratio 差达 8.99e−5；新生成链逐转移重算 ratio 误差为 0。这支持小方差相关 Gaussian 对 batch 布局舍入较敏感的假设，但探针模型不是失败时 update1,850，尚不足以断言根因。下一步在恢复的真实失败前状态验证并修复，不放宽阈值，不改 clip、LR 或旧链概率。[探针结果](C_SDE_checkpoint1024_layout_probe.json)。

已实际执行的入口：

```bash
CFN_PYTHON=/mnt/project/flow-safe-improvement-20261013/venv/bin/python
"$CFN_PYTHON" -m scripts.analysis.constrained_flow_noise.diagnose_probability_layout \
  --checkpoint /mnt/project/constrained-flow-noise-20261014/formal_C_SDE_seed42_v1/checkpoints/update001024_raw012760 \
  --failure /mnt/project/constrained-flow-noise-20261014/formal_C_SDE_seed42_v1/failures/trainer_rank3.pt \
  --output /mnt/project/constrained-flow-noise-20261014/C_SDE_ratio_failure_20261014T1609Z/checkpoint1024_layout_probe.json
"$CFN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m scripts.analysis.constrained_flow_noise.recover_saved_training \
  --source-run /mnt/project/constrained-flow-noise-20261014/formal_C_SDE_seed42_v1 \
  --checkpoint /mnt/project/constrained-flow-noise-20261014/formal_C_SDE_seed42_v1/checkpoints/update001024_raw012760 \
  --output /mnt/project/constrained-flow-noise-20261014/C_SDE_ratio_failure_20261014T1609Z/replay_from_update1024 \
  --target-update 1850
```

第二条命令是已运行任务的记录，不能在活动目录重复启动。完整源及参数在本地 `launch.json` 留存。原正式训练的 83 个冻结源文件未修改；新增两个独立诊断/恢复入口，旧失败、权重及 behavior 档案保留。

命令使用上述 Python3.10 环境；本机默认 Python3.9 的一次 `--help` 检查在导入旧类型注解时失败，切回已验证环境后通过，未触发模型训练或评分。

## 当前结论与继续执行

1. 已有真实 ODE 平均收益，不能说 RL 完全无效；但收益仍主要伴随 EP 提高，安全选择条件未满足。
2. 可学习噪声的额外价值尚未得到匹配证据；零优势与有用组不足仍是主要效率瓶颈。
3. 当前安全约束未保证部署 TTC 保持。有限 TRAIN 随机策略成本与 DEV ODE 的结果不能相互替代。
4. 先完成 C 的数值故障恢复及必要修复，同时继续 N 原定训练、完整 TRAIN 统计和固定 DEV 观察。没有增加算法、seed 或更换指标。
