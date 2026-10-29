# 运行入口、恢复与结果归属

## 先取得正确的源码

版本与源码SHA见[代码表](VERSIONS_AND_CODE.md)和[registry.json](registry.json)。当前索引分支继承同预算评测版本1724e45；优化标签训练的专用CLI/config仍在其独立905c5ef分支中。运行某个历史版本应建立新的worktree到对应源码，不能热切换正在运行的worktree。

原始IL科学源码是不可变snapshot，对应Git d54617b3，50文件内容已核对。checkpoint里的source_files/identity是恢复依据，不以最新报告commit替代。原始训练、优化标签训练、scope RL和loss fork拥有独立目录及owner。

## 实际存在并已使用的入口

下表是入口索引，完整已执行argv保留在相应运行资产的job spec／RUNNING／OWNER_LAUNCH中。已有run不要重复派发；重放应使用隔离输出目录和对应身份。

| 用途 | 对应源码中的真实入口 | 已执行证据 |
|---|---|---|
| 完整World IL | `python -m tools.qwen_world_postcot.train_il --config ... --output ...`，8rank torchrun | 原始runs/IL_full_seed42及formal_main_v2 source binding |
| 原始IL精确恢复 | 同入口＋`--resume <完整native checkpoint>` | IL_resume报告；原40k/52,156恢复收据 |
| attention RL | `python -m tools.qwen_world_postcot.train_rl ...` | parent40k_user20261027正式50k及每100步checkpoint |
| memory/time RL | `python -m tools.qwen_world_postcot.train_scope_rl ...` | scope_rl_IL40k_20261027两组正式50k |
| scope owner | `python -m tools.qwen_world_postcot.start_scope_rl --registration ...` | 每组REGISTRATION/ACTIVE_OWNER/真实八卡资格 |
| 63k＋2k坐标续训 | `python -m tools.qwen_world_postcot.coordinate_study_owner --root ...` | coordinate_loss_20261028完整CONTROL/PHYSICAL |
| 三类独立损失续训 | `python -m tools.qwen_world_postcot.loss_designs_owner ...` | loss_designs_20261028各组资格、2k更新及评分 |
| 优化轨迹标签准备 | `python -m tools.qwen_world_postcot.prepare_optimized_targets --output ...` | optimized_labels_IL_20261028/targets及label_identity |
| 优化标签100k owner | `python -m tools.qwen_world_postcot.optimized_il_owner --config configs/qwen_world_postcot/optimized_labels_il.yaml` | 专用905c5ef源码，ROOT/TASK环境必须指向该独立任务 |
| 原生导出 | `python -m tools.qwen_world_postcot.export --checkpoint ... --output ...`；RL另带`--rl-checkpoint ...` | 各批export/COMPLETE/native_export_parity |
| 固定检查点评测 | `python -m tools.qwen_world_postcot.latest_snapshot_owner --root ... --report-dir ...` | 77k/50k、RL中点、IL65—83、优化13/15、原GT10/15各批 |
| 完整结果汇总 | `python -m tools.qwen_world_postcot.summarize --evaluation ... --output ...` | 每模型SUMMARY及pairwise报告 |
| 损失窗口分析 | `python -m tools.qwen_world_postcot.analyze_loss_screen --output ...` | analysis_followup_20261029T0340Z，纯CPU复用已有结果 |

当前原GT10k/15k配对评测已实际执行的owner命令：

```bash
/mnt/project/flow-safe-improvement-20261013/venv/bin/python -u \
  -m tools.qwen_world_postcot.latest_snapshot_owner \
  --root /mnt/project/qwen-world-postcot-20261025/original_vs_optimized_labels_NAVTEST_20261029T0500Z \
  --report-dir reports/qwen_world_original_vs_optimized_labels_20261029
```

本轮唯一owner1524656使用SOURCE1724e45，现已完成退出，完整报告提交7ca648d4已推送。上面是执行记录，不是要求再启动一个owner。优化标签10k/15k结果只读复用。

## 可恢复状态与不可恢复证据

- 完整IL checkpoint保存模型、四readout、Adam、scheduler、全部rank RNG、数据/builder和bank binding；原子库存/hash与COMPLETE一起验证。
- RL保存LoRA及可训练memory、永久IL父模型身份、old chain/old概率、advantage/global std、inner进度、optimizer、RNG和数据游标。恢复inner2后继续inner3/4，不能换新链沿用旧概率。
- collection尚未提交的失败证据不能作为合法latest。先按阶段核查READY/已提交rollout/已提交inner，不能因目录存在就恢复。
- 部分评测只恢复缺失scene/rank，必须保持原模型、源文件hash、评分、scene-bound seed和输入身份。invalid不能从分母丢掉。

**已发生的回滚如实保留**：原始IL早期quota故障曾丢弃未保存更新；优化标签授权收据不一致导致12791回滚到完整12000，791步重新计算。这些失败的物理成本没有当作额外有效训练样本，也没有称作无损。当前训练正常与这些历史证据同时成立。

## 当前Huber补测的有限恢复

原始Huber rank0/1，以及首轮恢复rank2/3/6已完成，共7592场景。rank4/5又在模型子进程启动前遭遇共享GPU锁冲突，rank7没有派发；首轮owner已退出。失败文件、已完成scene和费用均保留。

[本次操作脚本](operations/complete_huber_admission_recovery.py)只允许这三个已核实分片，使用旧3e344df源码和原评分身份，最多3次**仅限子进程未启动的准入重试**；执行后的模型/科学失败立即停止，不当作资源竞争重试。它不能启动训练，也不改变actor或评分计算。

本次已执行的只读资格检查：

```bash
PYTHONPATH=. /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  reports/qwen_world_experiment_registry_20261029/operations/complete_huber_admission_recovery.py \
  --check-only
```

返回PASS，确认旧owner退出、既有五分片完成、两个失败未创建model child、最后一分片未派发和旧源码干净。正式执行去掉`--check-only`，由唯一detached owner和独立attempt002收据管理，完成后全量验证并推送原loss报告分支。

RL中点曾有同类准入冲突；缺失rank补完后，发布器因错误strip掉porcelain状态前导空格而拒绝修改的README。所有模型/评分已完成，人工按原HEAD与报告路径白名单复核后完成**仅报告发布**93489a2。没有重跑模型或改分数。该故障与训练算法效果分开。

## 当前资源与成本口径

本次只允许固定五台：training-vla-zt、training-vlawm-tsinghua-zt、zt3、zt4、zt5。vla-zt3不是固定清华zt3。旧报告中的临时rl-zt2／清华zt2／清华zt6等授权已撤回，不用于新任务、重试、下一阶段或pressure。

实际权限读取`/mnt/project/server_dispatch_policy.json`及匹配的不可变task grant。只读历史报告不能授权GPU。共享GPU按两次实际UUID/显存、可见owner、CPU内存及lease准入，不驱逐未知或其他项目进程。

费用至少分开：逻辑更新/场景/新候选、实际新推理与复用、失败/回滚重算、scorer调用、模型加载/前向/评分等待、checkpoint保存、job wall与累计worker wall。多个worker相加不是端到端wall；共享worker时间不是独占GPU计算时间。历史缺失的耗时字段不能倒算成精确数。

完整IL checkpoint约39.92GiB逻辑库存。硬链接锁定不会复制同一inode的权重字节；export和真实新模型输出仍有实际存储与计算费用。该索引不上传模型、原始数据、巨大cache或私有场景图。

## 本次核验范围

本次执行了文档JSON/CSV与结果身份校验、Git源码路径检查、原始IL50文件的snapshot/Git hash对照、前15k双组八rank匹配核验、源码归档新增blob审查，以及Huber恢复只读资格检查。现有source oracle、LoRA、真实八卡inner恢复等结果从已完成报告引用，没有把它们伪装成本次重新跑过的测试。

核心算法实现、正常训练数值、异常恢复、评价完整性和PDMS效果分别判断；不会因为测试PASS就宣称提分，也不会因为一个纯发布错误就否定已完成全量结果。
