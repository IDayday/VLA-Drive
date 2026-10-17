# B组100k SFT完成与筛选衔接修复

2026-10-17 05:50 UTC，B_SOURCE_ACTION_DIT完成固定100,000次SFT更新。八rank完成记录、实际进程退出、原生检查点、824份Adam状态和scheduler均已核验。**这是监督训练终点，B的正式RL还没有开始，不能作为RL收益或Navtest超过92的证据。** 终点双网格DEV与固定TRAIN诊断由现有里程碑控制器评价；完整TRAIN筛选在修复一个CPU派发接口错误后继续。

## 完整监督训练与原生状态

| 项目 | 实测 |
|---|---:|
| 全局optimizer updates | 100,000 |
| 场景曝光 | 3,200,000 |
| 不同TRAIN场景 | 101,592 |
| FM repeat | 8 |
| FM插值训练样本数 | 25,600,000 |
| SFT期间RL候选/奖励调用 | 0 |
| rank0训练循环wall | 43,812.049秒，约12.170小时 |
| 全程平均updates/s | 2.282 |
| 八worker累计wall | 350,495.876秒 |
| 派发到最后训练边界 | 43,827.465秒 |

完整八份更新日志逐行核对连续计数、rank归属、游标及实际scene人口，未发现DEV/Navtest混入本次SFT。FM repeat不是独立场景，八rank也不是八倍全局optimizer更新。累计worker时间包含重叠、共享设备驻留，不能解释为独占GPU计算时间。逐次保存、退出及模型加载未全部单独计时；未覆盖时间不补造归属。详见[完整预算](B_SFT_complete_budget.json)。

[原生终点核验](B_SFT100k_native_completion.json)重新计算全部11个payload的hash，并实际读取：824份Adam状态均step=100000，moment为FP32；scheduler.last_epoch=100000、末尾LR=1e-5；824个模型参数的键、形状、dtype与登记一致；八份保存状态均READY、cursor=3200000，Python/NumPy/Torch/CUDA RNG完整。实际trainer及该run进程已退出。

这不是另一次恢复重放；此前八卡恢复资格证据仍独立保留。新增[只读核验脚本](reproduce_B_SFT100k_completion.py)在真实仍运行的B上拒绝完成判定，在已退出的A上正确通过进程检查，然后在B终点实际运行通过，见[进程检查证据](B_SFT100k_process_probe_qualification.json)。没有额外推理、评分或optimizer更新。

- SFT身份：`63b2da609f3ab3b225f056a5399050fc217989d25b937876a5a5051a29575af7`。
- 原生文件清单SHA256：`a1c16de21ad305d2817f6ba0386005134f3362f1b67e84bdcc04d60dbfca45f3`。
- 原生model.pt：`4e71b1fcf963d38456174de1f8d17762a9a4ccf1597c22b7cec262026d9201b7`。
- 原生optimizer.pt：`487b4d106f81443c4ecb52682f1be74a212feaa1d343e36bfcef03e10c14b60d`。
- 部署导出权重：`548e122b7c1fad10cc8dbf0e2dbf5bc97efe7a242e0f9d4684b18b3f65fff91a`；不同序列化资产的hash不混用。

## 实际失败及修复范围

旧`endpoint_handoff_B_v1`正确等到100k完成，SFT摘要阶段成功。但screen_0在05:50:45 UTC退出，错误为`unrecognized arguments: --ranks`，退出码2。原因是多主机分区参数在`b80cdad`才加入CPU dispatcher，衔接控制器却从较早的`225b9af`执行该dispatcher。原27项测试验证了依赖/身份与mock执行，但没有覆盖这个实际版本组合，不能将先前测试通过视为完整运行验证。

失败发生在argparse阶段：screen输出目录不存在、没有任何候选或screen worker启动。已核验旧owner、stage与child全部退出。SFT、权重、评分记录和正在运行的终点评价均不受影响。失败日志与原controller目录原样保留。

`8dbd2bb2c82026d3a8a0b4b96c9b0ba211248f81`只修复CPU阶段的执行来源：screen_0/1/2从有`--ranks`接口的新快照执行派发；其显式`--source`、新worker的PYTHONPATH、模型、评分以及后续真实RL入口仍锁定`225b9af`。其他阶段继续使用原worker快照。计划新增并绑定`screen_dispatch_source`，阶段状态分别记录CPU执行source和numerical worker source；没有热改任何immutable目录。

新增回归覆盖三个screen阶段与endpoint gate/formal RL的实际cwd、PYTHONPATH及source区别，连同已有分区/准入测试共**33 passed**。新快照的真实dispatcher CLI已核对支持`--ranks`，修复后第一批八个真实screen worker启动且仍加载225b源码。测试边界与证据见[验证记录](B_endpoint_dispatch_source_validation.json)。

这是显式的新控制器attempt：`endpoint_handoff_B_v2`。原已完成SFT摘要经command、plan、身份及内容hash核对后复用，未重新执行；失败screen保留在v1，没有把原FAILED改成成功。新v2再做当时授权、显存与进程所有者检查后派发尚未启动的分片。[迁移receipt](B_endpoint_dispatch_source_migration.json)记录父计划、失败证据、复用来源及源代码变化。首批成功派发约在05:55:43 UTC，故障造成约五分钟衔接延迟，不能从实际执行成本中删掉。

## 后续与已执行命令

固定100k的完整DEV1696场景与TRAIN诊断512场景、native/shift5两种网格并行评价。完整TRAIN筛选仍是101,592个scene各一个native ODE样本，24个global shard，seed2101；生成自己的固定困难清单后，继续自己的真实八卡RL资格测试，再从同一100k SFT新初始化20k R_SOURCE。资格权重不用于正式初始化。最终Navtest仍等待两组RL终点及完整阶段证据。

[实际衔接快照](B_SFT100k_handoff_execution.json)已核验24个screen worker全部存活：tsinghua-zt6承接rank0–7、zt4承接8–15、zt承接16–23，实际cwd及PYTHONPATH均为225b。快照报告已提交4160个筛选scene，不能当完整筛选结果。DEV当时1112/1696、TRAIN诊断512/512；完整评分与分析仍按原入口完成，不报告部分DEV均分。rl-zt2的不可见GPU占用被跳过；没有停止真实/未知任务或新增科学预算。

以下命令已执行；已有输出不可覆盖，再次只读核算应使用新输出身份。原生核验读取自身可信检查点，CUDA不可见。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  reports/ddp_simwam_fm_transfer_20261016/reproduce_B_SFT100k_completion.py \
  --output /mnt/project/fm-transfer-20261016/qualification/B_SFT100k_native_audit_v1/completion.json

# 在immutable d58485b分析快照执行，训练源码未改变：
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.fm_transfer.analyze_sft_training \
  --run /mnt/project/fm-transfer-20261016/runs/B_SOURCE_ACTION_DIT_SFT_seed42_v1 \
  --output /mnt/project/fm-transfer-20261016/analysis/B_SOURCE_ACTION_DIT_SFT_complete_budget.json
```

实际一次性迁移脚本保存在`/mnt/project/fm-transfer-20261016/resources/migrate_B_handoff_dispatch_source_20261017.py`，hash见迁移receipt；它拒绝已有v2、未退出owner和任何可能存在的筛选输出，不是通用自动retry入口。当前v2已启动，不可再跑一遍。源码修复与预算证据不会被当作规划提分结论。
