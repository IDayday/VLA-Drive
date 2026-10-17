# B筛选后续汇总入口修复：保留正在运行的24个worker

2026-10-17 06:25 UTC，B自己的完整TRAIN筛选已报告55,952/101,592场景（55.08%），24个原worker仍在运行。对后续队列做实际入口检查时，发现`screen_summary`被路由到数值快照225b，但`analyze_screen.py`在稍后的56874b提交才加入。若沿用该路由，完整筛选结束后的CPU汇总会失败。

**这是尚未执行的CPU汇总入口缺失，不是采样、评分或训练错误。** 没有重采样或重训；B的100k SFT、完整DEV基线和所有已完成筛选场景保留。正式B RL仍等待完整清单与八卡验收，尚未启动。

## 复现与最小修复

实际在225b目录执行`python -m scripts.analysis.fm_transfer.analyze_screen --help`，退出1，明确报告`No module named scripts.analysis.fm_transfer.analyze_screen`。相同入口在包含既有汇总器的源码中正常加载。上一轮修复解决了多主机`--ranks`接口，但没有覆盖这个独立的后续入口；原33项测试不能作为全部CLI可用的证据。

新增显式`screen_analysis_source`，将只读汇总路由到独立CPU源码；新增启动前的全部阶段入口存在性检查，不进行静默fallback。文件存在检查只证明路由文件可用，不代替实际CLI、数值或效果验收。

| 阶段 | 执行源码 | 是否改变模型/评分 |
|---|---|---|
| SFT预算摘要 | 225b9af | 否 |
| 三个CPU分片dispatcher | d8e4c6c | 仍显式传`--source=225b`；本次已完成并直接接管 |
| 实际24个筛选worker | 225b9af | 不变，原进程继续 |
| 完整payload核验及困难清单 | 225b9af | 不变 |
| 只读筛选汇总 | d8e4c6c | 无新推理/评分；沿用既有analyze_screen |
| 真实八卡验收、正式RL | 225b9af | 不变 |

新CPU快照完整提交为`d8e4c6c7cdf11eed2a1f1ef8c3b84230b7e46999`。该代码提交只修改endpoint controller和针对性测试，没有修改actor、loss、采样器、优化器、数据清单规则或评分器。所有原不可变快照保留。

## 已执行验收和接管

- 三组针对性测试共**44项通过**，覆盖全部阶段源码映射、缺失后续入口的拒绝、旧错误路由复现、既有stage依赖和分片/汇总合同。
- 实际运行6个不同CLI的`--help`，分别验证SFT摘要、分片dispatcher、清单发布、筛选汇总、八卡验收和正式RL入口。另保留旧缺失入口的预期失败；没有因此启动GPU或评分任务。
- 发布新不可变快照后，对全部8个已登记stage进行实际源码路径/seal预检查。所有阶段命令与旧计划逐项一致。
- 旧v2控制器通过`STOP_AFTER_TICK`退出，实际PID已不存在。没有向筛选进程发送信号。
- 新v3在同一个全局owner lock下显式接管已完成的`sft_summary/screen_0/screen_1/screen_2`。新attempt关联旧intent/status的hash，没有再次执行这4个命令或启动另一套24个worker。
- 24个原worker的PID/start-time/module再次核验仍存活；24份原dispatch与screen manifest的hash不变。已锁定数值源码中64份正式依赖文件与已有正式identity一致。
- 新CPU owner实际运行在d8e4c6c，第二个owner无法获取全局锁；milestone与最终评价两个owner保持原身份。新owner状态为`WAITING_FOR_COMMITTED_FULL_TRAIN_SCREEN`，无failure记录。

本次接管不重置数据进度、RNG或任何模型状态。旧v1失败证据、v2正常停止记录均保留。跟踪v2的只读observer正常发现指定旧owner退出，随后以单独receipt说明这是计划内接管，不将它解释为筛选或训练失败。

## 当前入口和证据

当前唯一B衔接控制目录：

```text
/mnt/project/fm-transfer-20261016/endpoint_handoff_B_v3
```

当前owner PID=430753、start_ticks=1365690818，仅为本次现场记录，未来操作必须重新核验，不能据此盲目启动或停止进程。计划fingerprint为`2e2db9a76a8a721f2f0196d1891e9914c07849595dca69ba869070f00e68d4bd`。

已执行迁移命令如下；脚本是带原owner退出、当前worker存活、原stage/dispatch身份及全局锁检查的**一次性接管**，不是可以重复执行的自动恢复器。当前v3已存在，重复运行必须拒绝。

```bash
cd /mnt/project/fm-transfer-20261016/source/d8e4c6c7cdf11eed2a1f1ef8c3b84230b7e46999
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH="$PWD" /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  /mnt/project/fm-transfer-20261016/resources/migrate_B_handoff_analysis_source_20261017.py
```

脚本副本：[reproduce_B_endpoint_analysis_migration.py](reproduce_B_endpoint_analysis_migration.py)。
证据：[CLI实际检查](B_endpoint_v3_cli_validation.json)、[全部入口映射](B_endpoint_v3_entrypoints.json)、[发布身份](B_endpoint_v3_source_publish.json)、[迁移收据](B_endpoint_v3_migration.json)、[实际进程/锁/hash验收](B_endpoint_v3_live_validation.json)。

此次修复不证明后续八卡验收或RL已经完成；它消除了已复现的未来调度故障。继续固定100k自身SFT→完整TRAIN清单→八卡验收→20k RL，不增加科学预算、seed或方法分支。
