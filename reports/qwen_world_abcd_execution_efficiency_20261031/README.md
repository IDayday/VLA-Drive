# ABCD IL执行提速、严格原生恢复与Navtest队列核查

证据时间：2026-10-31，13:16–15:33 UTC。实验根目录为实际存在的 `/mnt/project/qwen-world-four-frameworks-20261029`。

四组已完成等价性验收和逐组原生恢复，继续原100k IL计划。没有改变结构、监督项、数据、随机序列、参数/梯度/优化器精度、global batch或预算。四组尚未完成完整IL/RL主实验；本报告的PASS只针对执行等价性和切换。

## 1. 实际开销与架构

| 组 | 实际计算 | 原生每rank micro/accum | 主要重复工作 |
|---|---|---|---|
| A | Qwen → C160；20×1024独立actor；20×384训练期world/R | 2 / 2 | Qwen、actor、world激活checkpoint引入的反向重算；同步CPU数据准备 |
| B | Qwen → C160；20×1024共享P32/A8/R16 | 2 / 2 | P、main A、精确候选A replay、R多个短流各自重算 |
| C | Qwen → C160；20×384 world；20×1024 actor与分组K/V桥 | 2 / 2 | world/actor激活重算、两路归一化attention、同步CPU数据准备 |
| D | Qwen原生28×2048语言层处理C/P/A/R；连续动作接口 | 1 / 4 | 完整语言层的前缀与短动作流反复执行；更高accum次数 |

ABC actor为16头、FFN4096；小world为6头、FFN1536。D复用大型语言主干，不能因为没有独立500M actor就认为动作计算小。包括训练readout的总参数分别为A 2,710,801,843、B 2,661,686,323、C 2,713,958,855、D 2,220,489,267；参数总量不是每次更新FLOPs。

在D每rank的一个逻辑update中，需要四份当前C、P、main A，以及一份已打包的候选A/R后置计算，均经过28层。ABC Qwen只有两份micro前向，后续计算使用更小的独立决策主干。D的这部分结构性开销仍存在。

以下均已在原实现存在，不能当成本轮新优化：SelectedQwenEncoder通过hooks读取8/16/28层、`output_hidden_states=False`；当前编码在主动作与候选replay间共享；D的候选A/R已打包并隔离；teacher全部离线；候选直接clean replay。IL每次更新不是跑一条10步ODE链，也不在线调用PDMS仿真。

原生八rank对齐64更新的窗口显示，CPU数据等待中位数约1.19/1.44/1.22/1.33秒。原始峰值约22GiB，仍对大量短序列乃至完整Qwen使用激活checkpoint，额外的反向重算是可去掉的工程成本。详细原始窗口见[旧窗口](LIVE64.json)。

## 2. 实际实施的优化

1. 保留完整计算图，减少激活checkpoint的反向重新前向。先验证短actor/world/R流，再验证包含Qwen的`all_off`配置，最终正式采用后者。计算路径和精度不变，用额外显存换时间。
2. 深度2的有界CPU预取。仅准备图像、teacher采样位置、GT、固定bank/evidence等数据；不缓存可训练Qwen/P/W表示。按原update/scene/position确定的RNG准备，消费后才推进builder/cursor，丢弃未消费准备不会制造训练曝光。
3. 每个worker独立打开SQLite只读连接。不能跨线程复用原连接。将已有不可变数据/processor/teacher/track读取器转交唯一预取worker，避免每次启动重新构建大清单；正常冷启动约2–3秒，一次冷I/O为12.76秒，原重复构建曾需34–45秒。
4. 原生恢复先按checkpoint中的Qwen配置构造结构，再严格恢复完整训练态，避免先加载最终会被覆盖的通用Qwen权重。不是改初始化：正式恢复的模型、FP32 Adam、scheduler、全部RNG和数据游标都来自原完整checkpoint。
5. 调度控制与训练源码分别冻结并登记迁移收据；逐组保存和接续，不做四组全局暂停。只对核实的旧CPU controller发信号，模型通过既有`STOP_REQUESTED`协议在完整边界自行保存退出。

保留ZeRO2、原micro/accum、FP32参数/梯度/Adam状态、BF16前向、FP32动作/损失计算。没有改batch布局、attention backend、token数量、七项loss或LR。旧DDP/ZeRO比较包含初始化/热身，未证明稳定吞吐优越，因此没有为了提速改训练后端。

## 3. 真实模型配对计算测试

同一已有训练权重、真实TRAIN输入、七项监督、原micro/accum和损失分母。每种模式先热身，随后交替4次计时。以下是单rank完整F/B切片，排除数据、optimizer和collectives，不能代替八卡端到端吞吐。

| 组 | 原F/B秒 | 保留全部图F/B秒 | 配对吞吐比 | 保留图峰值GiB | 全参数梯度 |
|---|---:|---:|---:|---:|---|
| A | 2.9621 | 2.1078 | 1.405× | 36.42 | 完全相同 |
| B | 2.4459 | 1.6725 | 1.462× | 36.97 | 完全相同 |
| C | 2.6895 | 1.8718 | 1.437× | 36.70 | 完全相同 |
| D | 4.7429 | 3.1319 | 1.514× | 29.56 | 完全相同 |

loss各项、RNG、梯度缺失位置均一致，梯度误差为0；实际核对参数梯度张量数分别1989/1403/2014/917。先前仅关闭短流checkpoint的F/B比为1.246/1.259/1.244/1.331；两波共享负载不同，不相减归因。原始逐次记录：[A](A/ALL_GRAPHS.json)、[B](B/ALL_GRAPHS.json)、[C](C/ALL_GRAPHS.json)、[D](D/ALL_GRAPHS.json)。

## 4. 八卡完整训练态等价性与实际切换

先完成37项针对性CPU测试及实际数据reader对照。四组随后分别从真实训练父checkpoint做原配置/高效配置各两次optimizer更新，world8/global32/原micro不变。模型文件及8个FP32 Adam分片序列化字节全部相同；每rank的sample/cursor/loss、builder和RNG完全一致。不是只测forward或requires_grad。

实际八卡资格父模型：A20k、B15k、C15k、D10k。命令保存在[实际调用](NATIVE_8RANK_INVOCATIONS.json)，逐组证据为[A](A/NATIVE_8RANK_COMPARISON.json)、[B](B/NATIVE_8RANK_COMPARISON.json)、[C](C/NATIVE_8RANK_COMPARISON.json)、[D](D/NATIVE_8RANK_COMPARISON.json)。最初比较器遗留了“四卡随机初始化”文字；原始证据不重写，使用[来源更正](QUALIFICATION_PROVENANCE_CORRECTION.json)明确真实world8及训练父模型，并修正未来分析器文字。该文字错误不影响任何模型、数字或恢复。

资格计时只有两个update，丢掉首个后只剩一个数据点。尤其B观察到3.12×，包含共享负载/数据等待变化，不能宣传为稳定的代码提速。此前另有D四卡global32的8+8update对照，模型/4个Adam文件也字节一致，记录在[D四卡资格](DISTRIBUTED_D.json)，不混称八卡正式实验。

| 组 | 原生切换边界 | 高效版强制保存 | 主机 | 原训练连续性 |
|---|---:|---:|---|---|
| A | 21,167 | 21,169 | training-vlawm-tsinghua-zt6 | Adam/RNG/cursor/累计更新保留 |
| B | 16,948 | 16,950 | training-vlawm-tsinghua-zt | 同上 |
| C | 19,335 | 19,337 | training-vlawm-tsinghua-zt4 | 同上 |
| D | 12,590 | 12,592 | training-vla-zt | 同上 |

四组旧完整恢复点均另外硬链接保护；从这些原生状态恢复，资格权重未混入正式初始化。每组必须实际完成两步并保存整个native checkpoint才判切换成功。全部完成后才adopt同一campaign，单一CPU owner接管原有任务。

- 科学起点：`fdaff809652370328e45c0d996ad706e1af2fc0c`，原snapshot未改。
- 正式高效actor/trainer：`5a2d5b50335e85fd84049ab1f1e5ebe34f537b4e`，`runtime/fast_5a2d5b5`，冻结。
- controller：`68eb4826056b72a2cacc28a92c6fd3b1d9153021`，`runtime/ops_fast_current`，冻结；名字含current不代表可热修改。
- 新稳态CPU owner：2636959，start_ticks1489836327；实时以`campaign/OWNER.json`为准。
- 完整收据：[四组切换](HANDOFF_COMPLETE.json)、[controller差异](OPS_MIGRATION.json)、[CPU验收](OPS_CPU_TESTS.json)。

B最初被共享评测槽挡住的三次启动均在native子进程前拒绝，无训练更新、无评分；改用同机空闲primary槽后成功，失败收据保留。两次CPU操作解释器缺少pidfd的尝试也在发信号前失败；使用系统Python后完成CPU owner接管。D预留覆盖primary/shared/research/evaluation四个lease命名空间，已有评测自然结束后取得其余24把锁，没有停止其他项目。守护器在该原生invocation退出或授权撤回时释放，不宣称永久独占授权。

## 5. 切换后的实测窗口

每组均读取八rank共同的64个连续有效更新，核对每步`cursor=32×update`、optimizer步数和skipped=0；远端核实原生worker、torchrun及8个实际rank均存在。

| 组 | 旧窗口秒/update | 新窗口秒/update | 观察吞吐比 | 新窗口末更新 |
|---|---:|---:|---:|---:|
| A | 5.4304 | 3.6805 | 1.475× | 21,721 |
| B | 5.6611 | 3.7568 | 1.507× | 17,222 |
| C | 4.9095 | 2.8854 | 1.701× | 19,560 |
| D | 8.6830 | 4.5554 | 1.906× | 12,711 |

这是不同时间、数据和共享GPU负载的观察比，不能把全部差异归因于代码。受控F/B实验与八卡状态等价性另列。计时不含checkpoint保存/初始化，不能据此给精确剩余总工期。原始窗口与核验见[新窗口](POST_HANDOFF_LIVE64.json)。

资格成本：[成本账本](COST.json)。八卡对照16次有效update/512次scene曝光；较早D四卡对照另16次/512次；两波F/B benchmark无optimizer更新。全部从正式预算分开，无新增在线评分。八卡命令累计wall2436秒，对应5.41个名义GPU-worker小时，包含加载/保存且共享GPU，不能当独占GPU计算小时。两波rank0基准各约0.145个worker小时。

## 6. 新checkpoint没有结果：队列核查

15:27 UTC只读核验：IL cadence及各相关RL CPU owner均存活。没有统一挂死。IL维持最多两个模型burst、每个最多32分片，seed2101、同一12146场景、single K10 shift5、canonical v1.1。资源不足的未启动分片保留拒绝证据后补派，不重跑已成功分片。

| 权重 | 核验结果 |
|---|---|
| D IL12k | 完整PDMS **86.001457**，12146、invalid0；此前10k为84.846212 |
| TIME_RESIDUAL IL52k | 完整PDMS **88.086223**，12146、invalid0；此前50k为87.592631 |
| HUBER IL32k | 15:27时32/32分片、12146场景已完成，随后CPU汇总结束，完整PDMS **87.137699**，invalid0；此前30k为86.421569 |
| Success Memory RL10k | 当时24/32分片完成、9114场景，余8片初始资源拒绝后已有attempt002，不重复派发 |
| FPO Fulltime RL5k | 完整PDMS **89.704902**，12146、invalid0，CPU汇总已结束 |

A21k/B17k/C19k等1k存档不自动评测：最小间隔2k，下一常规点分别22k/18k/20k。D下一14k，TIME下一54k，HUBER下一34k。额外无损恢复点也不构成新的评测网格。审计时没有已经sealed却漏登记的合格2k点。

报告顶部旧“已部署补测”表读取初始matched请求，当前periodic队列在后面的健康/cadence表。顶部表不能代表全部队列。完整D12k/TIME52k及HUBER32k已被动态报告纳入（最后核验15:31:14）。这里增加[只读队列审计](NAVTEST_QUEUE_AUDIT_1528.json)与真实CLI，明确区分未到间隔、等待admission、推理/评分、CPU汇总、完整完成；不改活跃的其他report/训练owner源码。新全量分数与hash见[封存摘要](NEW_NAVTEST_SUMMARIES.json)。

## 7. 实际可运行入口

以下是已实现并运行的入口。长资格调用必须先有资源admission；不要在生产根重复资格或创建第二个campaign owner。

```bash
# 真实模型配对F/B；原始执行argv以*_INVOCATION.json为准
python -m tools.qwen_world_compare.benchmark_execution \
  --snapshot /mnt/project/qwen-world-four-frameworks-20261029/performance_20261031/inputs/A/SNAPSHOT.json \
  --output /path/to/new_disposable_benchmark --repeats 4 --checkpoint-policy all_off

# 比较已完成的基线/高效两份完整native资格，不做新训练
python -m tools.qwen_world_compare.analyze_execution_pair \
  --root /mnt/project/qwen-world-four-frameworks-20261029/performance_20261031/native_qualification/A \
  --output /path/to/new_comparison.json

# 读取八rank真实连续窗口，含精确远端存活核验；输出路径必须未存在
/usr/bin/python3 -m tools.il_report.summarize_execution_handoff \
  --root /mnt/project/qwen-world-four-frameworks-20261029 \
  --performance-root /mnt/project/qwen-world-four-frameworks-20261029/performance_20261031 \
  --output /path/to/new_live_window.json --verify-remote

# 只读IL/RL队列，不做新模型前向或评分
/usr/bin/python3 -m tools.il_report.evaluation_queue_audit --output /path/to/new_queue_audit.json
```

正式四组的精确启动、resume路径、迁移receipt与lease都保存在`HANDOFF_COMPLETE.json`指向的真实job收据。发生故障时先核实当前owner及完整native恢复边界；不从资格权重重新初始化，不重新启动已完成的handoff，不重复启动仍正常的训练。

## 8. 结论与边界

本轮已确认的工程瓶颈是冗余激活重算和同步CPU数据等待；可以在不改变训练计算结果的情况下显著改善吞吐。D还有统一语言主干的固有计算成本，不能靠相同“参数总数”解释。

继续合并microbatch可能减少Qwen调用与小算子开销，但会改变BF16的batch/padding布局、对象匹配与归约路径，目前没有资格证据，**没有采用**。不通过冻结更多参数、降低精度、删任务/减少token或改训练预算提高速度。加速结果不意味着任何架构必然提分；性能判断继续等待统一完整Navtest。
