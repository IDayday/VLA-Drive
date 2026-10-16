# B100k SFT → 自身筛选与八卡验收 → 正式 RL 的衔接

2026-10-16 21:44 UTC，有限衔接控制器已实际启动，并核对了 PID 与进程启动标识。它目前等待 B 的完整100k SFT，没有提前产生筛选候选或启动 B 的 RL。实际证据见 [B_endpoint_handoff_execution.json](B_endpoint_handoff_execution.json)。

本改动只负责已有阶段的衔接。原 A RL、B SFT、里程碑评价控制器继续运行，未中断或热修改；没有新增训练路线、seed、候选预算或 Navtest 观察。

## 固定执行与边界

- 新的 CPU 调度源码：`5ccffc509a7a0919a9d3e8f2f1c70ad719ec65ab`。
- 实际筛选、清单生成、八卡验收与正式 RL 继续调用已验收的 immutable `225b9af660cb0210208e2c46fc4216d0f151275b`。不会将最新工作树作为训练源码。
- 只接受 `B_SOURCE_ACTION_DIT_SFT_seed42_v1/checkpoints/update100000` 的完整正式 seal，且实际 SFT progress 已标记完成。不接受25k/50k/75k、qualification 或其他路线的权重。
- 正式 RL 仍是同一 R_SOURCE：20,000 optimizer updates，seed42；评价 seed2101，native/shift5 分列。初始化来自 B 自身100k SFT，永久 reference 同源；验收产生的 LoRA 不进入正式初始化。

## 已接通的顺序

1. 复用 SFT 汇总器，核对100,000次逻辑更新、3,200,000次场景曝光，不能把 FM repeat 或回滚重算当作新的场景人口。
2. 在完整 RL_TRAIN 做 B 自己的单候选 native ODE 筛选，固定24个 global shard、seed2101。三个8-shard分区可以在三台当前可用主机并行；派发入口本身逐个提交，避免争抢原共享派发锁。若只有一台有合适资源，分区依次执行，不扩大预算。
3. 原清单生成器核验全部评分与 payload hash，按固定 `<0.9` 规则建立 B 自己的困难集合；原分析器单列完整人口、筛选人口和物理成本。
4. 等待独立里程碑控制器完成 B100k 的完整 DEV 和固定 TRAIN 诊断，两种网格都必须有分析结果。
5. 调用原八卡 endpoint gate：真实奖励、连续4次与2+恢复2次、完整状态比较、非零 LoRA 合并以及原始观测部署。B 的该项真实验收仍未发生，不能用控制器单测代替。
6. 原正式 dispatcher 再次检查父权重、困难清单、完整基线、验收源码和证据 hash，然后从自身 SFT 新启动 RL。衔接控制器确认 trainer 的 PID/启动标识及实际首条训练日志后退出。后续5k/10k/15k/20k评价由原里程碑 owner 接管。

候选资源为目前授权的 tsinghua-zt6、tsinghua-zt4、rl-zt2、tsinghua-zt；正式 B RL 位置为 tsinghua-zt6。每次派发重新读取授权文件，采集两次显存/所有者信息并核对物理 GPU UUID。容量不足或所有者无法确认时等待，不终止真实或未知任务。这个控制器不会自动杀占卡脚本；如需要释放，仍须先核实其实际身份。

## 恢复与重复派发

全局 B handoff owner lock 跨控制目录生效。每个阶段先发布 attempt intent，再启动独立 stage worker，记录 PID、启动标识、实际 child、执行源码、日志和退出码；成功派发不等于评分 worker 已成功完成。

失败、部分分区、缺失响应或无成功记录的退出都保留证据并停止自动推进。SSH 超时不被解释为进程已死；没有自动 retry、自动恢复 optimizer 或另起新 run 的路径。发现不确定状态需要先检查对应 receipt 与实际进程，不能直接重跑整条命令。

运行中的控制器收到 `STOP_AFTER_TICK` 文件后只退出控制循环，保留已启动的真实工作。它不拥有其他任务、原里程碑 owner 或 A/B trainer 的停止权。

## 已执行验证

执行了新增控制器测试及原筛选分区测试，合计 **27 passed**。覆盖固定训练源与预算、100k终点门槛、阶段依赖、分区提交串行化、失败不重试、进程观察失败不判死、未知 GPU 占用、部分评分未完成、完整双网格基线，以及 stage worker 实际发布逻辑的 CPU/mock 测试。

实际 CLI `--once` 在当前未完成的 B SFT 上返回 `WAITING_FOR_COMPLETE_B100K_SFT`，没有 stage intent、没有 B 正式派发。随后启动锁定版本的 CPU owner；第二个真实 CLI 尝试被全局 owner lock 拒绝（`BlockingIOError`）。这些证据验证调度边界，不是新的模型数值验收或规划效果证据。

实际启动命令如下；当前 owner 已运行，不应重复执行：

```bash
cd /mnt/project/fm-transfer-20261016/source/5ccffc509a7a0919a9d3e8f2f1c70ad719ec65ab
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.fm_transfer.endpoint_handoff \
  --control /mnt/project/fm-transfer-20261016/endpoint_handoff_B_v1 \
  --screen-hosts training-vlawm-tsinghua-zt6 training-vlawm-tsinghua-zt4 \
                 training-rl-zt2 training-vlawm-tsinghua-zt \
  --training-host training-vlawm-tsinghua-zt6
```

实际状态与阶段记录位于 `/mnt/project/fm-transfer-20261016/endpoint_handoff_B_v1/`。原里程碑队列仍位于 `milestone_controller_v3/`，两者职责和 owner 分开。最终 Navtest 的四模型锁定门槛不变。
