这份准备把既定 C3 的 512、1024、4096 三个完整 dev 端点绑定到已实际启动的训练。注册 SHA 为 `024e0e86eec8f1c9d2a837a6c8a2b600f2e1f39805cf9984cc7a6e2466a852d2`，运行 identity 为 `4b2c54f241b13cfdb9518f23b8759d51142c84a53db28826366fc7d280314999`，初次启动固定到 `invocations/1792212057408455570/LAUNCH.json`。本次保存的真实状态为 RUNNING、completed 25；它只证明训练已运行。

原 `full_dev_C2_C3_execution_v1` 的 C3-null 文件保持原样。这里沿用其三个 C3 job ID、输出目录、C0 同阶段控制和分析目录，因此新增实验数、额外评分调用预算和额外存储预留均为零。C2 继续使用原追加版本，原十项准备也保持原样。C4 不在本次绑定范围。

`EXECUTION_PLAN.json` 只新增 C3 的实际执行身份和观察到的状态。科学定义逐字段继承 v4：完整 dev 1696 场景 / 16 logs；训练与推理 seed 42；C3−C0；512/1024 仅作诊断，4096 为共同终点；2000 次配对 log-cluster bootstrap、PCG64 seed 20261017；全人口分项、零分双向转移、高分退化、进度/停车诊断和无效处理不变。其他 arm 的状态是此前观察，未在本次刷新。

三个 checkpoint 的 tag、COMPLETE、receipt identity 和 native file map 仍为 null。本次未打开 native 目录或权重，也没有把 step 0、latest 或邻近更新代替目标端点。实际评估注册必须先绑定精确目标步的完整 native receipt；冻结 runtime 在使用前继续验证完整 payload。所有 export、score、analyze 均固定在 clean `6714b1e9fb2e0c8059b90b8b4d48cd40796fe345`。

`COMMANDS.json` 保留原三项 export/score argv，仅将分析 plan 改为本版本，并补真实训练 identity。推理仍为 world1 / rank0、单轨迹原 Euler10、FP32 master、TF32 off。评分 `--run-id` 统一为 `st-world-policy-20261016`，每 arm/step 的输出和持久 ledger 独立。v1/v2 helper 仍按旧 pins 拒绝 C3；使用本版本需要 Lead 独立审阅并部署一个仅扩展固定 pins 的后续适配，不能绕过 helper 直接调用模板。

每个端点继续使用 1696 主调用、1728 总 admission 硬上限、retry_attempts=1、最多两次 export 与两次 score invocation、1 GiB 输出预留。32 是完成所余的总调用空间，不是独立重试计数器；恢复不重置 ledger。C3 合计 5088 主调用 / 5184 总上限 / 3 GiB，已经包含在旧六项预算中，不能再次相加。export 内/外时限为 7200/7800 秒，score 为 3600/4200 秒；全 campaign 最多四个独立单 GPU export、两个各两 worker 的 CPU score。原权重和 metric cache 原地复用。

`VALIDATION.json` 记录 45 项有界元数据核验：真实注册/三审/approval/初次 launch/runtime identity、源码 pins、科学定义、固定命令/人口，以及旧文件和 helper 未变。没有读取模型、评分、GPU、Navtest，也没有产生科学结果或派发授权。

`HOST_ROUTING_PROPOSAL.json` 仅提出 vla-zt / tsinghua-zt 在自然释放后作为候选宿主。现有证据支持同型硬件和基础软件环境的继承；当前空闲、现场环境与推理耗时仍须实际核实。最小后续修改需保持同一 job 的 export/resume/score 同宿主、共享 campaign 的四个 export / 两个 score 槽位锁、原本机 GPU 锁和新鲜准入。C2/C3 的 rl-zt2 训练卡不共享。本次没有修改或部署任何 helper。
