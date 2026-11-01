# B/D IL 多节点加速实测：2026-11-01

本轮完成了容器网络、两机 NCCL、真实 B/D 模型的 microbatch 布局测试。**正式 B/D 均继续使用已验证的单机八卡配置，没有重启、重置优化器或改变训练配方。尚未获得通过验收的十六卡加速结果。**

用户已同意开放 RDMA。实际阻塞在容器平台：能看到宿主机活动的 200 Gb/s IB 端口，但容器没有可打开的 `/dev/infiniband/uverbs*`、缺少 `libibverbs.so.1`，设备 cgroup 也未授予 verbs 设备访问权限。当前 Kubernetes service account 对 Pod 读取、修改和 exec 均无权限，无法在工作区自行完成设备开放。已请求平台先开放备用节点，保留现有 B/D 容器。

## 当前正式训练

下表为 10:41 UTC 的一次有限核验，后续进度以原训练日志为准。两组均为 global batch32、完整七项目标、FP32 参数/梯度/Adam、BF16 前向、100k 固定预算。运行源码保持 `5a2d5b50335e85fd84049ab1f1e5ebe34f537b4e`，与探针使用的 51 个模型/数据/训练相关文件逐文件 hash 相同。

| 项目 | B | D |
|---|---:|---:|
| 八个 rank 已记录的共同有效 update | 32,380 | 25,435 |
| 最近 64 次 update 的平均最慢 rank 时间 | 4.192 s | 5.329 s |
| 数据等待均值 | 0.0093 s | 0.0037 s |
| microbatch / accumulation | 2 / 2 | 1 / 4 |
| optimizer global steps / skipped | 32,380 / 0 | 25,435 / 0 |
| 可训练参数 | 2,661,686,323 | 2,220,489,267 |
| 等量 FP32 梯度 payload | 9.916 GiB | 8.272 GiB |
| 剩余有效训练时间外推 | 78.7 h | 110.4 h |

最后一行仅按当前 64 次更新速度乘剩余更新数，不包含 checkpoint、导出、评价、资源争用及故障停机，不能视为交付时间承诺。数据等待占比很小，新增数据读取 worker 不能解决主要瓶颈。

## 实际跨节点通信

已在 B/D 所在节点分别向备用 `vlawm-zt2` 完成有限 TCP 测量：

- B 四连接聚合：2.445 / 2.516 GiB/s；八连接：2.363 / 2.512 GiB/s。
- D 八连接聚合：2.486 / 1.898 GiB/s，存在波动。
- 这是容器 TCP payload 吞吐，不是 NCCL bus bandwidth，也不是 IB 的标称速率。

另在两个备用节点 `vlawm-zt2` 与 `tsinghua-zt3`，每节点一张 GPU，完成真实 NCCL AllReduce：256 MiB、两次 warmup、五次计时，平均 0.2084 s，payload 吞吐 **1.199 GiB/s**。日志明确为 `NET/Socket`、`Using network Socket`、`GPU Direct RDMA Disabled`，且报告无法加载 `libibverbs`。最大分配约 256 MiB，无模型或 optimizer 更新，测试已经退出并释放自己的 lease。

这不是 B/D 十六卡训练测速。不同集体通信算法、八条并发 GPU 链路、ZeRO reduce-scatter/all-gather 及通信重叠都会影响端到端结果，不能把这一数值直接当十六卡训练上限。但结合每步 8.27–9.92 GiB 的 FP32 梯度规模，**目前没有证据支持直接扩容会缩短现有 4–5 秒的更新**。只看到宿主机 IB 端口不足以声称 RDMA 可用。

## 真实模型更大 microbatch 测试

在备用 `vlawm-zt2` 的 GPU2、GPU3，分别读取封存 B32k、D25k 原生模型权重，核对实际权重 SHA256；没有加载或修改正式 optimizer。

每组测试构造相同 32 个 TRAIN 场景及全局有效分母，逐值检查 GT/noise/sigma/action validity/WA queries/WA targets 不变。实际执行原 rank0 的四个场景，交替六次前后向，前两次为 warmup。模型包含全部七项损失。不是短训、不是 Navtest，也没有新增评分调用。

| 项目 | B：micro2→4 | D：micro1→2 |
|---|---:|---:|
| 原 rank-slice F/B 时间中位数 | 2.907 s | 4.185 s |
| 新 rank-slice F/B 时间中位数 | 2.037 s | 2.848 s |
| 吞吐比 | 1.427× | 1.469× |
| F/B 时间减少 | 29.9% | 31.9% |
| 梯度相对 L2 差异 | 0.079919 | 0.066607 |
| 最大 loss 分项绝对差异 | 0.0002144 | 0.0002222 |
| 最大显存分配 | 38.790 GiB | 34.621 GiB |
| 缺失梯度不匹配 | 无 | 无 |
| 模型参数 / CPU及CUDA RNG | 未改变 | 未改变 |
| 无损验收 | **未通过** | **未通过** |

沿用已有 execution benchmark 门槛：梯度 relative L2 ≤ `1e-5`、分项 loss 绝对差 ≤ `1e-6`。未修改门槛。这些布局虽减少 Python/累积开销，但不能据此宣称满足当前无损切换合同，**未应用于正式 B/D**。

同一布局重复前向的分项 loss 一致；跨布局发生可复现差异。差异可能涉及 BF16 算子、padding/布局及离散匹配等，需要单独定位，当前测试没有区分这些来源。梯度差异百分比也不能解释为规划性能下降百分比。这里不据此判断较大 batch 的最终 PDMS，只判断其不满足已登记的数值等价门槛。

这次是单 rank 的 F/B 切片，未计 optimizer/collectives，不能把 1.427× / 1.469× 写成正式训练加速。两个成功探针 worker wall 分别为 122.59 / 118.73 s，含数据与模型加载；属于有限共享资源 worker 时间，不是独占 GPU-hours。每臂六次四场景 F/B，逻辑正式训练更新为零、scorer 调用为零。首次 admission 因已有 lease 拒绝，两个 attempt 均未创建模型子进程；保留日志，使用新 attempt 重试，未抢锁或驱逐评测。

## 十六卡切换仍需满足的条件

1. 平台先在至少两个授权备用容器开放 RDMA 设备访问、相应设备 cgroup 权限、匹配的 verbs/providers 与合法 memory-registration 条件。不能仅安装库，或创建设备文件后假设已获得访问权。不需要为此重启正在训练的 B/D。
2. 重跑容器 preflight 和有限 NCCL，确认实际使用 IB，并测量有效吞吐及可用 GPU/lease；当前预检结果不能自动升级为成功。
3. 再做同 global batch32、同 microbatch、同样本/随机序列的完整两机模型测速。现有八卡入口及 checkpoint 严格绑定 world8；不删除断言绕过。
4. 实现并验证 FP32 Adam 分片重分布、scheduler、全局 scene cursor、阶段、逻辑随机序列和恢复祖先。当前 ZeRO2 八卡 checkpoint 不能直接当十六卡原生恢复点，不能以 model-only 加载和重置 Adam 冒充无损迁移。
5. 只有实际端到端更快且有限 native 恢复/数值验收通过，才在合法 checkpoint 边界进行新的显式执行迁移。当前没有这样的迁移 receipt。

未运行完整十六卡训练或 full-Adam 重分片资格测试。没有为凑空闲卡复活暂停的 T 系列、已终止 ABCD RL 或 Huber；没有修改其他研究任务、训练数据、辅助权重或精度。

## 代码与复现

本任务隔离分支：`perf/bd-multinode-profile-20261101`。原训练工作区未修改。

```bash
# 平台开放后的只读检查：使用新输出名，保留原失败证据
PYTHONPATH=/mnt/project/VLA-Drive-bd-multinode-profile-20261101 \
  /usr/bin/python3 -m tools.il_report.bd_rdma_preflight \
  --output /mnt/project/qwen-world-four-frameworks-20261029/performance_20261101_BD_multinode/RDMA_AFTER_PLATFORM_attempt001.json

# 按已保存的真实实验记录重建本报告数值；不加载模型、不启动训练
/usr/bin/python3 -m tools.il_report.summarize_bd_parallel_profile \
  --assets /mnt/project/qwen-world-four-frameworks-20261029/performance_20261101_BD_multinode \
  --output reports/bd_parallel_profile_20261101/results.json
```

实际执行的模型探针参数保存在资产根 `B_MICRO_PLAN_attempt002.json` / `D_MICRO_PLAN_attempt002.json`，其中记录权重、精确 argv、GPU UUID、cohost 身份、source hash、限时与输出位置。直接重复旧 worker invocation 会拒绝；重新测试须新建 attempt 并重新检查实时授权、物理 GPU、owner 和 lease。

源码入口：

- `tools/il_report/bd_network_probe.py`：有限 CPU TCP 测试。
- `tools/il_report/bd_nccl_probe.py`：两节点一 GPU/rank 的有限 NCCL 测试。
- `tools/il_report/bd_finite_gpu_worker.py`：固定 cohost、UUID、headroom、lease 的有限模型探针；不允许训练入口。
- `tools/qwen_world_compare/benchmark_microbatch.py`：真实模型/全目标的切片数值与性能对照。
- `tools/il_report/bd_rdma_preflight.py`：只读设备与库预检，不修改容器。
- `tools/il_report/summarize_bd_parallel_profile.py`：从已执行记录生成机器可读 [results.json](results.json)。

额外验收：上述脚本 `py_compile`、真实模型 CLI `--help`、三容器 RDMA 预检、两个真实模型 F/B、两节点 NCCL 均已执行。没有把“探针成功运行”写成“更大 microbatch 数值通过”。实际探针 source hash 与发布版 worker 的唯一后续差异另存 `PROBE_SOURCE_LINEAGE.json`：探针结束后增加了同一 benchmark 命令白名单，不改变已执行模型路径。

完整证据位于实际资产目录 `performance_20261101_BD_multinode`：`BASELINE64.json`、`FINAL_PRODUCTION_WINDOW.json`、`SOURCE_AUDIT.json`、`GRADIENT_VOLUME.json`、两组 `MODEL_HASH_VERIFIED`、两组 `MICRO_RESULT_attempt002`、`NCCL`、TCP 输出和 RDMA 权限预检。原始模型、样本 token、缓存、私有凭据不进入 git。
