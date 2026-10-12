# G 系列执行优化：保持效果的验收状态（2026-10-12）

正式训练仍使用已验收的 `io_preserving_v1`，冻结训练源为 `0c54f1c49e7ed2a76b22826b654d6c72625f5b6d`。本轮没有切换正式任务、停止其他研究任务、生成训练权重或重建标签/共同几何资产。新的保守候选已实现，GPU 视觉缓存探针已执行；完整 host-only 训练对照、8/16 卡吞吐和效果验收尚未完成，不能称已经交付无损加速。

**已有 P0 不能因为速度更快就采用。** 原 24 个窗口测得 IO 约 3.20–3.27s/update、P0 约 2.85s/update，吞吐提高约12%–15%；但完整梯度矩阵12/12槽未通过原登记的 E2 A/A 相对门槛，原 strict 失败保留。相同输入的前向精确一致不等于优化器演化已获接受。G1 从25k继续500步的五点 CPU 全量比较早已完成，本轮复核并归档：E1人口/状态合同通过，原严格浮点门槛失败；500步整体 master A/B relL2=0.002390672，A/A=0.002411884，但新增模块组的 Adam 二阶矩 A/B 差异范数是 A/A 的4.601859倍。没有配对 canonical dev 结果，既不能判定规划效果已下降，也不能登记为无损。见[既有 P0 审阅](CURRENT_P0_ACCEPTANCE_REVIEW.json)和[五点完整复核](G1_NATIVE_FIVEPOINT_REVIEW.json)。前一文件中的 G1“待复核”文字由后一独立证据更新，原文件没有改写。

本轮拆出 `engineering_host_fast`，只组合当前输入 CPU 预处理、同一视觉 forward 的长度元数据复用、批量 pinned 传输，以及20-update日志/100-update详细证据。未来 MLN/TokenFuser 公共计算复用、query/footprint/time-field复用、时间向量化和几何BF16均关闭，checkpoint设置保留。预处理与数学复用有独立开关；host-only 不再计算或传输未使用的 label candidate 索引，原 GPU query 路径保留。没有缓存语言 hidden、W、H_A 或动作表示。

代码主要在 [execution.py](../../../starVLA/model/modules/structured_world/execution.py)、[VLA包装](../../../starVLA/model/framework/vla_structured_fgtr.py)、[当前输入collator](../../../starVLA/dataloader/structured_world/prepared_current.py)和[训练器](../../../tools/structured_world/train_vla.py)。所有 fast 模式的正式 scope guard 仍拒绝部署；`engineering_host_fast` 是待验收的 E2 候选，不是把被拒绝的完整P0换名放行。

| 已执行验证 | 结果与边界 |
|---|---|
| host-only CPU | 最终10/10通过；实际Qwen小模型、processor、当前输入/geometry路由、future/query/loss参考路径、RNG及formal拒绝；没有完整GPU模型 |
| world规模与资源门槛 | 7/7通过；8×4或16×2、global32、真实尾batch分母、禁止跨world原生恢复和占用其他任务预约 |
| 原benchmark CPU回归 | 最终49/49通过；首次新增模式缺少测试fixture的失败原样保留，并补齐fixture；不把mock计为GPU实验 |
| node启动入口 | 3/3通过；复用原worker/BoundaryTimer、精确参数、预约拒绝先于CUDA/子进程 |
| 原0c→当前参考数学路径 | 11/11测试、72组比较、2080个比较叶字节一致；直接加载双方原模块和依赖，13份旧文件匹配固定Git对象；仍不覆盖完整Qwen/GeoBEV/DiT/FGTR、GPU或optimizer |

最后一项见[模块桥接摘要](CANONICAL_BRIDGE_SUMMARY.json)和[完整逐叶证据](CANONICAL_BRIDGE_DETAILS.json)。不能用这些CPU证据替代同设备的完整A/A与A/B、原生保存恢复、短继续训练及规划评测。nuScenes实际正式配方仍需单独绑定后进行完整六相机兼容验证。

**冻结视觉缓存做了真实GPU测试。** 在 vla-zt3 的一个空闲 A800 上，使用8个真实NAVSIM训练场景、3个当前相机、已验证的通用Qwen视觉权重。参数FP32、原BF16 autocast/SDPA、TF32关闭；保留视觉最终输出及全部3层DeepStack，输出本身均为BF16，没有额外量化。两次探针的负例与补测分别见[V1](VISUAL_CACHE_PROBE_V1.json)、[V2](VISUAL_CACHE_PROBE_V2.json)。

| 实际测量 | V2结果 |
|---|---:|
| 4场景在线视觉前向及输入H2D，10次中位数 | 0.26644s |
| 8场景按原batch4编码并D2H | 0.62138s |
| 8场景缓存暖读、pin及H2D，6次中位数 | 0.07047s |
| 每场景全部视觉/DeepStack tensor payload | 27MiB |
| 按101592场景估算纯payload | 2,876,227,190,784字节，约2.62TiB |
| batch1编码对原batch4 | 非逐位一致；相对L2约0.20%，最大绝对差0.125 |
| 保持batch4/grid形状，更换场景组合与顺序 | 8场景×4输出共32项逐位一致 |
| V2进程墙钟/1卡成本 | 26.953s / 0.007487 GPUh |

当前只能支持“保持已验证batch形状的预编码值得继续核验”，不能支持任意逐场景缓存直接替代训练。batch2、尾batch3、不同grid、实际训练路径及冷读/多rank共享IO均未验收。暖读数字不可当冷存储吞吐；也不能把局部视觉节省直接当完整step收益。没有构建全量缓存，也没有启用旧teacher cache白名单。两次约216MiB的临时tensor bank均已删除，只归档统计；没有原图或权重上传。V1没有独立进程退出计时收据，不把缺失成本写成零。

视觉分量约0.27s，而语言、FM、真实10步proposal、可训练GeoBEV/未来场/FGTR及反向仍需在线执行。旧完整profile的proposal约0.38s、backward约1.62s，普通数据等待约毫秒级。不能将3秒多全部归因于“没缓存VLM”；可训练语言表示也不允许跨更新缓存。

**8卡改16卡按独立分布式执行候选处理。** 新入口只允许有限的fresh profile：8卡mb4或16卡mb2，global32、FM repeat8、真实10步q0、时刻、网格、K、监督、LR和100k/5000 scheduler均保留。共享几何与通用/随机驾驶初始化一致，但每场景的rank RNG映射、batch形状和归约舍入会变化。它不读取8-rank ZeRO检查点到16 ranks，不证明中期任务可直接无损换world，也不授予正式训练权限。

两台清华服务器均完成固定环境、MMCV可变形注意力CUDA前后向和FusedAdam更新资格验证。共享环境在 `/mnt/project/structured-world-fgtr-execution-assets/20261012/qwen_runtime_tsinghua_v1`，是共享overlay venv加两台已核验的既有ddp基座，并非独立全量conda镜像；原环境没有升级。训练图像实存在共享存储，通过既有`/var/tmp`路径的symlink兼容原manifest，没有复制一份容器数据。

07:23 UTC左右的16卡NCCL启动在准入阶段被真实IQE滚动缓存worker拦下，未启动NCCL worker；当前策略还将两台机器预留给IQE。已询问测试窗口的优先级，在收到安排前保留这些真实任务。没有16卡训练step结果、跨机带宽结果或完整host-only GPU结果可以报告。

已准备[有限窗口manifest V2](WORLD_SCALING_WINDOWS_V2.json)：G1/G3分别8/16卡，三次交替窗口，每次20预热+200稳定更新，共12个profile/2640 updates，默认不保存大检查点。按两种world都3.2s粗估约28.16 GPUh，另加启动成本；这是待实测估算。采用原worker在optimizer20→220边界的最慢rank墙钟；必须收齐8或16份rank收据。旧`analyze_job`硬编码8卡/原生origin，不可直接用于16卡。GPU成本用实际卡数计算；16卡只加快墙钟而未降低同曝光GPU成本时应明确报告。

可复现的入口如下，文件均已存在；当前预约未解除时`--execute`会拒绝。先在两台服务器获得共同空闲窗口并完成新的NCCL资格检查，再按同一job分别运行两节点。下面第一条仅显示命令，不使用GPU。

```bash
PERF_SOURCE=/mnt/project/VLA-Drive-structured-world-fgtr-perf-host-v4-frozen-20261012
PERF_PYTHON=/mnt/project/structured-world-fgtr-execution-assets/20261012/qwen_runtime_tsinghua_v1/bin/python
PERF_MANIFEST=/mnt/project/structured-world-fgtr-execution-artifacts/20261012/tsinghua_fast_validation/WORLD_SCALING_WINDOWS_V2.json

"$PERF_PYTHON" -B "$PERF_SOURCE/tools/structured_world/run_world_scaling_node.py" \
  --manifest "$PERF_MANIFEST" --job G1_w8_r1 --node-rank 0

# 在 training-vlawm-tsinghua-zt 上的8卡窗口
"$PERF_PYTHON" -B "$PERF_SOURCE/tools/structured_world/run_world_scaling_node.py" \
  --manifest "$PERF_MANIFEST" --job G1_w8_r1 --node-rank 0 --execute

# 同一16卡窗口：第一台执行node0，第二台执行node1
"$PERF_PYTHON" -B "$PERF_SOURCE/tools/structured_world/run_world_scaling_node.py" \
  --manifest "$PERF_MANIFEST" --job G1_w16_r1 --node-rank 0 --execute
"$PERF_PYTHON" -B "$PERF_SOURCE/tools/structured_world/run_world_scaling_node.py" \
  --manifest "$PERF_MANIFEST" --job G1_w16_r1 --node-rank 1 --execute
```

host-only 使用原 [benchmark入口](../../../tools/structured_world/benchmark_engineering_execution.py)的 `candidate_mode=engineering_host_fast` 和[数值验证入口](../../../tools/structured_world/validate_engineering_execution.py)的 `--optimized-mode engineering_host_fast`。完整早期/中期G1/G3数值、原生optimizer/BN/RNG恢复、配对继续训练和canonical dev仍须实际完成，不能直接解除guard。正式任务目前只能按已验收IO的安全checkpoint/resume-origin流程继续；尚无已批准的host-only或16卡正式恢复命令。性能pilot不额外用Navtest挑精度，已授权的正式Navtest节奏独立保留。

算法表述继续保持：G3改变road/current occupancy/future/body relations共同使用的结构化监督查询分布；监督塑造Bt，FGTR读取Bt latent进行修正；道路/占用数值不是部署硬约束。q0采样detach是方法定义，refine经H_A/Bt/FGTR回传，不经10步proposal链。这里的执行优化不改变这些研究合同。

实现源：host-only初版 `7fd9840e847385b86f132456a0f36e1f2468eff2`；缓存补测 `1ab59deeb6e4469a420d609a85b0b017cd1a94e1`；最新保守实现/节点入口 `30f70d63556342382367aaff25ddce4342ff256c`。分支为 `perf/structured-world-fgtr-execution-20261011`。结果归档提交与这些实际执行源码分别记录；[证据索引](EVIDENCE_INDEX.json)保留原路径、大小和SHA256。完整更快执行方案尚未获数值与规划效果验收，正式训练保持原版。
