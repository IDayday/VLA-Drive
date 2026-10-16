# DriveDreamer、项目 S 系列与当前 ST 的真实精度审计

本次只读审计；没有改配置、trainer、模型或安装环境，没有新增 GPU 实验、发信号或启动任务。日期：2026-10-16 UTC。审阅工作树起点 `a9cb45d0ac351b82bb4d6e4ed3eac174fe8d8eaf`。数据与文件校验在同目录 `PRECISION_EVIDENCE.json`。

结论：**官方 DriveDreamer 的实际训练入口、项目原 S 系列、当前 ST 均使用 BF16 执行与 ZeRO2 的 FP32 master/Adam 状态。当前 ST 没有把视觉、语言和动作前后向整体改成 FP32。** 新 master helper 增加了一次性的加载/校验工作，修复从已训练父模型初始化时的 FP32 来源；它不在每个 optimizer update 中重新执行。监督目标缓存的 FP32 存储是另一个独立问题。

## 1. 官方起点和实际入口

官方仓库固定为 `youngzhou1999/DriveDreamer-Policy` 的 `f35fe7836625434dc4f67176a3967bf42b0d446d`，由本次 `git ls-remote upstream HEAD refs/heads/main` 和固定 SHA 原始文件核实。本地 `/mnt/project/DriveDreamer-Policy` 是 IDayday 的适配工作树，HEAD 为 `9fe1459b8f6ab69a15274450ec301d541209bedd` 且有用户修改，**没有拿它代替官方源码**。官方原文件只下载到独立审计目录，没有修改任何现有 repo/ref。

官方真实调用链如下，不能误读同目录另一个 `zero2.yaml`：

1. [8-train.sh](https://github.com/youngzhou1999/DriveDreamer-Policy/blob/f35fe7836625434dc4f67176a3967bf42b0d446d/8-train.sh#L26)：8 卡、microbatch 4、DiT hidden 1536/24 层、repeat 8；第 49–54 行通过 Accelerate 加载 `deepspeed_zero2.yaml` 与 `cfg_yaw_1225.yaml`。后续 CLI 覆盖 YAML 示例宽度/层数，最大训练 100000 updates。
2. [deepspeed_zero2.yaml](https://github.com/youngzhou1999/DriveDreamer-Policy/blob/f35fe7836625434dc4f67176a3967bf42b0d446d/starVLA/config/deepseeds/deepspeed_zero2.yaml#L3) 第 4 行指向 `ds_config.yaml`。[后者](https://github.com/youngzhou1999/DriveDreamer-Policy/blob/f35fe7836625434dc4f67176a3967bf42b0d446d/starVLA/config/deepseeds/ds_config.yaml#L1)启用 BF16、关闭 FP16、ZeRO stage 2、accumulation 1、CPU offload false；通信 bucket 为 5e8 elements。
3. [train_starvla.py](https://github.com/youngzhou1999/DriveDreamer-Policy/blob/f35fe7836625434dc4f67176a3967bf42b0d446d/starVLA/training/train_starvla.py#L100) 第 104–110 行建立 `torch.optim.AdamW`；第 245–251 行调用 `TrainerUtils.setup_distributed_training`，实际在 [trainer_tools.py:281](https://github.com/youngzhou1999/DriveDreamer-Policy/blob/f35fe7836625434dc4f67176a3967bf42b0d446d/starVLA/training/trainer_utils/trainer_tools.py#L272) 进入 `accelerator.prepare`。训练 forward 在 BF16 autocast 中（530–531），然后 Accelerate backward/optimizer step（558–569）。
4. [QWen3.py:65–68](https://github.com/youngzhou1999/DriveDreamer-Policy/blob/f35fe7836625434dc4f67176a3967bf42b0d446d/starVLA/model/modules/vlm/QWen3.py#L65) 加载 BF16 Qwen 和 FlashAttention2。`QwenOFT.py:419–429` 的 Qwen 前向同样处于 BF16 autocast。代码存在局部 FP32 action/状态区段，不能把局部声明等同于全模型 FP32 执行，也不能据 BF16 配置声称所有算子都使用 BF16。

官方 `requirements_vla.txt:2,55,318` 固定 Accelerate 1.5.2、DeepSpeed 0.16.9、Torch 2.5.1。当前安装版本相同（Torch CUDA 构建为 `2.5.1+cu124`）。本次将安装的 DS engine、ZeRO2、FusedAdam 和 Accelerate accelerator 四个实现文件与相应官方 release 原文件逐字校验，**四者 SHA256 全部一致**。这是配置与源码推导；没有冒充本轮重新训练了官方模型，也没有官方运行时 profiler 证据。

## 2. 为什么官方 BF16 也有 FP32 master

这不是根据 `bf16=true` 猜测：已沿官方版本的执行路径逐级确认。

- Accelerate 1.5.2 `accelerator.py:1927–1953` 将现有 AdamW client 传给 DeepSpeed；官方无 offload，故此处不替换为 CPU Adam。DS `engine.py:1378–1386` 保留这个 client optimizer。
- DS 0.16.9 `engine.py:1251–1254` 将执行模块转 BF16。`get_data_types():1033–1048` 在该 BF16+ZeRO2 配置下默认梯度累积类型为 BF16。
- [ZeRO2:397–417](https://github.com/microsoft/DeepSpeed/blob/v0.16.9/deepspeed/runtime/zero/stage_1_and_2.py#L397) 将每 rank 的权重分区复制成 FP32 master，并把基础 optimizer 的参数指向该 master。Torch AdamW 2.5.1 `adamw.py:148–153` 通过对应参数的 `zeros_like` 初始化两个 moment，所以它们也是 FP32。
- [ZeRO2:1900–1929](https://github.com/microsoft/DeepSpeed/blob/v0.16.9/deepspeed/runtime/zero/stage_1_and_2.py#L1900) 在 optimizer 边界将分区梯度转为 master dtype，进行 clip/update，再把更新后的 master 写回 BF16 执行权重。BF16 梯度存储、FP32 optimizer 边界梯度与 FP32 master 是不同对象。

项目 S 系列和当前 ST 则在 DS config 中选择 AdamW，由 `engine.py:1438–1462` 建立 **FusedAdam，AdamW mode**。其 `fused_adam.py:155–157` 的 moments 跟随 FP32 master，169–192 进入 FP32 update kernel。官方使用 Torch AdamW client，项目使用 FusedAdam，这是实际实现差别；不能将原实现笼统称为 Python for-loop，也不能把 FusedAdam 的 FP32 更新说成新增的全模型 FP32 训练。

## 3. 三方区别及当前真实观测

| 项目 | 官方固定入口（源码推导） | 项目 S0 已完成训练 | 当前 ST 真实运行 |
|---|---|---|---|
| 来源 | 上述官方 f35fe783 | 1493deda，formal_S0_seed42_full100k_v2 | 2d3e4f5a；长窗 profile 源为 58f85044/ab2184bf |
| 执行权重 | BF16 | BF16 | BF16 |
| Qwen visual / language | BF16 模块及 BF16 AMP；FA2 | BF16；项目 SDPA 后端 | 实测 BF16 visual/merged/三个 DeepStack/语言；PyTorch BF16 Flash kernels |
| ZeRO 累积梯度 | BF16 默认 | 同 DS 配置；历史观察重建 FP32 full grad 不等于原存储类型 | trace 的 `averaged_gradients` 全 rank 为 BF16 |
| optimizer 边界梯度 | 转 FP32 分区 | 转 FP32 分区 | 转 FP32 分区 |
| master / exp_avg / exp_avg_sq | FP32 / FP32 / FP32 | FP32 / FP32 / FP32 | FP32 / FP32 / FP32，实测 |
| 基础 optimizer | Torch AdamW client | DS FusedAdam | DS FusedAdam，实际初始化记录确认 |
| CPU optimizer offload | false | false | false |
| allgather/reduce bucket | 5e8 elements | 5e7 elements | 5e7 elements |
| 当前 DINO target storage | 此入口的任务与本地 S 不相同 | FP16；teacher 本身 FP32 | SP0 复用旧 FP16；SP1/SP2 当前注册 FP32 |

S0 原文件 `tools/foresight/train_student.py:129` 已明确写出 BF16 execution / FP32 master+moments；193–202 为原 DS 配置，313–319 每个 optimizer 边界核实 master 实际变化。正式 S0 的 `identity.json` 与上述源码相符，global32/micro4/world8、offload=false。最后一步 100000 的日志仍有 `fp32_master_update_verified=true`，所以原约 1.7 秒训练并不是去掉 master 得到的。

当前 8 卡 trace：2,959,630,724 个独立参数，2,552,673,668 可训练；各 rank 的 master 和两个 moment 分别有 319,084,210 个 FP32 元素（含分区 padding）。真实 action 分支 batch 为 micro4×repeat8=32；没有缩掉 repeat/DiT。详细证据复用 `../profiles/MODULE_DTYPE_AND_CALLS.json`，而不是重新加载几十 GB checkpoint 计算一遍。

当前标准独立评估采用绑定的 `FP32_master_eval_TF32_off`。这是**评估路径**；`train_world.py:161–163` 加载规范 FP32 父权重后恢复原训练 AMP，随后 DS 重新创建 BF16 执行模块。不能把读到的 `precision="fp32"` 参数截取出来，当作全部训练 forward 都是 FP32。

## 4. 新 master 修复：启动成本和每步成本必须分开

新 helper 的意义：父模型从真实 FP32 master 重建后，DS 默认先转 BF16，再由已舍入 BF16 创建 FP32 master。新实验若直接接受此状态，master 将丢失父权重低位。`fp32_master_init.py:190–216,350–396` 在 fresh 初始化前捕获规范源，DS 初始化后恢复**原有 FP32 master 分区**，并核对 BF16 execution 未改变。没有新增永久 GPU master，也没有把 execution 改回 FP32；native resume 跳过 fresh 修复。

实际新源码 `2d3e4f5a` 两项真图资格运行均完成 2 updates、梯度/损失有限并保存原生 checkpoint。它们是正确性 smoke，不能当长期吞吐样本。

| 本次记录 | W / 2 GPU / G4 | C1 / 1 GPU / G2 |
|---|---:|---:|
| capture，rank 范围 | 27.228–27.297 s | 10.891 s |
| pre-write validation，rank 范围 | 16.849–18.216 s | 8.635 s |
| restore+post-verify，rank 范围 | 6.196–6.224 s | 3.720 s |
| 单 rank 三段时间和最大值 | 51.640 s | 23.246 s |
| 暂存 CPU 源分区 / rank | 5,105,347,336 bytes | 3,278,014,480 bytes |
| 新增永久 GPU bytes | 0 | 0 |
| execution unchanged | true | true |
| 原生更新实际实现 | FusedAdam | FusedAdam |
| 两次慢 rank inner update | 2.591 / 2.022 s | 31.020 / 27.476 s |
| 最终完整 checkpoint 写入 | 56.288 s | 22.583 s |

三段时间和是同一 rank 已计时区段的和，不包括所有区段间 barrier，不能当完整 launch wall。W 的 `run()` 内 allocation wall 为 **340.174 s**；包括数据/身份检查、父 checkpoint 校验/加载/重建、DS 初始化、两次更新、日志和完整保存。父 checksum、反序列化及 FP32 重建**没有分别独立计时**，因此无法准确给出它们各自秒数；不能把所有剩余时间都归因于 master helper，也不能用 340/2 声称每步 170 秒。

C1 的 2 步分别还执行 4 rollout/scene、10 Euler、MC50 old/current、官方评分，不能和 W 的普通 IL update 比速度。其 `current_BC_FPO_backward_optimizer` 是合并区段（16.744/16.937 s），没有纯 optimizer 单独测量；不能把该合并耗时全部计为 Adam。

## 5. 已有长窗和 trace 说明了什么

| 已实测配置 | 有效稳态 updates | whole-update mean / p50 / p95 | 场景/s |
|---|---:|---:|---:|
| SP2，8 A800，G32/micro4 | 200（总 240，两个固定窗口） | 2.04583 / 2.02588 / 2.23847 s | 15.6331 |
| SP3/W96，8 A800，G32/micro4 | 200（总 240） | 2.06030 / 2.04572 / 2.18834 s | 15.5240 |

这些真实测量本来就含 FP32 master/FusedAdam。各使用固定真实 64-scene 循环计算工作集，不能当全人口随机共享 IO 的资格。长窗实际源为 58f85044/ab2184bf；新 helper 只改 fresh 初始化，optimizer loop AST 不变。旧证据可以支持计算形状与稳态资源分析，不能冒称已测新源码完整 launch wall、全人口 IO 或数值训练轨迹。参见 `../profiles/FRESH_MASTER_EFFICIENCY_INHERITANCE.md`。

独立 8 卡 2-step trace 中，`st.optimizer` 即 `engine.step()` CPU inclusive span 平均 **44.371 ms**（各 rank/step 27.561–60.638 ms），不是纯 CUDA Adam kernel 的严格计时。`st.backward` 平均 872.924 ms；CUDA forward inclusive span：visual 250.813 ms、language 201.363 ms、DiT 72.546 ms、sparse readout 6.559 ms。未带 profiler 的 steady slow-rank data 字段平均 393.115 ms。计算、通信、CPU 和多 stream inclusive 时间互有重叠，不能相加成总 wall。

这些观测不支持“现在主要因 FP32 master 而慢”的归因。实际主要工作是 language/action backward 与重计算、visual/language forward、数据/通信协调。也不能据此保证删除 master 只有某个固定百分比收益；尚无这样的 A/B，且本轮明确没有追加 GPU 性能试验。

## 6. FP32 目标缓存、权重和旧动态池是三件事

原 S0 真实 `identity.json` 的 current DINO 缓存为 6×8×1024/视图、FP16；future clip 目标也存 FP16，而生成 teacher 使用 FP32。本轮 SP2 注册的原始 DINO target 为 18×24×1024/图、FP32，随后固定 PCA384 按已注册数学处理。该缓存是 **离线监督标签**，不是可训练权重，不是 Adam master，也不是用于替代在线 Qwen visual 的 IQE 冻结视觉动态池。

SP2 当前锁定清单 304,776 张 current images 的逻辑 feature+mask 量为 **539,424,261,504 bytes（约 502.4 GiB）**；文件/index/receipt/临时写入另算。SP0/SP1/SP2 合计计划逻辑张量为 629,335,619,712 bytes。这里是注册清单的规划容量，**不是已经完整生成的状态断言**。没有以旧动态池的数 TB 历史容量替代当前计量。

把同形状 FP32 target 存为 FP16 会减半 feature 部分，但会量化标签；这不等价于开启原官方 BF16 权重训练，不能静默修改当前 SP2 数据身份。旧动态池服务的张量、producer/consumer 与当前 teacher-label writer 不同，本审计没有启停或复用它。是否另立精度/存储变体由 Lead 与用户决定，不在此次只读审计中改变。

## 7. 去掉 master 的边界

把参数和 Adam moments 都直接保留 BF16 会改变 optimizer 数值定义和 checkpoint 契约。BF16 在 1 附近相邻值间距为 0.0078125，而当前真实 warmup master 更新已测为 1e-8–2e-7 量级；FP32 master 保留跨步积累，小更新不必在每步 BF16 execution 中可见。直接 BF16 更新可能被舍入掉，不能称为原 S 系列无损提速。

保留 BF16执行+FP32master 的结论与缓存存储选择分开。当前最有依据的行动是继续原已核验 backend，单列启动/恢复开销与训练吞吐；如需优化父 checkpoint 加载或一次性校验，先补真实分区计时并保持完整性。本审计没有给予未跑配置 formal PASS，也没有重开任何旧实验。

实际运行证据：W `smoke/world_fp32_master_real_v1/runs/world_fresh_master_updates2_v1/`；C1 `smoke/flow_fp32_master_real_v1/runs/flow_fresh_master_updates2_v1/`，均位于 `/mnt/project/st-world-policy-20261016/`。所有官方文件、依赖 hash 和真实记录定位在 `PRECISION_EVIDENCE.json`，未提交权重、缓存、图像或 trace。
