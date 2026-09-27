# Joint trajectory world model：最终实现与有限实验

本轮完成了真实图像条件下的联合自车／周车轨迹图、整条轨迹掩码训练、具有校准几何的 BEV 分支，以及向原 Flow-Matching DiT 的规划条件迁移。训练、真实在线推理、缓存重建和恢复验证均已执行。**工程状态 READY（限下述已验证路径）；研究状态 INCONCLUSIVE。** 当前实验不能宣称稳定规划收益，也不能由未确认收敛的一次实验否定研究方向。

## 代码、基线和协议

- 分支：`feature/masked-trajectory-world-20260927`，远端 `IDayday/VLA-Drive`。完整结果提交 `0703188`；最终文档所在提交由 `git rev-parse HEAD` 确定，交付答复给出远端核验的最终 SHA。
- 原基线可加载兼容代码：`0ecd2ae1f616844641a6d94cfafb50e0f32c26fe`。原权重训练时源码 SHA 未知，不将兼容代码冒充原训练代码。当前工作从 V1.1 `96ff2ee` 接续。
- 原权重：`/mnt/project/DriveDreamer-Policy/models/DriveDreamer-Policy`，SHA256 `9445f9da577a8e3c6b7c636c60a98d714d602668f9a8033b0891c703bc40210f`。
- 继承当前感知分支：`W_POST_64_bounded696_d537401/checkpoint.pt`，SHA256 `fe9aef18e5cd9a9008c538ad166af45fbeb081ed747beaf4f518fdc5e3c3ece4`。不是早期 PDMS 91.37 的候选选择实验。
- 图训练源码 `fed64aa`；图到 DiT 迁移 `7bd8901`；BEV 迁移 `b820f4c`。六个最终训练检查点的完整 SHA、文件校验和、参数及路径见 [FINAL_MANIFEST.json](FINAL_MANIFEST.json)，所有运行见 [RUN_LEDGER.json](RUN_LEDGER.json)。结果不因后续报告代码更新而更改训练来源。
- 固定当前左前／前／右前三视图，保持原导航、自车输入和标定契约；不增加历史图像或环视。当前缓存和监督标签分离。Qwen 使用原 BF16 路径，DiT 使用 FP32 和原 BF16 初始动作噪声。
- 增量训练 7284 场景／978 logs，从原 8192 场景中排除全部 59 个独立诊断 holdout logs。完整开发集沿用既有 1696 场景／16 logs；训练与开发按 log 隔离。**原基础权重是否见过开发集 UNKNOWN，不能称为完全未见过。** 超容量 233 场景从原日志补回完整 GT，未按难度删样本。
- 一个候选、10 个原 FM 采样步，原动作归一化和规划时间 8×0.5s；训练 seed 42，自车评估 seed 20260926，图评估 seed 2037。使用 NAVSIM v1 官方 PDMS 与已验证 compact-cache 适配器，非 v2 EPDMS。没有学习型 scorer、选轨、RL 或 PDMS 训练标签。所有最终检查点预先固定，未按 PDMS 挑选。

## 实际结构和训练范围

当前原视觉特征先形成固定 64 scene／64 agent queries，经 Qwen 后读取当前感知和 actor features。继承的 append-tail 布局保留原 action 条件；新增世界 query 位于原 action tokens 之后，利用显式规划 adapter 传递新信息，而不声称原 action tokens 自动看到了后缀。

联合图包含自车 slot 0 和 64 个周车 slots、8 个未来 xy 点。新模块使用时间注意力、带相对位置偏置的主体注意力及当前场景交叉注意力；坐标是 ego(t0) 米制，相对**预测当前中心**建模，输入不使用 GT 数量、框或未来位置。当前匹配沿同一 track 的有效未来监督，缺失点只参与掩码。

主要实验随机隐藏整条主体轨迹的 Bernoulli 子集，并混合 50% 全隐藏场景；对照是 100% 全隐藏。其他主体的 GT 未来仅用于辅助条件补全训练，部署全部未来隐藏。另实现 `single_actor` 精确隐藏一个主体的模式并完成 16 步真实工程检查；长实验没有使用该模式，不将其冒充“每次恰好隐藏一辆车”的比较。

图输出特征通过投影和门控注意力进入原 DiT，部署最终轨迹来自原 DiT，**不是图里的辅助自车预测**。原 DiT 权重始终固定。主要训练顺序是先训练图，再冻结图训练条件迁移；没有进行大型 Qwen／图／DiT 联合微调。真实小实验另验证了自车梯度可穿过这条链路。

BEV 使用冻结的 Depth Anything V2 ViT-L 视觉权重及新建的校准几何构造：ego 坐标 x∈[1,50)、y∈[−20,20)，1m 网格，49×40 格，0/1/2m 高度投影。它不是已预训练的 BEV 模型，也不是将图像特征任意 reshape；观测支持表示几何视野，不表示遮挡置信度。权重和代码来源／许可见既有 [第三方来源](../structured_world_v1/THIRD_PARTY_SOURCES.md) 和 [BEV_PIPELINE.json](BEV_PIPELINE.json)。提供当前图像离线提取和真实在线推理入口，部署仍需要外部 provider。

**本轮 BEV 在 Qwen 之后进入联合图及 DiT adapter，没有注入 Qwen。** 经过 Qwen 的是图像条件世界 queries。新 BEV 空间编码器／actor 融合使用当前占用、占用位置的跟踪位移、主体对最接近距离，以及全隐藏图 FM 辅助目标；没有未来 RGB 或未来占用栅格重建。与 WCog 的关系是结构化监督的启发，不是严格复现；无 Game-CoT、VAE、反事实动力学或 RL 声称。

| 阶段 | 实际训练参数 | 固定和缓存 |
|---|---:|---|
| 图训练，两个变体 | 949,386 | Qwen／视觉／Reader／当前头／原 DiT；只缓存冻结当前条件 |
| 图到 DiT 迁移，两个变体 | 17,053,697 | 上述模块及图参数固定；训练投影和动作 adapter |
| BEV 迁移，任务开／关 | 18,131,988 声明可训练 | 另固定外部视觉 provider；新增 BEV encoder／fusion 可训练。任务关的辅助头未更新 |

原 DiT 为 819,503,620 参数，外部固定 provider 为 304,368,640 参数，其外部预训练预算未知，未算作零。冻结图／DiT 保持输入梯度；有仅自车 FM 的真实梯度证据。当前缓存只在上游冻结时复用，严格核验版本、sensor contract、图像和标定／权重身份；不会缓存正在训练分支的最终输出。

## 训练长度与收敛

每个图变体使用 **7284 场景×8 遍，3642 optimizer updates**；每个规划迁移变体使用同一数据 **4 遍，1821 updates**。各配对初始化、数据、预算、采样及优化设置匹配；BEV 辅助噪声有独立保存的 RNG，保持配对自车噪声一致。

保留图 0/1/2/4/6/8 遍、迁移 0/1/2/4 遍的固定诊断；图独立 holdout 在全部规定的非零检查点评估。图最后两次 ego／agent 指标仍变化，部分排名随检查点变化。BEV 任务开时开发集 GT-cell motion ADE 从 1 遍的 5.2664m、2 遍的 4.9110m 降至 4 遍的 4.6691m，但占用 IoU 为 0.1349→0.1680→0.1610，并非所有目标稳定。**训练跑满预算不等于已收敛。** 详细曲线和限制见 [CONVERGENCE_ASSESSMENT.md](CONVERGENCE_ASSESSMENT.md)、[图曲线](extended_learning/learning_curves.png)、[迁移曲线](planner_learning/transfer_curves.png)、[BEV 任务曲线](bev_transfer_learning/task_learning_curves.png)。

## 完整规划结果

五组均为相同 1696 场景，失败 0；下表 PDMS 为百分制。完整 8480 行含各小分、失败状态和轨迹校验和的文件是 [scene_metrics.csv](planning_all1696/scene_metrics.csv)，配对场景结果为 [paired_scenes.csv](planning_all1696/paired_scenes.csv)。

| 变体 | PDMS | 零分场景 | 失败 |
|---|---:|---:|---:|
| 原基线 |93.144575|21|0|
| 随机整条主体掩码→原 DiT |93.326985|18|0|
| 全隐藏图对照→原 DiT |93.214365|20|0|
| BEV，同构辅助任务关 |93.312266|18|0|
| BEV，辅助任务开 |93.195079|20|0|

10,000 次 log-cluster 配对 bootstrap，16 logs，差值单位为 PDMS 百分点：

| 配对 | 差值 |95% 区间|
|---|---:|---|
| 随机掩码−原基线 |+0.182410|[−0.033954,+0.384770]|
| 随机掩码−全隐藏 |+0.112620|[−0.005173,+0.219839]|
| BEV 任务开−任务关 |−0.117186|[−0.325411,−0.001443]|

随机掩码的两个区间均包含零；BEV 任务开的规划结果在本固定配方／种子下更差，区间略低于零。不能选择性忽略这个负结果。区间描述当前训练种子的场景／log 不确定性，不包含跨训练种子的变化，也未做多比较校正。完整所有配对见 [SUMMARY.json](planning_all1696/SUMMARY.json)。

## 表征结果与瓶颈

完整开发集独立 2m 当前几何匹配：召回 13.26%，精度 15.95%，类别正确召回 7.79%，匹配中心误差 1.208m；29,212 个当前预测中 24,553 个误报。周车未来点覆盖仅 **13.38%**。这是所有四个新变体共享的冻结感知瓶颈；没有通过丢弃漏检目标改善均值，35,145 个受监督 GT 目标均保留在每组 object CSV。

| 部署导出的联合图 | 匹配周车 ADE(m) | 匹配周车 FDE(m) | 图内自车 ADE(m) | 最终 DiT 自车 ADE(m) |
|---|---:|---:|---:|---:|
| 随机掩码 |5.6961|6.8412|1.5760|0.07078|
| 全隐藏 |5.7905|6.8877|1.3126|0.07080|
| BEV 任务关 |5.6962|6.8412|1.5760|0.07073|
| BEV 任务开 |5.8438|7.0032|1.1814|0.07054|

同一预测当前中心的静止轨迹 ADE 为 2.3804m，优于生成周车轨迹。该结果与低检测覆盖率共同限制“学会驾驶交互”的结论。全部场景及未匹配目标见 [full_joint_metrics1696](full_joint_metrics1696/SUMMARY.json)。此完整导出图噪声协议不同于早期 64 场景 probe，不跨协议拼接数值。

BEV 任务开／关最终当前 IoU 为 0.1610／0.0293，GT 当前占用格上的运动 ADE 为 4.6691／6.8134m，静止参考 5.3803m，717,739 个有效点、0 失败。该密集任务使用 GT 当前占用位置，不能当成端到端感知和周车预测指标。更好的辅助任务指标在本次没有转化为规划提升。

额外 64 场景条件补全诊断保留其他主体真实未来，属于单独的特权信息分析，不进入规划；它显示上下文敏感性，不证明因果交互。32 个真实场景图包含转弯、交汇、静止、空目标和离开视野，保留未匹配 GT／预测；原始图像留在私有 artifact，Git 的 [VISUALIZATION_INDEX.json](VISUALIZATION_INDEX.json) 记录路径和校验和。

## 工程验证、成本和失败

- 29 个相关单元／回归测试通过。实际 1696 个基线轨迹及全部 7 个 PDMS 因子与原归档完全一致：[FULL_BASELINE_REPLAY.json](FULL_BASELINE_REPLAY.json)。
- 实际训练后的图像和两组 BEV 模型各 4 场景，在线当前图像与合法缓存的动作／联合轨迹逐位一致；污染标签不改变输出，独立严格恢复一致。模型保留真实训练门控，无人工放大。[在线结果](learned_bev_online/SUMMARY.json)。
- 原始当前缓存、BEV 缓存的错误版本／传感器／标签字段拒绝检查通过；真实有效缓存通过。世界关闭／零门控保留原路径。
- 图、迁移、BEV 迁移完成实际 GPU 连续训练与中断恢复的严格模型／optimizer／scheduler／RNG／采样顺序检查。BEV 自车单独反向时 encoder／fusion 非零梯度，图参数固定：[BEV_EGO_ONLY_GRADIENTS.json](BEV_EGO_ONLY_GRADIENTS.json)。
- 真实两 GPU NCCL 图模块检查含不同有效数、无周车未来标签 rank、梯度累积和恢复；单卡等效梯度最大差 4.77e−7，在预声明容差内。生产长实验是独立单 GPU 训练；**未实现／验证完整 Qwen 联合 DDP 长训**。[DDP_GRAPH_CHECK.json](DDP_GRAPH_CHECK.json)。
- 图训练峰值分配显存 0.515GB；图像迁移 3.810GB、约 33 samples/s；BEV 任务开／关 6.758／6.537GB、约 15.16 samples/s（均为缓存训练，吞吐包含启动及诊断）。实际 BEV 在线峰值约 15.14GB。
- 4 场景 warm 在线推理图像约 0.520s、BEV 约 0.752s；包含 provider／Qwen／图／DiT，排除图像预处理和模型加载，样本少，非部署延迟承诺。缓存推理更快但不能冒充完整在线成本。
- 消融 blank／跨场景 BEV 在真实训练门控下改变图和动作；任务开 blank BEV 动作最大 xy 差约 0.0007–0.0071m，作用很小。它是使用诊断，不是收益或因果证据。
- 主实验及五组正式场景评估失败均为 0。前期有 4 个失败运行（dtype／parity 检查 3 个、BEV 提取 SHA 参数误写 1 个）及一次后已修复的 RNG 恢复比较失败；全部保留记录和成本，不声称历史从未失败。

当前联合世界模型 campaign 最终 **21,835／24,000 optimizer updates，7.169199／48 GPU-hours**，包含前置 V1.1 的 2596 updates，早期 V1 是另行封存的 campaign。使用本机已允许的 GPUs 0–3，未停止其他任务。剩余 2165 updates 不够再做完整 1821×2 的第二种子配对，因此没有缩短一侧伪称稳定性。只使用 1／2 轮允许的收敛修复；没有据 PDMS 无限延长。

## 结论与未运行项

`engineering_status = READY`：针对固定上游、图／BEV 分支训练和原 DiT 迁移的明确支持范围，真实数据训练、在线重建、保存恢复、BEV 路径和关键回归均通过。该标签不意味着完整 Qwen 联合 DDP、视觉解冻或严格 WCog 复现就绪。

`research_status = INCONCLUSIVE`：随机掩码规划差值未排除零，BEV 辅助任务当前负迁移，周车感知／预测弱且尚未确认稳定收敛；不能给出 POSITIVE，也不能从这个预算内结果推断方向无效。

未运行：第二完整训练种子配对、长训练 exact-one 掩码矩阵、原 DiT／Qwen 全量联合微调、Navtest。没有用 Navtest 选择模型。后续研究优先检查当前感知覆盖与静止轨迹基准，再在新预算下预先固定更长、成对、跨种子的实验；本轮不自动追加。

## 恢复和复现

没有遗留活动训练，状态见 [CODEX_GOAL_STATE.md](../../CODEX_GOAL_STATE.md)。在已有 artifact 的本机重建全部规划表格的一条命令：

```bash
cd /mnt/project/VLA-Drive-masked-trajectory-world-20260927 && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. /root/miniconda3/envs/ddp/bin/python tools/joint_world/report_planning.py --artifacts /mnt/project/joint-world-artifacts/20260927 --output reports/joint_world/planning_all1696 --runs planner_dev1696_baseline planner_dev1696_randommask planner_dev1696_allmask planner_dev1696_bev_control planner_dev1696_bev_tasks
```

完整提取、训练、评估、恢复、可视化和在线检查入口见 [JOINT_WORLD_QUICKSTART.md](../../docs/JOINT_WORLD_QUICKSTART.md) 及各运行 manifest。训练恢复只支持相同代码／设备和精度／缓存身份下已暂停且已核账的运行；已完成 ID 不应重启。远端只推本任务分支的代码、配置和派生指标，不包含权重、缓存、原始数据、私有场景图或密钥。
