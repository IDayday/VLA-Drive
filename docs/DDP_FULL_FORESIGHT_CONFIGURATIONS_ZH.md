# C0–C5 完整四任务配置记录

共有 **6 个尺寸配置：C0、C1、C2、C3、C4、C5**。尺寸均为宽×高，token 总数是三个前视图之和。

- **C0**：DINO 输入 **128×96**，不池化；每视角 **48** 个 reasoning tokens，三视角共 **144** 个。配置：[c0.yaml](../configs/foresight_resolution/c0.yaml)。
- **C1**：DINO 输入 **256×192**，特征做 **2×2 平均池化**；每视角 **48** 个 reasoning tokens，共 **144** 个。配置：[c1.yaml](../configs/foresight_resolution/c1.yaml)。
- **C2**：DINO 输入 **256×192**，不池化；每视角 **192** 个 reasoning tokens，共 **576** 个。配置：[c2.yaml](../configs/foresight_resolution/c2.yaml)。
- **C3**：DINO 输入 **384×288**，特征做 **2×2 平均池化**；每视角 **108** 个 reasoning tokens，共 **324** 个。配置：[c3.yaml](../configs/foresight_resolution/c3.yaml)。
- **C4**：DINO 输入 **512×384**，特征做 **4×4 平均池化**；每视角 **48** 个 reasoning tokens，共 **144** 个。配置：[c4.yaml](../configs/foresight_resolution/c4.yaml)。
- **C5**：DINO 输入 **512×384**，特征做 **2×2 平均池化**；每视角 **192** 个 reasoning tokens，共 **576** 个。配置：[c5.yaml](../configs/foresight_resolution/c5.yaml)。

池化指 DINO 编码后将每个视角内部的相邻空间特征合并；不对 RGB 图像执行这种池化，不跨视角合并。C0/C1/C4 的最终监督网格都是 8×6，C2/C5 是 16×12，C3 是 12×9。DINO 的 patch size 为 16，特征通道为 1024；空间顺序固定为 view→row→column。尺寸定义的代码来源是 [tradeoff.py](../starVLA/model/modules/foresight/tradeoff.py) 中的 `CANDIDATES`，完整任务校验来自 [config.py](../starVLA/model/modules/foresight/config.py)。该共享尺寸定义也被历史 current-only 代码使用，但本文件指向的六个 YAML 均显式设置 `full_algorithm: true`，不能替换为历史 `configs/dino_tradeoff` 配置。

**六组都训练同一个完整算法：**

当前三前视、导航及允许的 ego 状态/历史 → Qwen 中的共享 reasoning queries W 和原 action queries → 原 DDP Flow-Matching 动作头直接输出 ego 轨迹。

每个原始场景都执行 ego 主任务、一次当前 h=0 DINO 监督，以及一次独立随机请求的 h=1/2/4 秒未来 DINO 监督；有合法交互目标时，再计算冻结 GT vehicle MAE 的表征对齐。缺少辅助标签只屏蔽对应 loss，不删除 ego 场景，不改变学生输入或 W 数量。六组共用同一个经过训练、验证和冻结的 GT-MAE 教师。

总损失为 `L_ego + 1.0*L_cur + 0.9794244300709714*L_fut + 0.8328945981862067*L_int`。future/interaction 权重在前 1000 个 optimizer updates 内 warmup，current 不 warmup。当前与未来使用同一个时间条件化 DINO 读出头；交互读出头读取同一个 W。两种教师和目标缓存均不进入学生当前观测前向。

Qwen 当前图像输入处理在六组之间固定。本表的分辨率仅属于 DINO 标签支路；当前和未来图像使用各自配置相同的尺寸及池化。DINO 从原始源图重新 resize，不放大低分辨率缓存。C0/C1/C4 主要比较离线教师分辨率；改变 W 数量的比较同时改变表示容量和监督空间粒度，不能称为单变量 token 实验。

**共同训练与部署协议：**

- 通用 Qwen3-VL-2B 初始化；驾驶动作头、状态投影、W 和学生辅助头随机初始化，不加载已训练驾驶策略。通用视觉骨干冻结，Qwen 语言及驾驶路径训练。
- Ego DiT 宽 1536、24 层；未来 8 点、间隔 0.5 秒；FM repeat 8，推理 10 步。
- Global batch 32；AdamW 学习率 1e-5、weight decay 1e-3；共同 100000-update cosine 计划、5000-step LR warmup。实际运行以冻结 registration 和 resolved run config 为准，不以遗留 YAML 的无效默认字段推断执行行为。
- 当前优先队列是本机 8 卡 C0、vla-zt2 8 卡 C1，microbatch 每卡 4、accumulation 1；C2–C5 尚未启动正式效果训练。它们的定义和目标缓存保留。本记录不新增训练任务，也不重启已有控制器。
- BF16 学生训练，FP32 optimizer master；正式 PDMS 采用 FP32 master 恢复、FP32 推理、TF32 off，单条执行轨迹。BF16 时延不与 FP32 分数合并。
- 部署保留 W 和原动作头，删除 DINOv3、GT-MAE 教师及两个学生辅助读出头；仅需当前允许观测，最终 ego 直接来自原动作头。

**本分支包含的实际训练代码：**

| 职责 | 仓库入口 |
|---|---|
| 完整学生模型与四项 loss | [ddp_full_foresight.py](../starVLA/model/framework/ddp_full_foresight.py) |
| 学生训练、optimizer、日志及恢复 | [train_student.py](../tools/foresight/train_student.py)、[checkpoints.py](../tools/foresight/checkpoints.py)、[student_state.py](../tools/foresight/student_state.py) |
| 配置/资产校验及多卡 torchrun | [run_student.py](../tools/full_foresight/run_student.py)、[distributed_job.py](../tools/full_foresight/distributed_job.py) |
| 固定里程碑训练、导出及继续训练 | [priority_main.py](../tools/full_foresight/priority_main.py) |
| GT-MAE 真实训练、冻结、ego-hidden 表征导出 | [train_trajectory_mae.py](../tools/foresight/train_trajectory_mae.py)、[freeze_teacher.py](../tools/foresight/freeze_teacher.py)、[export_interaction_targets.py](../tools/foresight/export_interaction_targets.py) |
| h=0/1/2/4 DINO 数据索引和特征缓存 | [build_index.py](../tools/full_foresight/build_index.py)、[cache_dino_targets.py](../tools/full_foresight/cache_dino_targets.py) |
| 纯当前输入轨迹导出、官方 PDMS 评分 | [export_predictions.py](../tools/foresight/export_predictions.py)、[score_pdms.py](../tools/foresight/score_pdms.py) |

完整启动与恢复说明见 [DDP_FULL_FORESIGHT_QUICKSTART.md](DDP_FULL_FORESIGHT_QUICKSTART.md)，已登记运行命令见 [FAST_MAIN_20260929.md](../reports/ddp_full_foresight/FAST_MAIN_20260929.md)。仅在相应控制器已退出且 run 已暂停时使用恢复命令，不能重复启动正在运行的任务。

正在运行的学生训练源码固定为 **d1d40854299b9599b2accc382bcfc4b676dd7623**，控制器源码为 **fb474a72a4be278bae805ddb0e22f26bae073111**。本次文档提交是后续记录版本，不改变这两个运行来源。本分支包含这些源码及后续评估/报告代码；缓存、教师/学生权重、原始数据和私有场景图仍在仓库之外。

C0 是本轮 128×96/144-token **完整四任务**新实验，不是已丢失代码的历史 89.41 当前对齐实验的严格复现。尺寸胜负须由匹配训练后的官方规划结果与实测成本判断，目前不预设任何配置最佳。
