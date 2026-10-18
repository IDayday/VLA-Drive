# 实际实现与边界

本轮从 S 执行源码建立隔离 worktree。没有把结果归档当训练源码，没有合并 G/RL 历史，没有修改正在运行的旧项目。

学生只接收 AllowedObservation；RepresentationTargets 与 SampleIdentity 单独交给训练/审计。dataclass 拒绝 teacher_cache、future、LiDAR、GT action 和 scene/log 字段；未新增标定输入。原 Qwen 前缀/DINO/action 序列保留，空间与时间 query 在尾部。world query mask 屏蔽 action 与其后的 suffix，避免通过结束 token 间接读取动作条件。空间不读未来组；时间可读空间。组内 causal。

原 prefix 的原生 Qwen3 SDPA/mRoPE/DeepStack 路径保持 shape，随后使用当前 forward 内可微 KV 计算尾部。不是跨更新缓存，不重复图像编码；此路径关闭语言 activation checkpoint，避免重计算时修改 cache。没有切换 attention backend。

空间 readout 从 256×2048 学生 query 产生 native 256×256×256 context，原 decoder 只用于损失。空间定位编码为二维物理布局。时间 readout 可生成 625×1536，但未训练的时间支路不宣称能力。后8层 attention projection 的 LoRA rank32 由实际 module inventory 选择；视觉、原 DiT/ego/DINO head 在 tiny 表征实验中冻结。

规划 reader 只读取学生生成、参与解码监督的 context/latent。各源 64 个 memory token，经当前 action query 读取并零初始化 residual 改写原动作条件。保留原 Flow Matching repeat8、独立噪声/时间 RNG、10步 Euler。空间位置不是 VGGT 的 view/uv/ray 标签。借鉴 PlanningConditionedDenseVGGTBottleneck 的读取思路，未声称其模块逐算子复用，也没有读取旧 example 的教师 cache。

loss 放在学生 forward 外。固定 decoder eval/requires_grad=False，但学生 context→decoder 不包 no_grad。tiny 的 DINO=1、feature=.1、decode=.1 是预登记探索配方，不是已校准正式配方。warmup200 后的 shared 梯度显示 decode 可明显大于 DINO，故不得不加校准直接用于全量。

恢复保存 trainable delta、optimizer、scheduler、所有 RNG、sampler顺序/位置、基础/教师/配置身份与曝光。基础参数只读载入，delta inventory 严格匹配；不使用 strict=False。四步 GPU 连续/恢复的离散状态完全一致，浮点严格逐位失败保留，并与真实 A/A 重复对照。主训练未采用 BF16；没有宣称长程 bitwise resume。

teacher-free 导出含原 Qwen/DiT、所需 W、LoRA、学生 readout 与 reader；训练用旧 DINO/future/MAE head 可剥离。导出是版本锁定的可信本地 PyTorch research artifact，不是任意不可信 pickle 的安全加载器。第一次因 Transformers input-gradient 局部 closure 无法序列化失败；只移除该训练 hook 后重跑。

统一 campaign CLI 同步执行真实子进程，核对 receipt SHA，复用身份相同阶段，科学依赖不齐时显式 BLOCKED。它不是后台调度平台。当前 full-population P2/P3、Rjoint/Rgt 尚未完成正式验收/实现；相关入口明确拒绝，不能把九组配置清单当九组完成。没有自动启动旧任务或对 Navtest 选择模型。

## 尚未交付为正式能力的实现

P1 的 dense feature MSE 使用完整 native context，解码 queries 限定前向区域；这不是已经完成 observed-free/unknown 支持下的正式 ROI 监督。T-native 的隐藏特征分支仅有接口和固定 head 梯度证据，时间蒸馏没有运行。现有小实验训练器不允许用一个 config 布尔值绕过全量人口/缺失标签/分布式验收；正式 P2 明确拒绝。Rjoint/Rgt 和多阶段完整训练循环尚未完成，不能将矩阵配置解释成实现齐备。

压缩表衡量教师预测一致性，A 对自身的一致性为1。99% IoU/0.5%漏检的数值用于严格保真代理检查，并非已测得的“相对真实 GT IoU 下降≤1个百分点”。原用户建议的 GT 保护线仍因独立标签不足而无法计算。代理不达线不能证明这类瓶颈在充分优化后必然无效，也不能证明最终规划一定退化。
