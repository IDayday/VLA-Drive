# 高分辨率教师 → 稀疏 W：目标对齐的有限方案

审计日期：2026-10-17 UTC。当前实现起点 `951711fbd2eae1730262a79575b08024bcd2001d`；本证据工作基于保留道路代理提交后的 `42ad6fcbeca99e1bf99df757191fca0edbad742f`。仅阅读论文全文、固定官方源码及本地实现；**没有修改模型、启动训练、使用 GPU/SSH 或改动运行中的实验**。这是 Evidence agent 的建议，不是独立正确性审批或效果验收。

推荐保留教师 `384×288` 的原生细网格。先用已经接线的同一训练集 PCA 基底判断 256 维是否保留所需信息，再在固定维度下比较原 MSE 与近零曲率匹配的 SmoothL1。不要先叠加更大的读出器、白化和梯度手术。PHI-S 可作后备的尺度/坐标参数化对照，当前单教师场景没有理由先把它列为必胜方案。

## 1. 当前实现不是“直接硬拟合所有高维特征”

| 已核实事实 | 代码与含义 |
|---|---|
| 教师高分辨率对应 `432×1024`，输出通道仍是 1024 | 分辨率提高增加空间位置/可见细节，不自动增加每个位置的通道数；不能把空间压缩与通道压缩混为一谈。具体教师尺寸由本 campaign 契约给定。 |
| 已有训练集通道标准化、非白化 PCA256/384，且两维度使用同一基底 | `tools/st_world_policy/teacher_pca.py::StreamingCovariance.basis / TeacherPCA.transform / inverse`；`standardize_channels=True, whiten=False`。这已是 DINO-Foresight 式处理，不是本次新发明。 |
| W48/96 与教师 patch 数量相互独立 | `SparseFeatureReadout` 从完整 W memory 读取给定视角/坐标/时间的查询；预测位置的数量不要求等于 W token 数。 |
| 已经有 2 层 cross-attention + FFN | `starVLA/model/modules/st_world/sparse_readout.py` 复用 `foresight/future_latent_head.py::CrossReadout`；仅最后输出层是 Linear。不能再把“增加非线性读出器”泛称为从无到有。 |
| 当前损失是按有效位置×通道全局归约的 FP32 MSE | `starVLA/model/modules/st_world/losses.py::masked_mse / optimizer_batch_counts`；不是未归一化的平方和。更多维度不必然导致数值更大。 |

同一 PCA 基底中，增加低特征值的后续通道，可能仅通过扩大均值分母降低原始 MSE。因此，256/384 的 raw mean MSE 不能直接用来判定哪组学得更好。推荐同时报告：

1. **公共前 256 维误差**：384 维输出取前 256 维，与 256 维模型使用完全相同的位置、通道和分母比较。
2. **相对目标能量误差**：`sum(error²) / sum((target - train_mean)²)`；分母来自固定训练统计，不能在 dev 上拟合。分母近零报告 NA。
3. **逆变换到原 1024 通道的误差**：区分教师 PCA 重建误差与学生预测误差；不把“保留方差高”直接解释为障碍物/边界信息充分。
4. 当前、未来 A/B、时间差分分别报告。未来不确定性、观测缺失和优化困难不是同一个问题。

## 2. 证据卡 A：DINO-Foresight 的紧凑目标与稳健回归

**原始证据。** 全文 §3.1/3.2、附录 A.1/A.5 使用 PCA 目标和 `SmoothL1(beta=0.1)`。表 5 的短期语义 mIoU 为无 PCA 71.3、PCA 71.8；中期为 59.9、59.8，并非所有指标都提高。表 10 的 MSE 与 SmoothL1 结果接近：短/中期 mIoU 分别为 71.7/60.0 与 71.8/59.8。该论文没有证明 SmoothL1 普遍优于 MSE。[论文全文](https://arxiv.org/html/2412.11673v2)

**实际官方调用。** 固定 SHA `d98f765d2b743ec90fb38bc657a39c07c7ca4ab0`：`pca.py` 先逐通道标准化，再调用默认不白化的 sklearn PCA；`train.py` 默认 `loss_type=SmoothL1, beta_smoothl1=0.1`，README 的训练命令也显式启用；`forward_loss` 对有效位置切片后调用 `calculate_loss`，最终是 PyTorch 默认 mean 归约。论文公式写求和，工程接线应以实际全局分母为准。[PCA 实现](https://github.com/Sta8is/DINO-Foresight/blob/d98f765d2b743ec90fb38bc657a39c07c7ca4ab0/pca.py)；[损失实现](https://github.com/Sta8is/DINO-Foresight/blob/d98f765d2b743ec90fb38bc657a39c07c7ca4ab0/src/dino_f.py)；[训练参数](https://github.com/Sta8is/DINO-Foresight/blob/d98f765d2b743ec90fb38bc657a39c07c7ca4ab0/train.py)

**对当前系统的有限适配。** PCA 部分已实现。真正值得优先验证的新增单因素是：固定高分辨率、同一 PCA 维度/基底、W 数量、稀疏位置、样本顺序、FM 配方和辅助权重，只替换逐元素平方损失。

令残差为 `e`，原损失 `e²`。直接照搬上游 `SmoothL1_β` 在 `|e|<β` 时为 `e²/(2β)`；β=0.1 时近零曲率是原 MSE 的 **5 倍**，并不只是降低大误差。为使对照更易解释，主建议采用：

```text
rho_beta(e) = 2*beta*SmoothL1_beta(e) = 2*Huber_beta(e)
            = e²                         if |e| < beta
              2*beta*|e| - beta²         otherwise
beta = 0.1  （在锁定 PCA 坐标单位中）
```

这保留近零 MSE 的损失/梯度，尾部梯度由 `2e` 变为 `2β sign(e)`。它是**曲率匹配的本项目适配**，不能标成未修改的 DINO-Foresight 配方。保持原有效 mask、全局有效元素分母、DDP/accumulation 规则以及辅助权重；未来差分继续用原幅值，不做逐向量单位归一化。

**β 是坐标单位，不是可随目标定义直接照搬的无量纲常数。** 上游 β=0.1 不是当前 DINOv3/PCA 的已验证最优值。进入真实比较前先用固定训练统计记录残差分位与 `|e|<0.1` 的比例，并锁定所用变换、尺度和阈值。若整体坐标缩放为 `z'=s*z`，等价转折点应为 `β'=|s|β`，损失再除以 `s²` 才保持原近零几何；逐主成分各自缩放则不能用一个新 β 恢复原目标。本轮建议的 0.1 只是明确的文献起点，不能跨 raw/PCA/白化坐标静默继承，更不能看 dev 后追调阈值。若绝大多数误差均落在线性区，应如实称接近 L1 的训练定义，而不是已证实只抑制少量异常值。

**可证伪假设 H-A。** 若少量难预测残差支配辅助梯度，曲率匹配稳健损失应降低其集中度，并改善固定留出 FM/规划指标或 W 的任务读出，而非仅降低所选损失数值。

**必要诊断与失败可能。** 按当前/未来、动态物体、小物体、道路边界等已合法定义区域统计残差尾部和预测误差；没有现成区域标签就明确缺失，不引入外部模型伪造。大残差可能恰是有价值的困难目标，截断会压低其学习信号。若收益只是总辅助梯度变小，应单列固定权重降低的解释；不得据此宣称鲁棒性已获证明。原论文的预测器与本项目稀疏 W 容量不同，其数字不能直接迁移。

## 3. 证据卡 B：PHI-S 是分布平衡，不是信息压缩或消除冲突

**原始证据。** PHI-S 对异质教师使用均值中心化、正交旋转和单一尺度；不按各特征值做白化。其主要证据是多教师平衡。论文表 3 的 DINOv2 原空间 MSE，global standardization 为 0.8801、PHI-S 为 0.8865，所以不能以该表声称单 DINO 教师必然受益。[论文全文 §2.2 与表 3](https://arxiv.org/html/2410.01680v1)

**固定官方实现。** NVlabs/RADIO SHA `4ffaf15d0ff48197c194e306772bdb5fb92db230`，`examples/common/phi_s.py::get_phi_s_matrix` 计算协方差特征分解、Hadamard 旋转和平均特征值平方根尺度。`radio/feature_normalizer.py` 将变换作为固定 buffer。[统计函数](https://github.com/NVlabs/RADIO/blob/4ffaf15d0ff48197c194e306772bdb5fb92db230/examples/common/phi_s.py)；[归一化模块](https://github.com/NVlabs/RADIO/blob/4ffaf15d0ff48197c194e306772bdb5fb92db230/radio/feature_normalizer.py)

行向量约定下，`cov(X)=U Λ Uᵀ`，`R=H Uᵀ`，`s²=mean(λ)`，`z=(x-μ)Rᵀ/s`。这是保留子空间内的可逆变换；完整协方差通常不是单位矩阵。

**必须避免的误归因。** 对相同误差向量和全通道 MSE，正交性严格给出：

```text
mean((R*error)²) = mean(error²)
mean((R*error/s)²) = mean(error²)/s²
```

若预测与目标都通过同一固定变换，旋转不降低拟合难度；标量只改辅助任务相对尺度。若让头直接预测旋转后的坐标，则初始化、参数化和 Adam 更新会改变；这是优化假设，不是增加信息。当前已有逐通道标准化，PHI-S 接在 PCA 后只保留**标准化且截断后的子空间**几何，不能称其保留原始 1024 维全部信息。

**建议定位：后备。** 只有训练统计证实条件数/尺度问题后，比较“单一标量标准化”和“相同标量+PHI-S”；固定维度，不同时换 loss。256 维可用精确 Hadamard；其他维度必须核验构造正交性与等幅性，禁止静默使用近似 fallback。官方 `allow_approx=True` 可能回退 QR；独立适配必须显式拒绝不符合声明的构造。[Hadamard 实现](https://github.com/NVlabs/RADIO/blob/4ffaf15d0ff48197c194e306772bdb5fb92db230/examples/common/hadamard.py)

逐 token 单位范数、逐主成分白化、逐时刻标准化都不是上述单标量策略；它们改变目标几何/时序幅值，应另立假设，不能混入本轮。

## 4. 证据卡 C：轻量预测头可隔离接口瓶颈，不能代替 W 表征

**原始证据。** RADIOv2.5 §2.1/2.2 使用学生到教师空间的轻量 MLP adaptor，基线为两层投影并含 LayerNorm/GELU；这提供接口适配的成熟先例，不提供“越深越好”的普遍证据。其表 3 分布平衡结果有教师间权衡，不能替代本项目下游验证。[论文全文](https://arxiv.org/html/2412.07679v2)

**官方实现。** 同一固定 RADIO SHA 的 `radio/adaptor_mlp.py::MLP2`，`num_inner=0` 对应 `Linear → LayerNorm → GELU → Linear`。该固定源码是当前仓库版本，不冒称它与论文发布时所有训练细节逐字一致；公开 adaptor 也不等于完整蒸馏训练流程已复现。[固定实现](https://github.com/NVlabs/RADIO/blob/4ffaf15d0ff48197c194e306772bdb5fb92db230/radio/adaptor_mlp.py)

**当前适配边界。** 已有 2 层 CrossReadout 保持不动，仅在确认接口欠拟合后比较现有 `Linear(512,C)` 与小型 `Linear(512,512) → LayerNorm(512) → GELU → Linear(512,C)`。它只作用于训练期辅助预测，不增加部署 VLM 或改变单轨迹动作头。不能同时增加 W token、读取层数和目标维度。

**可证伪假设 H-C。** 若同一冻结 W 上小头可以显著改善留出特征预测，而更强教师/PCA 目标本身仍有任务信息，读出映射可能存在容量限制；如果训练误差降低、留出不改善，或 joint training 后 W/FM 指标不改善，则不能把它解释为更好的世界表征。更强的头可能自行吸收任务、减少对共享 W 的约束，所以应报告头前 W 的固定线性/轻量 probe，不只报告头后对齐 loss。

此项低于 PCA 信息诊断和单因素损失对照的优先级。当前不是一个“仅线性学生”，没有证据支持继续盲目堆深 projector。

## 5. 可直接执行的最小比较顺序（本 agent 未启动）

| 顺序 | 固定项与唯一变量 | 回答的问题 | 停止/不支持条件 |
|---|---|---|---|
| 0，先复用产物 | 高分辨率原生 grid，现有共同 train-only PCA256/384，固定采样位置；不新增训练 | 教师重建是否保留任务信息；公共 256 子空间内学生是否更易预测 | 保留方差不够不能声称语义充分；缺下游标签时报告信息结论未定 |
| 1，优先一对 | 选定相同 PCA 维度、W、readout、训练时间轴；`e²` 对 `2β SmoothL1β` | 大残差是否支配辅助优化、是否影响 FM/任务读出 | 只看到 loss 数值变小；区域关键误差更差；规划无改善 |
| 2，条件触发 | 固定目标与 loss，只换最终轻量 MLP | readout 接口是否限制 W 的可读信息 | 只有头后 loss 改善，W probe/FM 无改善 |
| 3，低优先条件触发 | 固定维度与 loss，scalar-only 对 scalar+PHI-S | 优化参数化/多尺度是否有影响 | 单标量已经解释全部收益；不要再归因旋转创造信息 |

PCA256/384 对照沿用同一基底、固定教师图像和稀疏采样；不通过缩图降低教师细节。它与 W48/96 是不同变量，不做未经登记的全组合扫描。

对“辅助任务干扰主任务”，先测**真正同时收到两项非零梯度的共享参数**上的 `||g_aux||/||g_FM||`、`cos(g_aux,g_FM)`，以及残差分位区间的梯度占比。若 FM 对某 W 分支没有梯度，不能在该分支强算余弦。大 loss 不等于大共享梯度；正标量也不能改变梯度冲突的方向。只有证实长期尺度压制或负夹角，才由 Lead 另登记 GradNorm/PCGrad 单因素；本次未核查它们的官方实现，也不把它们作为自动叠加选项。

所有候选均保留原 FM 主任务、同样的全局分母、teacher detach、时间差分幅值、数据角色及部署单轨迹。没有新增第二套模型、没有更改部署输入分辨率，不能据此承诺 PDMS 提升。

## 6. 当前结论与真实证据缺口

已完成：全文与固定官方函数核验、本地 PCA/readout/loss 调用核验、可实施的单因素方案与反证条件。精确来源和文件 checksum 见 `SOURCE_MANIFEST.json`。

未验证：当前 256/384 保留的关键物体/边界信息、W48/96 的留出可预测性、FM/辅助共享梯度的尺度与冲突、上述候选的真实规划收益。没有这些证据，不能将高分辨率效果不足单独归因于“维度太大”，也不能把降低对齐损失等同于更好驾驶。

最小资源请求由 Lead 单独登记：读取已有固定 train/stage_val 的小样本教师/PCA/W 产物及既有曲线；先做无需新增教师计算的公共子空间/残差/梯度诊断，再决定唯一优先训练对照。本次本 agent 未运行新实验，未独立签发正确性 PASS。
