# PlanReg-WM-V1 综合算法审计

审计日期：2026-09-05

训练实现：`be066973`（续训使用的代码版本）

审计分支：`audit/planreg-wm-v1-comprehensive-20260904`
固定 scorer 来源：`valeoai/DrivoR@fc6e5aa144bbcb5a046e22c18f1bd5cf3af8634a`

## 结论先行

这版模型并非“只有 loss 在下降但没有学会任务”。现有证据表明：

1. 64-query generator 已学会覆盖可行轨迹；续训终点 epoch33 在完整 Navtest 的离线 `Oracle@64=0.98820`，相对 epoch27 的 `0.98724` 提高 `0.000957`；
2. scorer 确实学会了候选质量排序，但仍是当前部署性能的主要瓶颈：epoch33 选中 `PDMS=0.91333`，相对同一候选集 Oracle 的 regret 为 `0.07487`；
3. future-register predictor 确实学会了动作条件和真实未来，而不只是复制当前帧；
4. planning/scorer/WM 梯度能按设计进入 planning registers 和 24 层视觉 Q/V LoRA，冻结 LLM 没有梯度；
5. DrivOR 同源 scorer 的结构、输出、聚合、TTC mask 和梯度路由保持严格一致；
6. 续训 6 个 dataset epochs 没有带来部署收益：selected PDMS 相对 epoch27 为 `-0.000459`，95% 配对区间跨 0；Oracle 的小幅提升被更差的 scorer regret 抵消。

但同时存在四个会实质限制性能或破坏原科学意图的问题：

1. **EMA teacher 以 BF16 保存并原位更新，后期 EMA 更新几乎全部被量化吞掉。** 续训 4,842 step 后，teacher 的视觉 LoRA 有 `99.927%` 元素逐位未变化；以当前 momentum 模拟下一次更新时 `99.9999%` 元素不变。这是 P0 数值问题。
2. **semantic Q-Former 的 16 个输出 token 几乎完全塌缩。** 实测 token 间 cosine 约为 `1.0`，planning→semantic cross-attention 等于严格均匀分布；semantic 分支事实上是一个广播到所有 planning slots 的全局偏置，而不是 16-slot 语义补充。
3. **4 个 refinement 中前 4 个 trajectory heads 没有轨迹监督。** `prev_weight=0` 使 loss 循环最终只保留最后一阶段，head0–3 的变化几乎只有 AdamW weight decay；约 5.37M 输出头参数没有学习对应轨迹任务。
4. **64 条候选的有效多样性不足。** 无精确重复，但以 trajectory RMS 0.5 聚类时平均仅约 `2.94` 个有效簇；同时 `inter_weight=0`，没有任何显式 coverage/diversity 约束。

因此，当前架构的主要方向是可行的，但不能把它视为已经充分利用了 semantic tokens、spatial tile aggregation 和在线 EMA world target 的最终版本。现有 epoch33 权重也不值得在同一协议下继续盲目延长训练；下一轮应先修复上述结构/数值问题，再做等预算对照。

## 审计口径与可复现性

### 权重

- epoch27 full checkpoint：`f1182013c03f57afde8bb586240fb5c0465c3892a6a050ca38fd5b630f89c753`
- continuation epoch33 full checkpoint：`7811a877b25aff74c71cd46cf35816a7bd3e6c84bb2a2fc18ab3ccae957cee3b`
- continuation student-only checkpoint：`d6b6da75034437d0081f111a0cd5da601c070035cff83652df2ac760caf02de1`
- shared random planning-stack init：`2508524f12acba9faf615c1b63c31837412d7403065ad6f9eda3d13311929629`

### 训练与评测协议

- 训练集：完整 trainval，103,288 scenes；sampler 每 epoch padding 8 个 sample slots；
- 正式训练：16 GPU × 8/GPU，global batch 128，807 optimizer steps/epoch，27 epochs，共 21,789 steps；
- 续训终点：global step 26,631，相对 epoch27 多 4,842 steps，即 6 个 dataset epochs；
- Navtest：12,146 scenes、136 logs、64 candidates/scene；
- 模型推理：FP32，student-only，只有当前单前视图，不读取未来帧；
- Oracle：只是在冻结的同一组 64 候选上离线取官方 PDM 最大值，是不可部署的上限；
- 置信区间：按 log 聚类 bootstrap，而不是把相邻 scene 当作独立样本。

完整 64-candidate 官方 PDM 使用可恢复分片批计算。全量 12,146-scene 首次运行暴露了 `ProcessPoolExecutor.map` 一次性提交任务导致 wakeup pipe 填满的问题；评分脚本现改为 `chunksize=8` 分块提交，复用已完成分片后全量评分耗时 `2m45s`，没有重跑 VLM 推理。

本模型只有单前视图 InternVL；DrivOR 论文结果使用其自身的四相机视觉输入和训练协议。两者的绝对 PDMS 不是严格公平对比，本审计只复用其 scorer 源实现与 attention 可视化方法，不把论文数字当作本模型的直接控制组。

模型共有 28,695,423 个可训练参数和 2,396,488,448 个冻结参数，可训练比例约 `1.18%`。可训练预算中 generator 占 38.13%、scorer 21.22%、semantic Q-Former 16.07%、future predictor 11.48%、视觉 Q/V LoRA 10.96%、planning adapter 1.21%、fusion 0.92%。其中未受轨迹任务监督的 head0–3 合计 5,365,856 参数，占全部 trainable 的 `18.70%`。

### scorer 同源门禁

固定来源审计全部通过：

- 6 个 component logits 最大差异：0；
- 聚合 PDM 最大差异：0；
- selected indices：完全一致；
- 独立 scorer decoder：4 层；
- scorer-only backward：proposal gradient 为 0/None，scene feature gradient 非零；
- `target TTC == 2.0` invalid mask：通过。

本报告将其称为“本模型 scorer”；“DrivOR”只表示固定代码来源或论文可视化方法。

## 模块是否学会了对应任务

| 模块 | 任务证据 | 判断 | 主要问题 |
|---|---:|---|---|
| InternViT Q/V LoRA | 24/24 层 Q-A/Q-B/V-A/V-B 在 WM-only backward 中均为非零梯度；从共享初始化明显移动 | 学到了 | 续训期间仅再移动相对 L2 `0.00129`；EMA 副本数值失效 |
| planning register 参数 | 参数本身有效秩约 10.29；有 planning/WM 梯度 | 学到了 | 场景输出有效秩仅约 5.04，视觉注意力仍高度冗余 |
| tile aggregator | tile attention 权重本身有 slot 差异 | 结构在工作 | gate 后 crop residual 只有 thumbnail RMS 的约 2.96%，非 thumbnail 基本被关闭 |
| semantic Q-Former | 参数从初始化移动；Q-Former 有梯度 | 参数更新了，但 token 任务未形成 | 16 token 几乎相同，cross-attention 完全均匀 |
| planning-primary fusion | 去掉 semantic context 会改变 proposal/scorer | 有影响 | 影响退化成约 13.7% RMS 的同一全局语义向量广播 |
| trajectory decoder | Oracle@64 很高，最终轨迹 loss 大幅下降 | 学到了 | 只有最终 head 学轨迹；候选簇数低 |
| scorer | 完整 Navtest 排序相关和 selected PDMS 显著有效 | 学到了但不充分 | EP/TTC 是主要 regret；scene 内 rank correlation 仅中等 |
| future predictor | 正确动作明显优于 no-action、shuffled-action 和 current-copy | 学到了 | teacher 后期几乎固定；未证明 WM 对 PDMS 的因果增益 |
| frozen LLM | WM-only 梯度张量数为 0 | 边界正确 | 语义下游 token collapse 发生在 Q-Former/融合侧 |

## 轨迹生成任务

epoch27 完整 Navtest 的 proposal bank：

- `Oracle@64 = 0.987239`；
- candidate mean = `0.775227`；median = `0.940628`；
- P10 = `0.0`，P25 = `0.806359`；
- `75.34%` 候选高于 0.8，`60.50%` 高于 0.9；
- exact duplicate fraction = `0`；
- 64 个 query 都至少被 scorer 选中过，selection effective query count = `58.11`；
- 但 trajectory RMS 0.5 下 effective clusters 平均仅 `2.94`。

这说明 generator 已经能覆盖高质量解，但“64 条”不等于“64 个独立行为模式”。高 Oracle 与低有效簇可同时成立：多数场景只需要少数局部轨迹族，许多 query 在同一轨迹族内做小扰动。当前 `inter_weight=0`，因此训练目标只要求至少一条接近 GT，不奖励覆盖不同可行模式。

此外，final min-of-64 imitation loss 每个样本只把轨迹梯度给当前最接近 GT 的一个 query；其余 63 个 query 在该样本上没有 imitation 梯度。scorer 前的 `proposal.detach()` 又有意阻断了 PDM/scorer loss 对坐标的回传。因此 candidate mean/P10 明显弱于 Oracle 是训练目标的直接结果，而不只是 scorer 选错：generator 只被要求“至少产出一条像 GT 的轨迹”，没有被要求让全部候选安全、可行或互补。

### refinement supervision 缺口

loss 递推为：

```text
trajectory_loss = prev_weight * trajectory_loss + current_min_loss
```

正式配置 `prev_weight=0`，5 个 proposal stages 中只有最后一个 `traj_head.4` 对最终 loss 有贡献。checkpoint 审计显示：

- head0–3 从初始化的 relative L2 变化约 `0.01765`，init/current cosine `0.999881`；每个 head 有 7/10 tensors 逐位未变；
- head4 relative L2 变化约 `0.53014`，cosine `0.88146`；
- 续训期间 head0–3 仅再变化约 `0.000654`，与 weight decay 的均匀收缩一致；head4 再变化 `0.02235`。

因此 decoder 中间层仍可通过最终 head 的计算图学习，但 head0–3 本身没有学会“中间轨迹预测”。如果不需要 deep supervision，应删除这些 dead output heads；如果希望每级 refinement 都可解释，应给中间级明确监督。

## scorer 学到了什么

### 完整 Navtest 选择质量（epoch27）

- selected PDMS = `0.913788`；log-cluster 95% CI `[0.905960, 0.921397]`；
- offline Oracle@64 = `0.987239`；95% CI `[0.984839, 0.989516]`；
- scorer regret = `0.073452`；95% CI `[0.067048, 0.080121]`；
- scene 内 Spearman：mean `0.41566`，median `0.51640`；
- 全部 777,344 candidates 的 global Spearman = `0.54428`；
- scorer 所选候选的真实官方 rank：mean `13.46`，median `3`；
- Oracle>0.9 但 selected<0.5 的灾难性误选比例：`3.16%`；
- scorer 预测 selected score 均值 `0.93797`，真实 selected PDMS `0.91379`，平均高估 `0.02418`。

scorer top-k 中能找到的最佳官方 PDM：

| k | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| best PDM | .9138 | .9261 | .9383 | .9493 | .9618 | .9752 | .9872 |

这条曲线说明 scorer 并非随机：前几名已经显著富集高质量候选；但 top-1 与 top-64 之间仍有大空间。

### 六个 component 任务

| component | 指标 | 结果 | 解释 |
|---|---:|---:|---|
| Comfort | AUROC / Brier | .99979 / .00335 | 几乎学满，且正例率 96.39%，任务偏易 |
| DAC | AUROC / Brier | .93164 / .05542 | 已学会 |
| DDC | AUROC / Brier | .95796 / .03140 | 已学会，但推理聚合权重为 0 |
| NC | AUROC / Brier | .88656 / .04556 | 有效但仍弱 |
| TTC | AUROC / Brier | .87299 / .08485 | 最弱的安全分类头之一 |
| EP | global / scene Spearman | .61161 / .51198 | 能排序，但误差仍主导选错 |

selected 相对 Oracle 的 component gap：

- EP `-0.09448`；
- TTC `-0.05096`；
- DAC `-0.01770`；
- NC `-0.01523`；
- Comfort `+0.00058`；
- DDC `+0.01079`。

因此当前 scorer 瓶颈主要是 **EP 和 TTC**，不是 Comfort。训练时六个 BCE 等权，而推理时聚合权重为 `NC/DAC/DDC/TTC/EP/Comfort = 1/1/0/5/5/2`。这与固定 DrivOR 来源一致，但从本模型的优化效率看存在目标错配：一个推理权重为 0 的 DDC 和近乎饱和的 Comfort 仍获得与 EP/TTC 同量级的直接训练预算。

## 世界模型是否学会未来和动作条件

在 12 个分层抽取的 held-out Navtest scenes 上，以真实未来帧只做诊断（部署推理不读未来），correct prediction 的 cosine loss 为：

| horizon | 0.5 s | 1.5 s | 3.0 s |
|---:|---:|---:|---:|
| correct | .03868 | .06827 | .08885 |
| no action | .06539 | .16689 | .18131 |
| shuffled action | .08405 | .23611 | .30236 |
| copy current register | .06633 | .15705 | .22747 |
| correct prediction vs shuffled future target | .46855 | .45395 | .44278 |

由此得到：

- correct 相对 no-action 的收益：`.02671 / .09862 / .09245`；
- correct 相对 shuffled-action 的收益：`.04536 / .16784 / .21350`；
- correct 相对 current-copy 的收益：`.02764 / .08878 / .13861`；
- correct target 相对 shuffled target 的 margin：`.42987 / .38568 / .35392`。

逐 scene 胜率也支持同一结论：correct 在 1.5 s 和 3.0 s 几乎总是优于动作/复制控制。WM-only backward 中：

- future predictor grad norm = `0.41829`；
- planning register parameter grad norm = `0.24733`；
- current register output grad norm = `0.01217`；
- 24/24 视觉层的 Q-A/Q-B/V-A/V-B 梯度全部非零；
- scorer、semantic Q-Former、LLM 的 WM-only 梯度均为 0。

结论是：**predictor 学会了动作条件未来任务，梯度边界也正确。** 但尚无相同初始化、相同训练预算的正式 no-WM 对照，因此不能从这些数字推出“WM 提高了最终 PDMS”。

还要区分“predictor 学会”与“WM 强力塑造视觉表征”。后期 `wm_loss≈0.059`、权重 0.1，所以 `weighted_wm_loss≈0.0059`，仅约总 loss 的 `0.5%`。12-scene WM-only 审计的视觉 LoRA 全局梯度范数约 `0.309`，乘正式权重后约 `0.031`；同期训练记录的视觉 LoRA 总梯度中位数约 `2.75`。register 参数对应约 `0.0247` 对 `2.84`。虽然不是同一 batch 的严格梯度分解，但量级上表明 driving/scorer loss 对视觉表示的作用约强两个数量级，WM 很可能主要训练了 predictor，本身对 register/LoRA 的塑形较弱。

在完全相同的 12 个场景上复测 epoch33，correct cosine loss 为 `.03889 / .06846 / .08790`。相对 epoch27 的变化仅 `+0.00021 / +0.00019 / -0.00095`；动作控制、错配未来 margin 和梯度范数也基本不变。这证明续训没有让世界模型任务继续取得可辨别的进展，与训练曲线中 WM loss 平台一致。

### P0：BF16 EMA teacher 数值失效

EMA teacher 参数实际 dtype 是 BF16。global batch 128 下 momentum schedule 为：

- start `0.9684444339`；
- end `0.9992002799`。

epoch27 终点按真实 BF16 原位公式模拟下一次更新，不发生任何逐位变化的比例：

- planning register tokens：`100%`；
- register neck：`99.9837%`；
- tile aggregator：`99.9970%`；
- vision Q/V LoRA：`99.9928%`。

续训终点更严重：register tokens 和 neck 为 `100%`，vision Q/V LoRA 为 `99.9999%`。从 epoch27 到 epoch33 的 4,842 个 optimizer steps 中，teacher 参数逐位完全不变的比例为：

- planning register tokens：`99.500%`；
- register neck：`99.641%`；
- tile aggregator：`99.896%`；
- vision Q/V LoRA：`99.927%`。

续训又因 resume-safe clamp 将 momentum 固定在之前的终点 `0.9992002799`，teacher 几乎成为 fixed target。正确修复是保存 **FP32 master EMA**（或 FP32 accumulator），再在 forward 时转换到计算 dtype；仅改变日志精度不能修复 teacher。

## planning / semantic / tile 表征审计

### planning scene tokens

完整 Navtest：

- scene-token effective rank mean = `5.038`（16 tokens，中心化后的理论最大秩为 15）；
- pairwise cosine mean = `0.423`，P90 = `0.867`；
- 前 5 个奇异值能量合计约 `94.8%`。

注意：可训练的原始 register 参数本身 effective rank 约 `10.29`、pairwise cosine 约 `-0.022`，所以退化主要发生在图像编码/聚合/融合后的 scene representation，而不是 register 参数初始化本身。

### semantic Q-Former 与融合

4 个代表场景的增强诊断显示：

- semantic token pairwise cosine mean ≈ `1.0`；
- token-centered RMS 仅约 `7e-6–1e-5`，而 token 总 RMS 约 `0.87`；
- planning→semantic cross-attention entropy = `2.7725887 = ln(16)`；
- normalized entropy = `1.0`；
- max attention probability ≈ `0.062503`，即 `1/16`；
- 离均匀分布平均绝对偏差仅约 `1e-6`；
- semantic gate 从初始化概率 0.20 学到 `0.15738`；
- gate 后 semantic context RMS 约为 planning RMS 的 `13.70%`。

根因与 `scene_embeds ~ N(0, 1e-6)` 的极小对称性破缺一致：Q-Former 对 query slot 是 permutation-equivariant 的，几乎相同的 query 很难自行分化。当前 cross-attention 虽然代码路径存在，但没有获得 slot-specific semantic context。

该分支并非完全没有场景信息：不同场景的 semantic-context RMS 会变化，关闭它也会改变轨迹；问题是它只形成一个 scene-dependent global vector。当前为得到这个全局向量仍执行完整冻结 LLM forward，计算成本与表达能力不匹配。若增大 query identity 后仍不能形成 slot-specific token，应直接比较更便宜的 global pooling/单-query 方案。

### spatial tile aggregation

每个 register 对 8 个 crop tiles 的 attention 有明显差异，说明 attention 模块本身不是常数；但：

- `mean(abs(tanh(tile_gate))) = 0.02422`；
- gate 后 tile residual / thumbnail RMS = `2.96%`；
- 去掉所有非-thumbnail tiles，proposal RMS 仅变化 `0.01854`，predicted score RMS 仅变化 `0.00863`，12 个场景只改变 1 个 selected index。

所以正式模型虽然名义上使用 thumbnail-query tile attention，功能上接近 thumbnail-only。

## 仿 DrivOR 的 attention 可视化

方法严格来自固定 commit 的：

- `scripts/viz/attention_maps_viz.ipynb`；
- `scripts/viz/build_cosine_similarity_maps.py`。

实现了三种同源视角：

1. 最后一层所有 heads 平均；
2. 最后三层 `M = 0.9A + 0.1I` shallow rollout；
3. 每个 register 选择最低熵 head。

针对本模型的必要适配是：DrivOR 的四相机 DINO registers 改为单前视图 InternViT 内部 indices `1:17` 的 registers，patch 从 index 17 开始；图上使用 thumbnail。scorer/generator→register→patch 的乘积只作为 heuristic attribution，不是因果解释。

观察：

- 最后一层 register attention map 的平均两两 cosine 约 `0.748`，normalized entropy 约 `0.906`，总体偏冗余且偏分散；
- 最后三层 rollout 的 16 张图几乎相同，说明跨层累计后 register 专门化很弱；
- 最低熵 head 能看到少数 slot 对右侧建筑、路面或边界的差异，但多数 slot 仍相似；
- 多个场景的高响应落在天空、图像边缘或近车路面，并未形成稳定、清晰的“车/行人/车道/信号灯”分工；
- highest-regret 场景里 scorer 和 generator 的 heuristic composite 几乎相同，未显示 scorer 独立抓住了能纠正误选的视觉区域。

下游也没有均匀利用 16 个 scene slots：4 个代表场景中，所选轨迹的 scorer attention 对应约 `4.64–8.39` 个有效 registers；generator attention 对应约 `1.13–3.86` 个，其中 hardest-oracle 场景几乎只依赖一个 slot。这与 scene-token effective rank 约 5 的统计一致。

epoch33 在同一 4 个场景上的配对可视化几乎没有改变：最终层 register-attention 两两 cosine 从 `0.74848` 变为 `0.74828`，normalized entropy 从 `0.90621` 变为 `0.90628`；scene-token effective rank 均值仅从 `4.8076` 变为 `4.8119`。semantic cross-attention 仍精确接近均匀分布，semantic-context slot-centered RMS 仍只有约 `1e-7`。这排除了“续训已经悄悄解决 token 分化，只是 loss 看不出来”的可能。

关键图：

- [epoch27 全部四场景可视化目录](baseline_epoch27/attention_v3/)
- [attention depth summary](baseline_epoch27/attention_v3/attention_depth_summary.png)
- [highest-regret last-layer maps](baseline_epoch27/attention_v3/09d424ddf3a558b3/register_to_patch_last.png)
- [highest-regret shallow rollout](baseline_epoch27/attention_v3/09d424ddf3a558b3/register_to_patch_shallow_rollout.png)
- [lowest-entropy heads](baseline_epoch27/attention_v3/09d424ddf3a558b3/register_to_patch_lowest_entropy_head.png)
- [scorer/generator composite](baseline_epoch27/attention_v3/09d424ddf3a558b3/selected_trajectory_composite_attention.png)
- [tile attention](baseline_epoch27/attention_v3/09d424ddf3a558b3/tile_attention.png)
- [epoch33 全部四场景配对可视化目录](continuation_epoch33/attention_paired_epoch27_tokens/)
- [epoch33 paired attention depth](continuation_epoch33/attention_paired_epoch27_tokens/attention_depth_summary.png)
- [epoch33 highest-regret composite](continuation_epoch33/attention_paired_epoch27_tokens/09d424ddf3a558b3/selected_trajectory_composite_attention.png)

attention 不是因果证据；上述结论与 pathway intervention、token geometry 和 checkpoint movement 联合使用，而不是只凭热力图下结论。

## 路径干预：模型实际依赖什么

在 12 个任务分层场景中，原权重重放与 candidate bank 的 proposal/score 最大差异均为 0。下表给出关掉路径后的敏感度：

| 干预 | proposal RMS change | predicted-score RMS change | selected index 改变 |
|---|---:|---:|---:|
| 全部 ego status 置零 | 8.501 | .408 | 100% |
| 仅 ego velocity 置零 | 8.171 | .371 | 91.7% |
| navigation command 置零 | 1.271 | .234 | 50.0% |
| 16 planning slots 压成同一均值 | .834 | .271 | 83.3% |
| semantic context 置零 | .197 | .105 | 41.7% |
| 去掉非-thumbnail tiles | .0185 | .0086 | 8.3% |

这说明模型强依赖 ego motion，确实利用 planning slot 差异，也会使用导航和 semantic 分支；crop tile 路径则几乎没用。由于这些是 out-of-distribution intervention，而且样本刻意包含高 regret/困难场景，干预后的 PDMS 均值偶尔提高不能解释为“该模块有害”。敏感度可以证明路径被使用，但不能单独证明路径贡献为正。

epoch33 在按其自身失败类型重新分层的 12 个场景上得到相同依赖顺序：全部 ego status / 速度 / 导航 / slot-collapse / semantic-off / non-thumbnail-off 的 proposal RMS change 分别为 `9.734 / 9.017 / 1.741 / .672 / .154 / .0178`，selected-index change 分别为 `100% / 83.3% / 83.3% / 91.7% / 41.7% / 0%`。两次抽样 token 不同，因此不把数值差当作训练增益；能确认的是续训没有改变“强依赖 ego、弱依赖 crop tiles”的结构性结论。

## 训练动态与续训是否仍在学习

从最初训练到 step 26,599 的 TensorBoard 记录：

- total loss early/late：`6.5865 → 1.1745`；
- trajectory loss：`4.5414 → 0.5197`；
- scorer loss：`2.0365 → 0.6490`；
- WM loss：`0.2158 → 0.05746`；
- register pairwise cosine：`0.6385 → 0.4307`；
- register effective rank：早期均值 `6.31`，后期稳定约 `5.01`；
- action/scorer/register/vision-LoRA/predictor 的所有已记录梯度均 finite 且非零。

但 epoch27 尾部与续训末尾相比：

- total loss：`1.198 → 1.211`，基本不变；
- trajectory loss：`.504 → .555`，未继续稳定下降；
- scorer loss：`.688 → .650`，仍有小幅改善；
- WM loss：`.05917 → .05909`，完全平台；
- register geometry 基本不变。

续训期间各模块相对 epoch27 checkpoint 的直接变化：

- scorer `3.08%`；semantic fusion `2.44%`；final trajectory head `2.24%`；
- action generator 整体 `1.71%`；Q-Former `1.49%`；future predictor `1.32%`；
- planning adapter `0.35%`；vision Q/V LoRA 仅 `0.129%`。

因此续训不是完全没有更新，但主要在 scorer/fusion/final head 上做低学习率调整；视觉表示和 WM 已基本饱和。完整配对 Navtest 最终确认：Oracle 小幅提高 `0.000957`，selected PDMS 却降低 `0.000459`，所以训练尾部 scorer loss 的小幅下降没有转化为真实选择性能。

## 风险分级与建议

### P0：下一次正式训练前必须修

1. **FP32 EMA master**：teacher 参数/accumulator 保持 FP32；checkpoint 明确保存 FP32 teacher；forward 前按需 cast。增加“实际 EMA update 非零比例”和 teacher drift 日志。
2. **修复 semantic token 对称塌缩**：将 `scene_embeds` 初始化尺度提高到常规 query embedding 范围（如 std 0.02），或加入明确的 slot positional identity；测试 cross-attention entropy、slot centered RMS 与有效秩。
3. **明确 refinement 设计**：若保留 5 个 heads，给各 stage 非零 deep supervision；否则删除前 4 个输出 heads，避免把 weight-decay movement 误认为学会。

### P1：最可能提升 PDMS

1. 提升 proposal coverage，而非简单增加 query 数：加入 generator-only diversity/coverage 目标、分模式 query 或 endpoint/curvature 分层。保持 scorer 源代码不变即可做受控实验。
2. 将优化预算集中到 EP/TTC 难例。直接改六头 loss 权重会偏离严格 DrivOR 同源，建议先用 hard-example sampling、训练样本重加权或单独的 parity-off 实验验证。
3. tile aggregation 要么提高 gate 的可学习性并加利用率诊断，要么承认当前数据下 thumbnail-only 更经济；现状承担 8 crop 的视觉计算，却只获得约 3% 的 gated residual。
4. 对 planning scene tokens 加轻量 variance/covariance 或 slot diversity 约束，并观察有效秩与 Oracle 是否共同提升，避免只追求热力图变“漂亮”。

### P2：科学验证缺口

1. 用同一 shared init、相同数据顺序和总 steps 做 WM/no-WM 配对，才可证明 WM 对 PDMS 的净贡献；
2. 至少 3 seeds 报告 log-cluster CI；
3. 将 highest-regret scenes 按 EP/TTC/NC/DAC 失效类型分组，评估修复是否针对真正瓶颈；
4. attention 可视化继续保留，但以 intervention 和任务指标作为主证据。

## 当前明确没有实现的内容

- 没有 multi-trajectory consequence modeling；
- predictor 训练仍是 `K=1` 的 GT trajectory 条件；
- scorer 不读取 predicted future registers；
- 没有 candidate-specific future image；
- 没有 future RGB reconstruction、CEM 或 TOAD。

## 证据文件

- [epoch27 candidate/scorer task audit](baseline_epoch27/candidate_task_audit.json)
- [world-model task audit](baseline_epoch27/world_model_task.json)
- [pathway interventions](baseline_epoch27/pathway_interventions.json)
- [attention audit](baseline_epoch27/attention_v3/attention_audit.json)
- [checkpoint learning through epoch33](checkpoints/checkpoint_learning_epoch33.json)
- [EMA continuation numerics](checkpoints/ema_continuation_numerics.json)
- [training dynamics](training/training_dynamics.json)
- [scorer parity report](../drivor_scorer_parity.json)

## continuation epoch33 完整 Navtest

完整推理和官方 PDM 批评分均已完成：12,146 scenes、每场景 64 条轨迹、当前单帧输入、FP32 action/scorer 路径。scored candidate bank SHA-256 为 `e5cc0e20dbe258a57db4166ff7a1204589b7b83561dee50ae2d5cb6fba07cf58`。

### epoch33 绝对结果

- selected PDMS：`0.913328`，log-cluster 95% CI `[0.905825, 0.920627]`；
- offline Oracle@64：`0.988196`，95% CI `[0.986045, 0.990213]`；
- scorer regret：`0.074868`，95% CI `[0.068582, 0.081423]`；
- candidate mean / median / P10 / P25：`.775211 / .940906 / 0 / .813540`；
- scene/global Spearman：`.41432 / .54246`；
- selected true rank mean / median：`13.52 / 3`；
- scorer 预测的 selected score 均值 `.94157`，真实 selected PDMS `.91333`，高估 `.02824`；
- Oracle>0.9 且 selected<0.5 的灾难性误选比例 `3.41%`。

### 与 epoch27 的严格配对差异

两个 bank 使用完全相同的 12,146 tokens、官方 metric cache 和 64-candidate 定义。逐 log 聚类 bootstrap：

| 指标 | epoch27 | epoch33 | 配对变化 | 95% CI | P(变化>0) |
|---|---:|---:|---:|---:|---:|
| selected PDMS | .913788 | .913328 | `-.000459` | `[-.002481, .001436]` | .328 |
| offline Oracle@64 | .987239 | .988196 | `+.000957` | `[+.000253, +.001702]` | .996 |
| candidate mean | .775227 | .775211 | `-.000016` | `[-.002446, .002439]` | .476 |
| scorer regret reduction | — | — | `-.001416` | `[-.003489, .000498]` | .074 |

这里 regret reduction 为负表示 regret 变大。续训确实改变了模型：query-aligned proposal RMS change `.2374`、endpoint RMS change `.5590`、predicted-score RMS change `.0780`，且 `57.34%` 场景更换了 selected query；但这些变化没有提高部署指标。

selected component 的配对变化为：EP `+.00231`、DAC `+.00115`、DDC `+.00091`、NC `-.00202`、TTC `-.00395`、Comfort `0`。由于 TTC 和 EP 在聚合中权重均为 5，TTC 的退化抵消了 EP 的小幅提升。scorer 子任务也显示相同趋势：EP global Spearman 从 `.61161` 降到 `.60317`；TTC Brier 从 `.08485` 增至 `.08745`、ECE 从 `.04044` 增至 `.04575`。因此 scorer loss 从约 `.688` 降到 `.650` 不能解释为部署排序能力改善。

generator 的高质量候选比例略有改善：`PDM>0.8` 从 `75.34%` 到 `75.69%`，`PDM>0.9` 从 `60.50%` 到 `61.07%`，P25 从 `.80636` 到 `.81354`。但 candidate mean 不变，有效簇数反而从 `2.941` 降到 `2.803`。Oracle 的统计显著提升来自少数场景的尾部覆盖，而不是整个 bank 均匀变好。

planning representation 同样基本静止：effective rank `5.0383→5.0413`，pairwise cosine `.42344→.42252`，semantic gate `.15738→.15828`，tile gate 近乎不变。结合 WM 和 attention 配对审计，最终判断是：**这 6 个续训 epochs 已处于平台期；继续沿用同一结构和目标延长训练的预期收益很低。**

证据：

- [epoch33 scorer/task audit](continuation_epoch33/candidate_task_audit.json)
- [epoch27→epoch33 paired comparison](continuation_epoch33/paired_vs_epoch27.json)
- [epoch33 world-model paired task audit](continuation_epoch33/world_model_task_paired_epoch27_tokens.json)
- [epoch33 pathway interventions](continuation_epoch33/pathway_interventions.json)
- [epoch33 attention audit](continuation_epoch33/attention_paired_epoch27_tokens/attention_audit.json)
