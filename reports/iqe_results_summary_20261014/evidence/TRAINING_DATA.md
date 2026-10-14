# 专家实际训练数据

本报告只整理已完成的固定实验，不修改筛选规则、模型、Scorer 训练或部署池。

分析身份 `882d2dee39f986897ed7d238900d5c40f3cc09a83c4deab13392b637f36fffc2`；完整运行产物 `/mnt/project/iqe-specialist-scene-analysis-20261014/base70k_pool_v1`。

## 数据角色

|角色|场景|log/source group|Base 已见记录|
|---|---:|---:|---|
|dev_report|1696|16|{'no': 1696, 'unknown': 0, 'yes': 0}|
|expert_fit|74311|845|{'no': 0, 'unknown': 0, 'yes': 74311}|
|scorer_fit|16581|211|{'no': 0, 'unknown': 0, 'yes': 16581}|
|selector_cal|1024|62|{'no': 1024, 'unknown': 0, 'yes': 0}|
|stage_val|1024|53|{'no': 1024, 'unknown': 0, 'yes': 0}|

专家数据只来自 D_E=74,311 场景/845 logs。D_S=16,581 场景/211 logs 未进入专家训练，但原 Base 训练见过 D_E/D_S。stage_val、selector_cal、dev 的已见标记来自既有数据登记，本轮没有重新定义这些角色。完整 Navtest 12,146 场景/136 logs 与研究数据按 scene/log 隔离。

## 部署池选中的三个动作分支

|分组|真实训练场景|logs|全部训练 updates|最终测试端点|普通 anchor|
|---|---:|---:|---:|---|---:|
|道路组|125|89|5000|3000 ema|0|
|交互组|116|90|5000|3000 raw|0|
|进度组|393|231|5000|1000 raw|0|

合计 634 个“专家×场景”样本，622 个不重复观测；道路/交互重叠 12 个，进度组与两组不重叠。它们是原有真实观测，未新增合成图像。三个选中模型均为私有条件读取层+独立动作头，使用 `offline_optimized` 目标；不是把原始 GT 直接改名为优化目标。

道路试点另有 E00/E10/E01/E11 四组对照，完全共用这 125 个场景。E00/E01 使用原 GT，E10/E11 使用优化目标；E01/E11 放开私有读取层。最终只选择 E11 加两个扩展专家，没有将四个试点都加入测试候选池。六次训练完成不等于六个部署专家。

## 实际筛选规则

- 交互组：Base 的 NC 或 TTC 不满分。
- 道路组：Base 的 DAC 或 DDC 不满分。DDC 在当前 PDMS 中权重为 0，方向改善单列。
- 进度组：不属于上述失败，且 Base PDMS < 0.9；其中也存在少量 comfort 非满分样本。
- 三组训练目标都要求优化目标来源在固定白名单内、几何/时间/坐标及历史严格几何合法、官方 PDMS ≥0.95、NC/DAC 满分、比当前 Base 至少高 0.02。
- 道路试点为了固定 2×2 同观测目标对照，另外要求原 GT ≥0.9 且安全合法、优化目标不低于 GT。另外两组按选定配方扩展，不强加道路试点的 GT 配对门槛。

这些是当前实验落盘的实际规则，不是本次看完测试结果后新增的门槛。历史其它 baseline “GT 只需高于 Base”的配方不能与本专项 2×2 的数据定义混称。

## 逐步筛选数量

|分组|累计筛选阶段|剩余场景|logs|
|---|---|---:|---:|
|道路组|all_Base_failure_type|1732|371|
|道路组|original_GT_legal_and_safe_PDMS_ge_0_9|1371|336|
|道路组|optimized_target_available|1371|336|
|道路组|source_whitelist|980|265|
|道路组|optimized_geometry_and_historical_strict|854|252|
|道路组|optimized_PDMS_ge_0_95_NC_DAC_full|698|228|
|道路组|optimized_minus_Base_ge_0_02|125|89|
|道路组|optimized_not_worse_than_GT|125|89|
|交互组|all_Base_failure_type|404|198|
|交互组|optimized_target_available|404|198|
|交互组|source_whitelist|166|119|
|交互组|optimized_geometry_and_historical_strict|125|97|
|交互组|optimized_PDMS_ge_0_95_NC_DAC_full|116|90|
|交互组|optimized_minus_Base_ge_0_02|116|90|
|进度组|all_Base_failure_type|14476|745|
|进度组|optimized_target_available|14476|745|
|进度组|source_whitelist|1794|495|
|进度组|optimized_geometry_and_historical_strict|1524|447|
|进度组|optimized_PDMS_ge_0_95_NC_DAC_full|393|231|
|进度组|optimized_minus_Base_ge_0_02|393|231|

因此 125/116/393 不是 Base 所有失败的数量。较大的限制来自优化目标来源、高质量/几何审计，以及原目标配对条件。未合格的原场景和审计证据保留，没有删除失败样本。

## 监督与实际曝光

|分组|Base 均分|原 GT 均分|优化目标均分|优化较 GT ≥1分|5000步平均每场景曝光|选中端点总曝光|
|---|---:|---:|---:|---:|---:|---:|
|道路组|18.24|98.30|99.25|39/125|5120.0|384000|
|交互组|55.62|99.39|99.61|8/116|5517.2|384000|
|进度组|86.38|86.59|97.89|392/393|1628.5|128000|

每组 5000×128=640,000 次样本呈现，采样为按 scene 均匀、有放回；没有额外 anchor、检索或全场景蒸馏。已重放实际 SceneSampler，逐项核对最终 sampler identity、消费量和 RNG 与完整 checkpoint 一致。逐场景/逐组曝光见 CSV，重复曝光不算新增数据。

原始输入仍为 Base 的三视图/当前状态；共享 VLM 冻结。只有专家的私有读取层、Query、Decoder 与必要轨迹头被训练。AdamW lr=1e-4、wd=0、warmup100、cosine5000至0.1峰值、FP32 EMA=.999。

优化轨迹来自现有 `context_clusters_snapshot_v1`，包含 GT 衍生、PDM 参考混合、地图中心线、corridor 与 parent construction 等来源。逐条 source/run/hash 保留；本次独立用官方环境重评分。没有加载历史动作教师或外部 Scorer 权重，但也不把历史优化过程笼统描述成完全没有外部 ancestry。

## 分组名称不等于能力

|分组|训练时 Base NC 失败|DAC 失败|TTC 失败|DDC 失败|comfort 非满分|
|---|---:|---:|---:|---:|---:|
|道路组|0|97|12|32|0|
|交互组|4|2|116|10|0|
|进度组|0|0|0|0|13|

例如交互组 116 个训练场景全部触发 TTC，而 NC 失败只有 4 个；不能把它描述为已学会普遍避免碰撞。本报告中的名称只是数据筛选标签。

逐条清单：`TRAINING_SCENES_AND_EXPOSURE.csv`；累计筛选：`TRAINING_FILTER_FUNNEL.csv`；完整来源/重叠/曝光：`TRAINING_DATA_AUDIT.json`。
