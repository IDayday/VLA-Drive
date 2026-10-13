# Two-candidate research baseline

本实验固定一个已训练的单轨迹 Query Base，只新增一个独立动作专家和一个从随机初始化训练的内置 Scorer。研究问题分别是困难场景专精、两策略互补性，以及 Scorer 能否实现 Oracle 的潜在收益。工程接通、专家留出改善和实际选轨收益分别报告。

使用 `python -m iqe.two_candidate`，复用 `iqe.internal_baseline`、原 `S0Adapter`、轨迹 IL、官方单轨迹评分和 IQE trainer。完整可执行命令见 [本轮 RUNBOOK](../../reports/iqe_two_candidate_baseline_v1_20261013/RUNBOOK.md)。

当前固定配置是 [base70k_v1.yaml](../../configs/iqe/two_candidate_baseline/base70k_v1.yaml)。实验身份为 `two_candidate_base70k_seed31415_v1`，输出目录为 `/mnt/project/iqe-two-candidate-baseline-20261013/base70k_v1`。`BASE_BINDING.json` 已于 2026-10-13 02:20:10 UTC 绑定：

- Base：`/mnt/project/iqe-remote-rolling-cache-20261012/continuation/step_070000.pt`，70,000 次 optimizer updates。
- SHA256：`30517484efcc8740c331f42f3f2ba2d1896a4f95cf70e2bf662e0d6c4484e8b0`。
- 来源 run identity：`1d6a94befc12ebb7a20dca21b9332bf6fbbae9f390d2f57e30600706fb77c8c6`。
- Base 推理沿用 FP32 visual、原 BF16 autocast、SDPA、关闭 TF32 的绑定契约。头部训练的执行设置另存 `HEAD_TRAINING_RUNTIME.json`。

原持续训练的 Base、22k 等已锁定实验以及正式 ACTIVE 均保持原身份。70k 的既有 Navtest 结果仅是选择快照的背景信息；本实验不读取 Navtest 场景、标签或失败列表。

用户随后明确扩大了更新预算：专家 10,000 updates、Scorer 20,000 updates，其中 Scorer 前 5,000 步为自然流阶段。版本化预算见 [training_budget_v2.yaml](../../configs/iqe/two_candidate_baseline/training_budget_v2.yaml)，仅包含三个更新数。使用 `amend-training-budget --budget-config` 将它接入独立、不可变的 `TRAINING_BUDGET_AMENDMENT.json`；原 YAML 中的 1000/2000/500 保留为初始绑定记录，不手改 YAML、`BASE_BINDING.json` 或缓存身份。下文表格描述 amendment 生效后的实际配方；应用命令见 RUNBOOK。

**数据先固定，再评分。**

`partition_fit` 在已有 `incremental_fit` 内，按完整 log/source-parent 的连通分组划分 D_E 与 D_S，比例约 80/20；随后各角色内按固定 hash 取预算。某个 log 未被预算选中的其他时间窗不会转入另一角色。`DATA_PARTITION.json` 和 `DATA_PARTITION_EXPOSURE.json` 保存实际清单及 hash。

| 角色 | 当前场景数 | 当前 logs/groups | 用途 |
|---|---:|---:|---|
| D_E / `expert_fit` | 16,000 | 832 | 失败发现、GT 审计、专家 H/N 训练 |
| D_S / `scorer_fit` | 4,000 | 205 | 冻结专家后的评分银行、Scorer 拟合 |
| `stage_val` | 1,024 | 53 | 专家与 Scorer checkpoint 选择 |
| `selector_cal` | 1,024 | 62 | 可选 single-margin，继承现有固定角色 |
| `dev_report` | 1,696 | 16 | 完整固定开发评估 |

这些数量来自本轮绑定记录，合计 23,744 个场景。D_E/D_S 均出现在 Base 自身训练清单中；D_S 的留出含义仅是“新增专家专精训练阶段未见”。绑定记录显示本 run 的训练清单不包含 stage_val、selector_cal、dev_report；这不意味着通用预训练模型没有接触过任何相关数据。

数据/metric-context manifest 的历史目录名包含 `formal_round1_22k`，这里只继承合法数据角色和官方上下文。本轮 W/A、规划特征、候选和分数身份全部绑定 70k，不能复用其他 Base 的语言/规划特征。

**模型与缓存边界。**

一次原 `encode_current` 得到当前观测产生的 W、action-query 状态 A 及原规划条件。Base 与新增专家读取相同冻结的规划特征；Scorer 读取 W、A、当前 ego 和实际物理轨迹。取 W/A 不增加 token，不更改 prompt/mask，不执行未来辅助预测头，也不输入 GT、真实分数或未来 actor 标签。

本轮缓存保存原始冻结 W/A、F_plan、ego、mask，以及独立的监督字段；不保存 Scorer 可训练投影后的 memory。缓存身份包括 Base、runtime、观测、schema 和代码。头部训练不构造 VLM；真实图像端到端推理仍执行原 VLM 一次。此缓存与持续 Base 训练的动态视觉池是不同产物，不依赖数 TB 的视觉缓存完成。

**唯一专家的固定配方。**

在 D_E 上，Base 分数低于 0.8，或有效 NC/DAC 未达到满分的场景构成 F。满分比较容差为 `1e-8`；缺失保护分项记为不可用。H 进一步要求 GT 的输入、坐标、时间和有限性审计通过，GT 分数至少 0.8、相对 Base 增益至少 0.05，并且 GT 的 NC/DAC 满分。GT 始终是监督目标，不进入部署候选。

N 从 D_E 的合法非失败普通样本中按固定 hash 选最多 2,048 个，排除目标问题；与 H 不重叠。sampler 以累计配额实现 80% H + 20% N，loss 不再乘这两个比例。固定 stage_val 上同规则的 F_V/H_V 在专家训练前锁定。空 H/N 不以更低阈值或普通样本替代。

深拷贝 Base 的 Query、Decoder 和必要轨迹头；只更新该独立专家，原 VLM、规划共享链和 Base 保持冻结/eval。继承原单轨迹 IL、normalizer、中间监督与原动作分支 dropout/DropPath。

| 参数 | 专家 | Scorer |
|---|---|---|
| optimizer | AdamW | AdamW |
| lr / weight decay | `1e-4` / `0.01` | `1e-4` / `0.01` |
| effective global batch | 128 | 128 scenes |
| 当前单 GPU microbatch | 32，accumulation 4 | 16，accumulation 8 |
| optimizer updates | 连续 10,000 | 连续 20,000 |
| warmup | 50，之后恒定 lr | 100，之后恒定 lr |
| grad clip | 1.0 | 1.0 |
| 固定验证/保存间隔 | 250 updates | 250 updates |

专家 checkpoint 按 stage_val Oracle@2 增益、H_V 直接收益、较早 step 依次选择。不用 D_S、cal 或 dev 决定专家 checkpoint；扩充只来自已记录的用户授权，不按 dev 分数自动增加预算。专精报告分别给出 H_train、H_val 的 `mean(r_expert-r_base)`、普通场景退化及 Base 参数/buffer/raw/physical 保持证据。

**Scorer 从零训练，两个采样流。**

W/A 分别经过自己的 LayerNorm+Linear，ego 经小型投影，统一到 256 维，同一场景投影一次。物理轨迹编码使用固定尺度 XY、sin/cos yaw 和时间。评分 Decoder 为两层、256 维、4 heads、dropout 0；候选内部时间 token 可自注意力，候选之间无 attention，也不输入专家 ID。

Scorer 的投影、轨迹编码器、Decoder、总分和分项输出头全部随机初始化；初始化种子 31415，权重来源记录在 `SCORER_INITIALIZATION.json`。不加载外部评分器或之前实验的 Scorer。新增参数和线上延迟由本实验实际测量。

主 loss 为 `SmoothL1(beta=0.1) + 0.2*component_BCE + 0.2*ranking`。自然流只在 D_S 均匀抽场景，用于 value/component。比较流对有效且 `abs(r1-r0)>0.01` 的场景，按 `q=abs(r1-r0)/sum(abs(r1-r0))` 抽样，计算普通 pair softplus 均值，温度 0.1；不再次乘 gap，也不在场景内部将 gap 约掉。缺标签保留 mask，无比较信号明确报告，不能伪造 pair。

- 前 5,000 updates：每步 128 个自然场景，不用 ranking。
- 后 15,000 updates：每步 64 个自然场景 + 64 个比较流场景。比较流不混入 value/component 回归。
- 第 5,000 步只发生同一网络内部的采样/loss 阶段切换，不暂停进程，不重启 optimizer 或 scheduler。

Scorer checkpoint 按 stage_val 原始 argmax 的真实 selected PDMS、新增 NC/DAC 违规数、较早 step 选择。候选始终为同一 Base 与已选定唯一专家。两候选每条独立通过固定参考上下文的官方 NAVSIM v1.1 单轨迹评分，异常不会变为有效零分。

**主结果与附加 margin 分开。**

主规则严格为 `pred_expert > pred_base` 才选专家，同分保留 Base，不混入分项阈值。附加 margin 仅在 selector_cal 的 `{0, 0.01, 0.02, 0.05, 0.1}` 网格选择，预设新增 NC/DAC 违规和高分转零分预算均为零；无非平凡可行收益时保留 Base，不能把这种回退称为 Scorer 成功。

dev 同一完整清单报告 Base、Expert alone、Oracle@2、scratch Scorer argmax、可选 margin。令 A 为专家正增益均值，B 为专家负增益绝对值均值，R 为 selection regret，则验证 `expert-base=A-B`、`oracle-base=A`、`oracle-best_fixed=min(A,B)`、`selected-best_fixed=min(A,B)-R`。同时报告大 gap 错选、正/负替换代价、高分转零分、NC/DAC 变化，以及固定 seed 的 2,000 次 log 配对 bootstrap。16 个独立 dev groups 和单训练 seed 的边界必须保留。

最新执行要求是专家连续 10,000 步、Scorer 连续 20,000 步。本轮不安排人为暂停、恢复实验或专门真实 resume probes；已有恢复能力只用于意外中断，不能把未执行的恢复试验写成当前真实证据。

文档编写时，特征提取与 CPU 评分仍在进行，尚无本轮科学结果；`expert/best.pt`、`scorer/best.pt` 和最终报告是依赖阶段完成后的产物。研究 bundle 不设置 ACTIVE，不产生新的 Navtest 任务。
