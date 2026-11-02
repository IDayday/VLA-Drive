# Qwen-World RL：当前代码、实验设计、结果与结论

Snapshot: **2026-11-02 15:04:13 UTC**. This publication consolidates the current research branches without changing a running learner or evaluator.

- [可离线阅读的完整HTML报告](report.html)：当前实验 → 最新结果 → 评测队列 → 论文/本地配置 → 详细分析。图表内嵌；未随包发布的NAS链接已标注，不作为离线可用链接。
- [全部完整分数CSV](results/scores.csv)：151个完整NAVTEST点，包含明确标记的native0初始函数校准点。
- [完整曲线与配对分析](results/latest_complete_review.json)：132组已有对照；[逐项配对原始JSON](results/paired)。
- [14份实际锁定配方](experiments)、[逐文件配置提取](analysis/recipe_configurations/LOCAL_CONFIGURATION_COMPARISON.json)、[11篇论文配置及来源](analysis/recipe_configurations/CATALOGUE.json)。
- [科学源码/配置版本映射](SCIENTIFIC_SOURCE_MAP.json)、[输入来源和哈希清单](PUBLICATION_MANIFEST.json)。

## 当前实验与主要结果

最高完整点仍为 **SafetyEP Control 10k：91.45359756**。共同原IL70为88.79290991；所有当前对照从同一IL70独立初始化。训练seed42、评测seed2101，canonical NAVTEST12146场景/invalid0，single B1 ODE K10/shift5。不同Adam更新、场景曝光、PG候选、额外B1查询分别核算，不能统一称为等算力。

当前已确认在训（只代表上述快照，后续实时状态看NAS报告）：

- `INTEGRATED_SAFETY_EP_B64_G16_RETENTION_WKC`: 1,343 actual optimizer updates
- `IMPROVEMENT_VERIFIED_RETENTION_G8`: 15,824 actual optimizer updates

| 路线 | 最新完整：actual Adam / PDMS | 本路线已测峰值：actual Adam / PDMS |
|---|---:|---:|
| H·更早衰减 | 14,000 / 91.067069 | 10,000 / 91.356638 |
| H·真FP32动作投影 | 5,000 / 90.778388 | 5,000 / 90.778388 |
| H·固定G8混合探索 | 20,000 / 91.037454 | 18,000 / 91.106650 |
| H·不同场景Batch32 | 4,000 / 91.065543 | 2,500 / 91.113278 |
| H·质量门控历史RL保持 | 14,000 / 91.035270 | 10,000 / 91.429083 |
| I·B64／G16保持＋WKC综合 | 250 / 90.337360 | 250 / 90.337360 |
| Active课程 | 26,000 / 90.839778 | 22,000 / 91.076574 |
| DenseDAC | 28,000 / 91.280974 | 10,000 / 91.355759 |
| G16 | 50,000 / 91.099432 | 24,000 / 91.424936 |
| 饱和信用 | 40,000 / 91.269116 | 36,000 / 91.320771 |
| Control | 44,000 / 90.965725 | 10,000 / 91.453598 |
| 三信用组合 | 40,000 / 91.249520 | 36,000 / 91.298130 |
| Endpoint低LR | 16,000 / 85.897928 | 5,000 / 89.334679 |
| 纠偏信用 | 40,000 / 91.139289 | 10,000 / 91.353421 |

## 目前可以支持的结论

1. **SafetyEP的分解信用有实际阶段收益。** 原同预算10k对照从PPO_FULL91.01948815到SafetyEP91.34989174，+0.33040358，44物理log配对95%CI[+0.04876,+0.62257]；但零分243→256、修复37/新增50，不能把总分增益读成所有安全性均改善。这个旧恒定LR实验与后来的标准Control是不同训练配方。
2. **历史保持有阶段收益，尚未提升最高分。** 保持组5k91.00395950对Control5k90.71690218，+0.28705732，CI[+0.15385,+0.43256]。新补齐10k91.42908305，接近但未超过Control10k91.45359756；14k91.03526994再次回落，后期稳定性仍未解决。额外真实TRAIN B1验证成本另计。
3. **更早衰减可改善相对后期Control的部分节点。** 12k、14k分别+0.23495761、+0.13881161，但自身10k91.35663808之后仍回落，不能称已经解决10k后下降。
4. **G16有发现和部分预算条件下的收益，但没有刷新绝对上限。** 10k91.34195318对Control20k90.96099697，+0.38095621，仅主PG候选量对齐；新补齐24k91.42493557、50k91.09943163，未超全局最高。不能把这些不同Adam/场景/调度的结果当成纯采样数因果对照。
5. **更大场景batch、FP32、混合探索不能默认视为有效组件。** Batch32的2500对Control10k为-0.34031975；FP32的5k点+0.06148533且CI跨零；混合探索5k为-0.36090088，虽后期改善仍未刷新最高。阶段负结果不等于永久否定方法族。
6. **warmup/cosine、批std与IL参考锚定是当前有效配方的共有部分，尚无完整的独立因果消融。** 不把“使用了这些组件”说成“已证明每个组件带来收益”。论文中DAPO、ScaleRL、JustRL也有强恒定LR配方，并非统一cosine最优。

以上区间是固定训练seed条件下44个物理log的配对bootstrap，不是多训练seed不确定性。

## 当前综合组：B64 / G16 / 历史保持 / WKC

当前综合组从同一原IL70 fresh训练，实际10,000次Adam更新后才判断是否长训；未来完整NAVTEST仅actual2k/5k/10k。250点90.33735971保留历史，它发生在保持启用之前，不能据此判断整套方法最终有效或无效。科学配置固定12500调度总时钟：warm125→keep625→cos2500→floor0.1，运行预算扩展不重置模型、Adam、RNG或历史bank。

- global64不同scene × G16＝每窗口1024 fresh PG，复用4次inner。
- action LoRA和scene/predictive memory实际训练52,707,840参数；Qwen冻结。两组峰值LR5e-5/5e-6。
- BC关闭；`kl_type=x, beta=50`是参考动作转移均值MSE锚定，不是LLM token分布KL。PPO clip .005、优势cap5。
- SafetyEP优势：有可行候选时分配安全相对信用，并在可行子集比较EP；全不可行时用NC/DAC/TTC缺失成本比较。最终按全批fresh raw-A std归一化，历史不混入fresh PG。
- [625步全量机制分析](analysis/integrated/TRICK_REVIEW_20261102/DIAGNOSTICS.json)：全部8rank、10048个不同TRAIN组、160768候选。前8条→全部16条的strict-good覆盖83.6584%→90.1274%，新增650组；这是同一G16抽样内的边际发现，不是独立G8策略或部署提分。
- 历史保持23/5952槽触发（.3864%），25%只是触发时的系数份额；全局名义历史系数约.0572%。额外11813 B1查询约为PG查询的7.3479%，不代表GPU时间。失败/恢复重复成本另记。
- [同版本B1与G16诊断](analysis/integrated/TRICK_REVIEW_20261102/NOMINAL_VS_SAMPLED.json)：6016个TRAIN场景中884个采到strict-good但当时B1不达标，含12个B1零分；这是更新前机会，不证明更新后仍持续学不会。
- [825个TRAIN救援候选](analysis/integrated/TRICK_REVIEW_20261102/TRAIN_RESCUE_POOL_MANIFEST.json)和[后续微调提案](analysis/integrated/TRICK_REVIEW_20261102/REFINEMENT_PROPOSALS.json)仅为准备态，未注册、未派发；不把NAVTEST名单回流TRAIN。

## 代码与设计入口

| 内容 | 入口 |
|---|---|
| 训练/原生恢复与评测CLI | [tools/qwen_world_safety_exploration](../../tools/qwen_world_safety_exploration) |
| SafetyEP优势、batch/G16、历史bank与调度 | [starVLA/rl/qwen_world_safety_exploration](../../starVLA/rl/qwen_world_safety_exploration) |
| PPO/CISPO等目标、动作约束与策略接口 | [qwen_world_research_algorithms](../../starVLA/rl/qwen_world_research_algorithms) |
| 综合组全量诊断脚本 | [analyze_integrated_trick_diagnostics.py](../../scripts/analysis/flow_safe_improvement/analyze_integrated_trick_diagnostics.py) |
| 评测CPU补齐优先调度 | [eval_completion_owner.py](../../tools/qwen_world_safety_exploration/eval_completion_owner.py) |
| 报告采集、HTML与图表代码 | [reporting_source](reporting_source) |
| 可重复导出仓库快照 | [publication_snapshot.py](../../tools/qwen_world_safety_exploration/publication_snapshot.py) |

Publication HEAD汇总代码用于审阅。实际运行的科学/执行源码继续按原commit和NAS冻结目录绑定，见SCIENTIFIC_SOURCE_MAP；本次提交不替换任何活跃source，也不自动派发训练。配置中的数据/IL父模型路径仍指原NAS依赖，仓库不包含权重、数据集、环境缓存或密钥。

## 评测恢复与运行证据

[CPU调度/补评收据](operations/evaluation_completion)保留精确15个wm5进程被SIGKILL中断的失败审查；原结果和partial scenes保留，按同模型/source/spec/world32/seed重试。具体终止原因未确认，不假设OOM，不伪造旧RESULT0。G16优先补齐尾分片，历史保持按10k→12k→14k；不停止健康GPU子进程。

当前训练、未齐评测和已结束路线的状态分开显示；ABCD RL、TIME及其他既有用户终止项不恢复。实时报告仍为 `/mnt/project/qwen-world-rl-optimization-20261030/report.html`，仓库HTML是带时间戳的静态快照。

## 发布来源与验证

- 科学训练及综合组：`research/qwen-world-safety-exploration-20261031` / `fcb2060`。
- 报告及配置目录：`research/qwen-world-abcd-rl-best-20261031` / `84565f2`；导出时采用最新NAS报告源码及已封存结果。
- 评测恢复：`ops/qwen-world-eval-completion-20261102` / `c4e3451`，在本分支整合为`1caeaae`。
- 验证只检查导出完整性、配置SHA、报告链接、Python语法以及已有相关CPU合同，不重复模型评测。
