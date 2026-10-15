# IQE 当前结果汇总：Base、独立专家和自训 Scorer

证据截止：**2026-10-15 00:06:22 UTC**。仓库 `IDayday/VLA-Drive`，分支 `research/iqe-robust-scorer-20261014`，整理起点 `94542caae5497cc90e9cf74592471395e73131ec`。本页更新此前的[10月14日快照](../iqe_results_summary_20261014/README.md)，保留历史文件及失败结果。

**Base 已完成100k更新和既定Navtest；专家有真实补充价值，但自训Scorer仍未稳定兑现Oracle机会。** 道路与交互专家在相应测试失败组上有正向平均收益，进度专家存在显著尾部退化。九个完整Scorer实验均已训练和评估，尚无可靠、无安全退化的留出收益。

本次只读取既有产物、整理报告和推送；没有启动训练、官方评分或新的Navtest，没有修改模型、候选池、ACTIVE和数据规则，也没有向已撤回的限时主机派发工作。既有时序风险诊断的数据准备进度单列，未将其当成新Scorer有效结果。

所有表格的PDMS均为 **[0,1]原始分数×100**，差值为百分点。训练集、开发集和测试集分别列示，不能横跨集合或Base步数相减。

## 1. Base：训练与既定评测均完成

连续训练终态为 `COMPLETE / 100000`，最终完整checkpoint：

```text
/mnt/project/iqe-remote-rolling-cache-20261012/continuation/step_100000.pt
SHA256: 6f62523a97b4cd2d229ad0b950e489d83f76bb8b30966669836d2e8d23e7d2e4
```

下列既有评测均为相同Navtest清单：**12146/12146有效场景、136 logs**；dataset和评分协议hash逐项一致。

| Base更新步 | Navtest PDMS | 零分场景 |
| ---: | ---: | ---: |
| 50000 | 88.6485 | 602 |
| 60000 | 89.0760 | 547 |
| 70000 | 89.1006 | 560 |
| 74000 | 89.1851 | 543 |
| 80000 | 89.2446 | 533 |
| 90000 | 89.2175 | 540 |
| 100000 | **89.3962** | **518** |

100k的NC/DAC/TTC/EP均值分别为98.3122/97.3407/95.4141/83.3573；完整曲线和每步checkpoint身份见 [BASE_NAVTEST.csv](BASE_NAVTEST.csv)，原始分项与逐log聚合见[100k结果](evidence/base/step_100000_NAVTEST_RESULT.json)。100k相对70k的均分差为+0.2956分，这是已完成监测的描述，不是本次重做显著性检验或按测试集重选专家起点。

这些是原持续训练lineage的结果，不能代表另行发布的“仅轨迹IL”新实验。Navtest已被重复用于用户要求的里程碑观测，不能称为完全未接触测试集；[访问记录](evidence/base/step_100000_NAVTEST_ACCESS.json)明确记录这一边界，且没有训练写回。

训练效率：最后89732→100000段，8×A800 80GB、global batch32，最慢rank每步统计为mean1.7461秒、p50 1.0617秒、p95 8.0598秒，10257个计时更新，另排除10步预热和1步排空。该段checkpoint耗时492.79秒单列，不把p50当长期平均或把最后一段当整个Base总成本。来源为[已有训练终态的字段摘录](BASE_TRAINING_COMPLETION.json)，不是本次新测速；本次只核对完整receipt及权重文件存在，没有重新加载或重hash大权重。

## 2. 专项研究仍固定70k Base

本轮专家/Scorer绑定的Base始终为70k，SHA256 `30517484efcc8740c331f42f3f2ba2d1896a4f95cf70e2bf662e0d6c4484e8b0`。100k训练完成没有改变这个研究身份，也没有更换原22k增量起点。

| 数据角色 | 场景 / logs | 实际用途 |
| --- | ---: | --- |
| D_E | 74311 / 845 | 专家目标审计、专项筛选 |
| D_S | 16581 / 211 | Scorer训练；三个新专家未使用，Base已见 |
| stage_val | 1024 / 53 | 专家配方、raw/EMA端点和Scorer选模 |
| selector_cal | 1024 / 62 | 仅选择附加margin |
| dev_report | 1696 / 16 | 固定完整开发评估 |
| 既有Navtest | 12146 / 136 | 已授权完成的测试描述，未用于本轮拟合 |

原始[Base绑定](../iqe_results_summary_20261014/evidence/BASE_BINDING.json)和[数据来源](../iqe_results_summary_20261014/evidence/DATA_ROLES_AND_PROVENANCE.md)保留实际曝光记录。D_E/D_S按log隔离；D_S只是新增专家训练阶段的留出，不能称为Base未见。

已完成6次专家训练：试点E00/E10/E01/E11及两个专项扩展，每次5000更新、global batch128。最终池为Base+道路E11/3000/EMA+交互/3000/raw+进度/1000/raw，池hash `baccbfbfa27b387c30a58acf59cd6bd53a564cadd7dc6bdb29fab80de8b80029`。raw/EMA没有混成额外专家。

## 3. 专家训练数据与真实专精

| 专家 | H_train场景 / logs | 监督 | 5000步平均每场景曝光 | 选中端点 |
| --- | ---: | --- | ---: | --- |
| road_direction | 125 / 89 | offline_optimized | 5120 | 3000 EMA |
| interaction | 116 / 90 | offline_optimized | 5517 | 3000 raw |
| safe_progress | 393 / 231 | offline_optimized | 1628 | 1000 raw |

三组有634个专家×场景、622个独立观测，没有普通anchor或新增合成观测。这些是完整D_E中通过专项与目标审计后的集合，**不等于Base低分场景总数**。筛选漏斗、目标来源、每组曝光上下界和checkpoint hash见[训练数据表](EXPERT_TRAINING_DATA.csv)、[原始漏斗](../iqe_results_summary_20261014/evidence/TRAINING_FILTER_FUNNEL.csv)和[数据说明](../iqe_results_summary_20261014/evidence/TRAINING_DATA.md)。大量重复呈现确实存在，不能用5000步声称拥有大量独立监督。

| 专家 | H_train Base→专家 | 训练直接收益 | 对应测试组场景 / logs | 测试直接收益 | 测试收益95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| 道路/方向 | 18.24→95.26 | +77.03 | 622 / 92 | +22.60 | [16.85, 28.72] |
| 交互 | 55.62→99.61 | +43.99 | 603 / 88 | +13.63 | [10.98, 16.36] |
| 安全进度 | 86.38→97.40 | +11.03 | 2279 / 117 | **−3.48** | [−5.62, −1.36] |

对应测试组只按Base自己的分项失败规则定义，不按专家胜负、GT或Scorer结果挑选。它们比目标质量筛选后的H_train更宽，并非严格同分布H_test；道路/交互重叠58场景。严格H_val仅1/9/11场景且参与选模，不能将这三组小型验证样本包装成强泛化证据。

道路和交互结果支持“在这类Base失败场景中改善”，不意味着不会碰撞或越界，也不意味着该命名对应独占能力。进度专家虽有83.06%的对应测试场景改善超过1分，却有186次Base高分转零分，导致平均净退化。[完整交叉结果](../iqe_results_summary_20261014/evidence/EXPERT_SCENE_GAIN_REPORT.md)保留其他专家在同组的表现与尾部风险。

固定5000步raw/EMA的2×2表和配对对照分别见[专家矩阵](../iqe_results_summary_20261014/evidence/EXPERT_2X2_RESULTS.csv)、[因素对照](../iqe_results_summary_20261014/evidence/EXPERT_FACTORIAL_CONTRASTS.md)。现有结果不足以把私有读取、优化监督或更长训练写成普遍收益。

## 4. 测试集四候选：Oracle有增量，尚无Scorer测试分数

| 固定策略，全部12146场景 | PDMS | 零分场景 |
| --- | ---: | ---: |
| Base70k | 89.1006 | 560 |
| 道路专家alone | 77.4266 | 1895 |
| 交互专家alone | 84.1248 | 1062 |
| 进度专家alone | 74.9215 | 2191 |
| Oracle@4 | **94.2039** | **249** |

相对同一Base的Oracle增益为**+5.1033分，95% CI [4.5680, 5.6959]**。剔除道路/交互/进度专家分别损失0.7275/0.7443/1.3837分Oracle。单独使用任何专家均弱于Base，进度专家组内净退化与其仍有边际Oracle贡献可以同时成立。

[原始测试结果](../iqe_results_summary_20261014/evidence/EXISTING_NAVTEST_RESULT.json)明确 `Scorer_executed=false`。**尚无这批自训Scorer的Navtest结果**；下文90分左右的Scorer值来自开发集，不能填入测试集列。本次没有新开Navtest。

## 5. 同一开发集上的全部已完成Scorer

以下均为1696场景/16 logs、同一Base70k和同一四候选池。Base **90.5302**，道路/交互/进度alone为77.5541/85.1406/81.4423，Oracle **95.3517**，最佳固定策略为Base。总Oracle机会为4.8215分。

| 模型 | 输入/训练变化 | selected PDMS | 比Base | 高分→零分 | 新增NC/DAC/TTC失败 |
| --- | --- | ---: | ---: | ---: | --- |
| S00 | W/A/ego，部署银行 | 90.2965 | −0.2337 | 25 | 18/19/61 |
| S01 | W/A/ego，丰富银行预训练 | 90.6436 | +0.1134 | 19 | 18/12/68 |
| S10 | 全局空间读取，部署银行 | 89.4852 | −1.0450 | 29 | 21/22/76 |
| S11 | 全局空间读取，丰富银行预训练 | 90.2695 | −0.2606 | 17 | 20/7/80 |
| R00 | 语义输入，软标签BCE期望分数 | **90.6960** | **+0.1658** | 11 | 14/4/44 |
| R01 | R00目标，dropout0.1 | 90.4376 | −0.0926 | 25 | 18/16/66 |
| R10 | 联合非零概率×条件分数 | 90.5066 | −0.0236 | 18 | 9/15/30 |
| R11 | R10目标，dropout0.1 | 89.7463 | −0.7839 | 23 | 15/16/57 |
| 几何读取 | 轨迹条件局部空间读取，原S11损失/配方 | 90.5824 | +0.0523 | 18 | 20/9/82 |

每个模型均完成6000更新；表中使用各自stage_val选择的端点。S矩阵预选S11，R矩阵预选R00，几何模型选择4000步，不能在dev选最高值重新宣称它是预注册赢家。R矩阵没有空间输入，因此R00与S11的差异不是纯损失单因素对照。

R00增益95% CI **[−1.1025, +1.2360]**，捕获3.44% Oracle机会；几何模型增益95% CI **[−1.7163, +1.5967]**，捕获1.08%。两者区间都跨零并有新增安全失败，不支持可靠选轨结论。完整数字、分母、regret、置信区间和来源在[统一结果表](CURRENT_RESULTS.csv)。

S11、R00和几何模型的独立selector_cal校准均未找到满足既定风险预算的非平凡margin，最终always-Base。此结果只能说明校准退回Base，不算Scorer成功。

## 6. 最新几何实验揭示的瓶颈

| 集合 | 场景 / logs | Base | Oracle | selected | 收益 | Oracle捕获 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D_S拟合集 | 16581 / 211 | 94.3310 | 97.5383 | 96.4392 | +2.1082 | 65.73% |
| stage_val | 1024 / 53 | 91.3915 | 96.5215 | 91.2401 | −0.1514 | −2.95% |
| 固定dev | 1696 / 16 | 90.5302 | 95.3517 | 90.5824 | +0.0523 | 1.08% |

开发集正确替换收益+3.0223分，错误替换损失2.9701分，几乎相互抵消。`G−R=4.8215−4.7692=0.0523`。切换1330/1696场景、78.42%，原始argmax的小收益不是始终回退造成的。

开发集零分63 vs Base55，18次高分转零分；regret>0.1有141场景、>0.5有42场景。真实零分候选的平均预测仍达0.5446。总NC失败39 vs22、TTC失败147 vs71；DAC总数27 vs37有所改善，但仍新增9次DAC失败，不能用净总数掩盖新增风险。

同一stage_val、固定权重的正确V/错配V/屏蔽V得分为91.2401/91.6734/91.4241，没有形成“正确空间配对提高留出选择”的证据；不能把错配诊断改成部署方案。投影覆盖审计中，85次错误替换有33次所有轨迹点都投影到至少一个画面内，占错误代价40.65%；失败不能只归结为视野外，但投影在画面内也不代表无遮挡或真实未来可见。

这更支持**训练可拟合、留出严重误选和泛化不足**。现有证据不足以认定“空间信息没有价值”或“单纯延长同一训练必然有效”。一个主seed、16个开发logs，且旧dev结果在跟进实验前已观察；不称为多seed稳定结论或全新盲测。

详细结果与最终bundle证据见[几何完整结果](../iqe_results_summary_20261014/GEOMETRIC_FINAL_UPDATE.md)。

## 7. 来源、效率与新诊断状态

全部新增Scorer参数从随机初始化训练，没有加载DrivoR、M0或历史教师评分权重。图像部署仍执行一次视觉、一次语言前向；W/A/V复用该次结果。每场景评分上下文投影一次，各专家增加独立读取/动作计算，评分网络没有跨候选attention。

R00新增3301639参数，几何Scorer新增4898311参数。真实图像路径A800 80GB、batch1、预热3次、计时32次：几何K4+Scorer总mean791.209 ms、p50 790.996 ms、p95 817.008 ms，评分头mean11.280 ms；同期Base mean737.235 ms。总路径峰值allocated9269392384 bytes。不是缓存头部时延，也不称零新增成本；测量条件和端点见[实测记录](../iqe_geometric_readout_20261014/REAL_LATENCY.md)。

几何选中的4000步模型已经完成真实图像、独立进程bundle保存加载检查，候选、分数和最终轨迹最大绝对差均0。[最终检查](../iqe_results_summary_20261014/evidence/geometric_final/SELECTED_BUNDLE_REPORT.json)只证明工程一致性，不证明科学收益。既有冻结、两流采样与恢复证据沿用原始运行记录，本次不把历史测试数量累计成新测试。

训练开销按阶段列在[专项训练/评分成本](../iqe_results_summary_20261014/evidence/STUDY_TRAINING_AND_LABEL_COST.md)和[几何收尾结果](../iqe_results_summary_20261014/GEOMETRIC_FINAL_UPDATE.md)。R矩阵update记录共15005.40 GPU秒，不含验证、加载、checkpoint和后评测；几何最后5998更新进程墙钟4061.06秒、4卡分配16244.25 GPU秒。统计范围不同，不能直接相加当全项目总账。

既有时序风险诊断在本次汇总前已启动数据准备：**64个D_S场景、52 logs、747个既有候选，4个CPU worker全部完成**，总分和分项与旧银行最大差0。新标签是官方仿真产生的首次碰撞、越界、TTC违规时序，仅作为训练监督，不是Scorer输入。见[完成证据](evidence/temporal/COMPLETE.json)与[既有注册](../iqe_temporal_risk_20261014/REGISTRATION.md)。截至本快照尚无训练结果、无留出评估，不以标签准备成功宣布选轨改善，也没有因本次整理追加训练。

历史ReCogDrive首版/强化版四教师的测试Oracle为95.1680/95.8931，外部DrivoR选轨为92.1649/92.2993。它们使用不同RL Base和外部评分编码器，既不是单模型分数，也不能证明当前自训Scorer有效。保留[原始历史汇总](https://github.com/IDayday/VLA-AD/blob/5f241ec78f7505af924c8b24bd09ee78ad45a6d8/reports/recogdrive_four_teacher_20261014/README.md)和[匹配K拆解](../iqe_specialist_scorer_study/HISTORICAL_MATCHED_K_REPORT.md)供追溯；MOPD按用户要求保持停止。

## 8. 可核验交付与后续边界

- [STATUS.json](STATUS.json)：各实验完成和科学状态，明确Scorer测试分数未生成。
- [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)：原始路径、SHA256、32份本次归档/摘录及60份既有归档核验。
- [VERIFICATION.json](VERIFICATION.json)：本次报告检查结果；不是新的模型资格测试。
- [统一结果CSV](CURRENT_RESULTS.csv)、[Base曲线](BASE_NAVTEST.csv)、[专家专精](EXPERT_SPECIALIZATION.csv)、[训练数据与曝光](EXPERT_TRAINING_DATA.csv)。
- 真实命令分别在[专项研究RUNBOOK](../iqe_specialist_scorer_study/RUNBOOK.md)、[R矩阵RUNBOOK](../iqe_robust_scorer_20261014/RUNBOOK.md)、[几何完整实验RUNBOOK](../iqe_geometric_readout_20261014/FULL_TRIAL_RUNBOOK.md)。这些已完成实验不因报告整理重训；换Base必须使用新身份。

本次可在任意克隆了本分支的机器只校验小型报告，不加载模型或访问Navtest数据：

```bash
python reports/iqe_results_summary_20261015/verify_summary.py
```

当前应保留的科学结论是：**专家候选有可测互补性；可靠自训选轨尚未建立。** 已完成的工程链路、训练拟合、Oracle、实际选择和风险必须分别报告。没有修改正式ACTIVE，没有重启MOPD，没有提交权重、原始图像、metric cache或动态缓存实现。
