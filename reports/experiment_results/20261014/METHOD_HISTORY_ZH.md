# C 到 G 的方法演进与实验结论

本报告以 2026-10-14 的证据为准。正式结果、中间权重、短训、推理步数实验和工程候选分别标记；没有结果的位置不补零。原始 CSV、冻结配置与失败记录都保留，报告索引见 [README](README.md)。本次整理不改变任何训练算法或运行授权。

## 共同基础与比较边界

主干是通用预训练 Qwen3-VL-2B、原 DDP DiT 动作头与合法当前 ego 编码。NAVSIM 使用 3 个当前前视、导航和当前/合法历史传感器状态，输出后轴 ego(t0) 坐标的 4 秒 8 点轨迹；部署不读取 GT 未来、地图标签或 teacher。原动作表示为归一化 x/y/sin(yaw)/cos(yaw)，原 FM repeat8，正式推理 10 步 Euler、单候选。每次辅助监督按原始场景计算，不随 repeat8 放大。

C/S 正式训练均 100k optimizer updates、global batch32，实际曝光以原记录为准：C0/C1/C4、S0–S4 均为 3,199,752 次，不能机械写成恰好 3,200,000。C/S/A/V 都只有一个训练 seed42。训练 BF16 路径的吞吐不可与 FP32 canonical 部署的质量拼成同一个精度点。原结果和源身份见 [COMPLETED_RUNS](evidence/C_S_COMPLETED_RUNS.json)。

“最佳”只指已完成完整 Navtest 的检查点中最高 PDMS。它是事后选择，不是最优配置证明；对应置信区间不覆盖训练种子波动和选权重偏差。相同步数也不保证相同计算预算：G 增加了在线几何、逐时刻空间表示、真实 proposal 与修正，以及共享几何准备。

## C：当前/未来图像特征与车辆交互监督

完整 C 方案在 Qwen 当前序列内加入带 view/row/column 身份的 W query，位于 action queries 之前，保留原 causal attention。共享读出头学习当前与未来 DINO；独立交互头从同一 W 预测冻结 GT-MAE 的交互表征。MAE 的“vehicle”是当时定义的其它汽车，不能和 G 的环境障碍 occupancy 类别混为一谈。未来时刻从 1/2/4 秒请求，每个原始场景一次当前与一次未来监督；当前权重 1、未来 0.97942443、交互 0.83289460，辅助项 warmup1000。部署移除辅助读出头，保留 W。

DINO 输入分辨率变化的是 teacher 目标，不是重新把 Qwen 输入也改成相同大小。Qwen 仍使用固定允许源图像与原 processor；C teacher 从原源图像 resize 至登记画布。六组目标与配置准备完毕，不等于六组都有正式规划结果。

| 组 | DINO 目标宽×高 | pool | W | 实际证据 |
|---|---|---:|---:|---|
| C0 | 128×96 | 1 | 144 | 100k 完整；最佳 89.1512 |
| C1 | 256×192 | 2 | 144 | 100k 完整；最佳 89.2395 |
| C2 | 256×192 | 1 | 576 | 配置/目标准备，无正式规划分数 |
| C3 | 384×288 | 2 | 324 | 真实短链路与 4 步连续/2+2 恢复验证，无正式长训分数 |
| C4 | 512×384 | 4 | 144 | 100k 完整；最佳 89.2776 |
| C5 | 512×384 | 2 | 576 | 120-update 性能测试，无正式规划分数 |

C1−C0、C4−C1、C4−C0 在 100k 分别为 +0.0884、+0.0381、+0.1265 点，历史配对区间均跨零。没有证据支持继续搜索分辨率/W 来获得明确规划提升。这里的“完整 C”不是更早 `configs/dino_tradeoff` 中同名 current-only 容量测试；历史未核验的 89.41 不作为复现基线。

来源：[原 C 实现说明](https://github.com/IDayday/VLA-Drive/blob/d1d40854299b9599b2accc382bcfc4b676dd7623/reports/ddp_full_foresight/IMPLEMENTATION.md)、[六组登记](evidence/c/formal_registration_v1.json)、[恢复实测](evidence/c/full_real_resume_comparison.json)、[C/S 历史配对](../20261008/analysis/c_s_paired_100k.json)。本次没有重新搜索这些配置。

## S：逐帧到视频、动作条件与规划接口

S 固定 C1 当前 DINO/W144，把未来监督扩为 0.5–4 秒 8 帧。逐帧目标有 8 个物理时刻；V-JEPA 视频目标对应 4 个相邻双帧时间区间。未来目标使用逐 token 非仿射 LayerNorm，future weight1.15495435；current1、MAE0.83289460。因此 S0−C1 同时包含时刻曝光、归一化与权重变化，不是只换一项 teacher。

| 组 | 改动 | 100k / 已测最佳 PDMS | 对照与结论 |
|---|---|---:|---|
| S0 | 逐帧 DINO；MAE 从 W 读出 | 89.5279 | 相对 C1 +0.2883，区间 [-0.0185,+0.6056]，整体配方信号尚不确定 |
| S1 | S0 的未来头增加 GT ego 动作条件 | 89.3411 | 相对 S0 −0.1868，未确立动作条件收益 |
| S2 | 未来改为 V-JEPA 视频，无动作条件 | 89.3365 | 相对 S0 −0.1913，未确立视频收益 |
| S3 | S2 的未来头加入动作 memory | 89.4082 | 相对 S2 +0.0717，区间跨零 |
| S4 | S3 的动作规划器直接读最终 W | 89.1293 | 相对 S3 −0.2789，区间 [-0.5619,−0.0023]，当前接口有负面证据 |

S1/S3 的 GT ego 条件只进入训练期未来辅助头；不能据此指控部署读取未来，也不能把它与 G 中模型自身 q0 条件混同。S4 是直接给动作头接入 W 的接口尝试；其失败不等于所有场景表示都无价值。

辅助头确实学到信息：S3 同一视频目标下，置换 W 使归一化预测 MSE 0.2497→0.5513，冻结 MAE 解码 ADE 0.80m→7.42m。它证明表示依赖，不能证明规划收益，更不能按不同 teacher 的 MSE 横排模型。[同教师辅助诊断](../20261008/analysis/auxiliary)与[历史最优分析](../20261008/analysis/best_checkpoint_summary.json)保留。

## A/V：辅助监督放在哪里、动作怎样注入

| 名称 | 实际修改/复用 | 已测最佳 PDMS | 解释 |
|---|---|---:|---|
| A_W | 精确复用 S0 | 89.5279 | 别名，不是新 seed |
| A_NO_MAE | S0 去掉 MAE，保留当前/未来 DINO | 89.4826 | 比 S0 −0.0452，未证实 MAE 增益；也不是纯 Action-Only |
| A_ACTION | MAE 读出从 W 移到 action 表示 | 89.0933 | 比 S0 −0.4346，区间 [-0.7502,−0.1275]，当前迁移接口有负面证据 |
| V_NONE | 精确复用 S2 | 89.3365 | 别名 |
| V_MEMORY | 精确复用 S3 | 89.4082 | 别名 |
| V_QUERY | S3 动作条件同时注入 memory 与 query | 89.5320 | 比 S0 +0.0041、比 S3 +0.1238，均未证明提高上限 |

三个新训练均完成 100k，最佳都在 100k；各实测节点见[全表](RESULTS_BY_CHECKPOINT.md)。W 侧 MAE 可以学习交互，但当前实验没建立稳定 PDMS 增益；把它推到 action tokens 反而退化。V_QUERY 与 S0 是当前历史强参照，而非已经证明世界模型获益的一对结果。[别名权重等价性](../20261008/analysis/baseline_reuse_proof.json)、[冻结配置](../20261008/configs/av)可核对。

## 优化规划标签：r1/V12 与原 FM

r1/V12 是离线优化 ego 轨迹后按接受规则替换/回退 GT 的标签版本，不是模型在线奖励，不是数据集原生风险 GT。101,592 个训练场景中，r1 接受 88,499，V12 接受 92,841；V12 新接受 4,342，且两者共同接受的轨迹中有 37,799 条发生变化。V12 不只是补更多样本。

R/S0/S3 的 r1 训练保留**原始 FM**，均完成 100k；未来视频条件仍是 original_logged_ego，没有偷偷改为优化 ego 答案。

| r1 原 FM | 最佳更新 | 最佳 PDMS | 零分数 |
|---|---:|---:|---:|
| R Action-Only，W=0/无辅助 | 100k | 89.0167 | 639 |
| S0 | 90k | 89.4671 | 608 |
| S3 | 80k | 89.5281 | 592 |

S0/S3 相对同标签 R 最佳分别 +0.4503/+0.5114，历史配对区间为 [0.0347,0.8878]/[0.1566,0.8885]。这是整体方案对同标签 Action-Only 的正向信号，检查点/曝光不同；不能据此单独归因于视频、MAE 或动作条件。

优化标签 S3 最佳相对原标签 S3 最佳：EP +4.53、DAC +1.07，但 NC −1.68、TTC −4.98，零分增加 76，PDMS 只 +0.1199 且区间跨零。S0 也未确立综合分提升。应保留“提高进度/道路合规但牺牲安全分项”的结论，不用接近的总分掩盖取舍。[标签差异](../20261008/analysis/historical/label_comparison.json)、[优化配置](../20261008/configs/optimized)、[最优与逐场景数据](../20261008/README.md)仍为来源。

## 新轨迹目标：成熟短训、长训反转与 v2 拒绝

最先在 S3@100k、S0@50k 成熟权重上，以共同新 optimizer/LR 做 512-update 配对继续训练。各架构分别有原 FM control、归一化通道权重、通道加 rollout、绝对 x 增益、绝对 x/yaw 增益、FM bridge 端点运动学、再加 braking 共 7 组；另存只读 initial。共 14 个短训结果加 2 个 initial，见[完整分项及区间](data/OBJECTIVE_MATURE_PILOTS.csv)。这些不是 512 步从随机训练出的 90 分模型。

| 512 步目标 | S0 dev PDMS | 相对 control | S3 dev PDMS | 相对 control |
|---|---:|---:|---:|---:|
| 原 FM control | 89.5551 | — | 90.5808 | — |
| normalized axes | 89.8864 | +0.3313 | 90.9332 | +0.3524 |
| axes + native rollout | 90.1209 | +0.5658 | 89.3432 | −1.2376 |
| absolute x | 90.9290 | +1.3739 | 90.8711 | +0.2903 |
| absolute x/yaw | 90.5003 | +0.9452 | 90.9085 | +0.3277 |
| endpoint kinematics | 90.5731 | +1.0180 | 90.7455 | +0.1647 |
| endpoint + braking | 90.4394 | +0.8843 | 91.2425 | +0.6617 |

筛选数据为完整 dev 1,696 场景/16 日志，未以 Navtest 选该目标。最终选择 endpoint+braking 有两个架构的 TTC 信号，但总 PDMS 区间跨零，ADE 与重复噪声波动仍变差。其[当时放行理由及限制](evidence/objective_pilots/selected_objective_review.json)原样保留，之后的否定证据不被这份旧放行文件覆盖。

随后四组 S0/S3 × r1/V12 从共同通用/随机驾驶初始化长训，采用绝对 x4、其它通道1，braking scene×2，以及 bridge 端点速度/加速度权重 5e-4/2.5e-5。它不是对最终 10 步 rollout 的损失。仅 S3 两标签有归档的完整 50k Navtest：

| S3@50k | PDMS | 相对 r1 原 FM |
|---|---:|---:|
| r1 原 FM | 88.3770 | — |
| r1 新激进目标 | 87.6191 | −0.7579，95% CI [−1.2780,−0.2656] |
| V12 新激进目标 | 87.3451 | 同时换标签，不能只归因于目标 |

r1 新目标修复 197 个零分却新增 306 个，净增加109；新增道路失败和 yaw 尾部退化明显。x 编码轴也不是随未来道路方向转动的 Frenet 纵向轴。49k–50k 诊断中，全局梯度均值约 0.840→1.808、clip>1 比例 12.19%→98.50%。这支持损失尺度/裁剪与道路姿态失衡的解释，但没有单项消融证明唯一根因。[原分析](../20261008/analysis/historical/objective_analysis.json)、[分项配对与失败转移](../20261008/analysis/historical/objective_paired_50k.json)保留。

四组最终暂停边界分别为 S0-r1 54,428、S0-V12 38,468、S3-r1 54,518、S3-V12 52,200；不是四组都完成 100k，也不能把缺失 S0 Navtest 写成 0。[实际登记及配置](evidence/objective_formal)、[暂停收据](../20261008/analysis/objective_v2/original_training_pause.json)仍可追溯。

v2 修复与校准包括：FM [1.5,1,1.5,1.5]、braking增量0.25且按全局有效场景规范化、FM warmup1000、端点速度降为5e-5/加速度关掉、每8步弱 native rollout position4e-5/heading8e-5。训练器落实实际模块 LR（Qwen5e-6、action/base1e-5）并记录 clipping/master update。原 trainer 覆写模块 LR 的缺陷妨碍后续调参，但旧对照所有模块本来同 LR，不能将该 bug 说成旧退化的单独原因。

v2 在 r1 做四组匹配冷启动 control/candidate，全部停在 2k；同配对只有目标不同。共同 LR/warmup 与旧 100k 配方不同，因此跨配方改善也不可只归因于目标修订。

| 组 / update | control dev PDMS | v2 dev PDMS | 差值 |
|---|---:|---:|---:|
| S0 / 1k | 50.2861 | 52.6290 | +2.3429 |
| S3 / 1k | 41.9948 | 51.4819 | +9.4871 |
| S0 / 2k | 66.7418 | 63.7722 | −2.9696 |
| S3 / 2k | 67.7448 | 62.8548 | −4.8900 |

2k 差值区间分别 [-5.0923,−1.4213]、[-7.1474,−3.0984]；128 场景×4 **推理**种子也全部退化。结论是拒绝这一联合配置继续长训，保留原 FM 与训练器修复；不是证明所有新目标都无效。5k/100k 未完成，V12 没有 v2 完整验证。[v2 最终报告](../20261008/analysis/objective_v2/final_report.json)、[配置不匹配与修复清单](../20261008/analysis/objective_v2/config_mismatches.csv)保留。该研究继续暂停。

## G：真实几何空间、逐时刻表示与单次修正

G 保留 Qwen、当前 C1 DINO(weight1)、原 FM。与 S 最大区别是新增真正的单帧图像几何 BEV，而不是把 W reshape 成空间图；训练时同一 H_A 用独立 proposal RNG 从噪声执行原 10 步 q0。q0 用 no_grad/eval、detach；refine loss 经 H_A、Bt、FGTR 回传，不经采样链，原动作头仍由原 FM 学习。

B0 使用公共 ImageNet R50、FPN、RCSample/DepthNet 与 BEV 编码器；单当前帧、标定同步 resize，无未来图像输入，LiDAR 深度只作标签。TokenLearner/MLN/共享 decoder/TokenFuser 根据物理时间、H_A 和模型 q0 生成 Bt=B0+ΔBt。FGTR 复用成熟可变形读取核心，在 q0 对应时刻的 Bt 上读取，再输出一次归一化动作残差。q0 和 q_final 都评测，主表为 q_final。

道路 SDF、当前/未来 occupancy、条件 enter/release、完整车身局部余量由离线地图/框派生，坐标统一到 ego(t0)，保留有效性/unknown 和显式越界。它们不是数据集原生风险 GT，也不等于责任碰撞/TTC。G occupancy 使用可靠共有的环境障碍类别，不能套旧 MAE 的 vehicle 语义。遮挡缺证据、栅格误差和高度近似均依赖已有标签合同，详见[原方法与实现差异](../../structured_world/round1_20261008/ANALYSIS_AND_DIFFERENCES_ZH_20261010.md)。

四组结构相同，BEV范围 x[-20,80]、y[-40,40]、0.5m/256通道、K1024、NAVSIM8时刻/nuScenes6时刻。几何图像256×448。每个数据集共同做 5 epoch 几何监督准备：NAVSIM曝光507,960，nuScenes116,150；只训练感知，不从 S0/ResWorld 驾驶权重继续。正式几何分支可更新。共同新损失系数 geometry0.11612487、future0.10546160、query0.38712657、refine1，refine warmup1000；完整 optimizer/初始化/标签来源以[实际合同](data/SOURCE_VERSIONS.json)为准。

| 组 | 监督定义 | 隔离的比较 |
|---|---|---|
| G0_NO_FUTURE_LABEL | 完整 B0/Bt/FGTR 保留，不开未来语义/未来关系标签 | 结构存在的规划驱动表示，非原生 Action-Only |
| G1_FULL_UNIFORM | 两类未来 occupancy，全局均匀 K 查询 | G1−G0：未来标签是否额外有价值 |
| G2_EVENT_UNIFORM | GT O0 分开监督条件 arrival/release；全局均匀 | G2−G1：事件目标组织 |
| G3_EVENT_LOCAL | 同 G2 目标；50%q0车身邻域、25%边界/变化、25%全局 | G3−G2：规划局部结构化监督分布 |

G2/G3 部署的未来概率为 `(1-p0)*arrival + p0*(1-release)`，使用预测 p0，GT O0 仅决定监督 mask。enter/release 是当前与指定未来的端点变化，不是累计首次事件。G3 的同一 `sample_queries` 同时用于 road、current occupancy、future semantics、body relations，准确表述为 **planning-local sampling for structured scene supervision**；不是只改变未来监督。

语义 head 在部署删除，结构化监督塑造 Bt，FGTR 利用 Bt latent 修正；没有将预测道路/占用数值作为硬约束或 risk score 直接交给执行规划。GeoBEV/TokenFuser/FGTR 是复用组件，贡献候选在监督组织与规划收益。[第三方锁定与许可证](../../../third_party/LOCK.json)、[补充锁定](../../../third_party/ADDENDUM_LOCK.json)保留；ResWorld commit 为 a5ae973，VAD1688c4b、UniAD609ee08，具体文件/测试对应关系均在锁定清单。

## G 的规划结果：组内与历史横评

| updates | G0 | G1 | G2 | G3 |
|---:|---:|---:|---:|---:|
| 25k | 87.4427 | 86.3189 | 88.2273 | 86.7897 |
| 45k | — | — | 89.3858 | — |
| 50k | 88.8899 | 88.9409 | 88.3943 | 88.5195 |
| 60k | 89.0247 | 89.2823 | 89.3232 | — |
| 70k | 88.8936 | 89.0074 | 89.0891 | 88.8701 |
| 75k | — | — | — | 88.9167 |
| 80k | — | 89.1511 | — | 88.9706 |
| 90k | — | — | — | 88.8844 |
| 100k | — | — | — | 88.7913 |

这是已完成观测，横线不是失败或零分。G0/G1 最新暂停边界并未完成对应 Navtest，不填未测分。

关键结论随阶段变化：25k G2−G1 +1.9084，但 50k 变为 −0.5467，60k +0.0409，70k +0.0817。后两项区间分别 [-0.3154,+0.3867]、[-0.1692,+0.3407]。G1−G0 在 70k +0.1138、区间 [-0.2043,+0.4354]；G3−G2 为 −0.2190、区间 [-0.5149,+0.0675]。因此还没有稳定的未来标签、事件组织或局部采样因果优势，不能只展示25k或各组不同步数的最好成绩。[全部分项、置信区间和失败转移](data/PAIRED_COMPARISONS.json)可复算。

| G2 与原标签历史对照 | 差值 PDMS | 配对日志 95% 区间 |
|---|---:|---|
| G2 50k − S0 50k | +0.1512 | [−0.2970,+0.5935] |
| G2 50k − S3 50k | −0.1095 | [−0.5457,+0.3325] |
| G2 60k − S0 60k | +0.4717 | [+0.0001,+0.9615] |
| G2 60k − S3 60k | +0.3024 | [−0.1430,+0.7290] |
| G2 70k − S0 70k | −0.0573 | [−0.4726,+0.3317] |
| G2 70k − S3 70k | −0.1917 | [−0.5759,+0.1517] |
| G2 最佳45k − S0最佳100k | −0.1421 | [−0.6301,+0.2975] |
| G2 最佳45k − S3最佳100k | −0.0224 | [−0.5121,+0.4096] |

60k 相对 S0 区间下界非常接近零，又经过多个观察点，不能包装成稳定显著优越。70k 优势已不在。G2最佳相对 C1 +0.1463、V_QUERY −0.1462，区间也跨零。当前仅能说增加真实几何/修正后的整体方案达到历史强基线附近，尚无可靠的最终上限提升；也没有同算力优势证据。G2 还没完成100k，不能因早期起伏否定整个方向。

同模型 G2 70k 的修正使 PDMS +0.7219、95% CI [+0.4913,+0.9792]，修复 q0 零分106、新增25。证明该模型执行时修正有用，但不能替代独立训练的原 DDP、同容量 current-only 修正器或监督对照。

G0/G1 现因资源紧张按用户决策暂停；这是研究优先级，不是统计上证明无价值。G3完成100k且未超过S强基线；G2按原计划继续。当前无法交付四组共同100k终点或第二训练seed结论。

## nuScenes：跨数据来源证据仍不完整

使用官方 train700/val150，转换后23,230个有效训练样本、4,969个val样本。6个当前相机，3秒6点，合法ego状态；VAD式高层导航由日志未来构造，是共同协议条件，必须披露，但没有输入精确未来坐标或终点距离。地图仅标签用。每数据集各自通用底座/随机驾驶模块与5epoch共同几何准备；没有把NAVSIM驾驶权重迁移过去。

24epoch是本轮预登记预算，不是某篇论文的原配方；726更新/epoch、17,424更新、557,520有效曝光，加一次共同几何准备。论文中的epoch还涉及训练人口、预训练、冻结、batch与每步成本，不能直接等同。已有[外部结果对比及协议差异](../../structured_world/round1_20261008/NUSCENES_EXTERNAL_COMPARISON_20261010.md)、[训练量与ego-conditioned解释](../../structured_world/round1_20261008/NUSCENES_TRAINING_BUDGET_20261010.md)保留，属于10月10日文献快照，本次未重新检索。

主要协议是锁定VAD/UniAD-style开环指标，prefix分别平均前2/4/6点，再对1/2/3秒均值取平均；endpoint另表，不混排。ego状态、缺未来处理、GT自身碰撞过滤均以保存协议为准。不是nuScenes官方闭环挑战。

| 组 / epoch | q_final 平均 L2(m) | q_final 平均 box碰撞(%) |
|---|---:|---:|
| G1 / 6 | 0.6652 | 0.4813 |
| G3 / 6 | 0.6906 | 0.7614 |
| G3 / 12 | 0.6157 | 0.7357 |
| G3 / 18 | 0.5836 | 0.5484 |
| G3 / 24 | 0.5848 | 0.2516 |

相同6epoch，G3未优于G1。G3后期碰撞降低，但G1暂停在7,811更新/10.759epoch，只有6epoch完整评价，不能把G3@24e对G1@6e当监督设计的收益。G0已准备但未正式开跑，随后随nuScenes整体暂停；无G0成绩。G3@24e endpoint平均L2为1.1110m、box碰撞0.5836%，与prefix0.5848/0.2516分开解释。

G3@24e场景读出：全局道路MAE1.0679m、边界附近0.9062m、current/future IoU0.5139/0.3930、enter P/R0.2683/0.3298、release P/R0.3912/0.5578、静止障碍recall0.7224。GT车身查询89,442个，道路有效43,487、占用有效14,866；道路/占用代理MAE0.4755m/0.05638。有效覆盖不足不能隐去；视野分组是标定/ROI支持代理，不是真实遮挡GT。这些指标尚未建立与规划收益的稳定因果联系。[全时刻/可见性/局部指标](data/SCENE_SUPERVISION.json)保留。

## 推理步数与执行优化：收益和未通过项

历史固定 C1@100k 的 1/2/3/5/8/10/15/20/30 步完整 Navtest 均实际完成，PDMS分别88.1780/89.1379/89.2573/89.2175/89.3102/89.2395/89.2701/89.2632/89.2020。1步比10步 −1.0615，区间 [−1.3898,−0.7506]；3–30步并非单调更好。该实验不是训练加速验收，也没有授权把G真实10步q0改短。

固定64场景 FP32 batch1 时延测量（当前encode+Euler+decode，不含图片IO/模型加载）：1步0.951s、10步1.188s、30步1.730s。10步的solver全场景中位约0.267s，说明原DDP大部分延迟也在VLM编码。不要把它与在线BF16训练时延或G完整部署延迟混用。[9组数据和配对](evidence/fm_steps/RESULTS.json)、[逐场景数值](data/C1_FM_SWEEP_SCENES.csv.gz)均已归档。

历史 C0 的8卡实测中位1.812s/步（100测量步），16卡短测中位6.732s（21步后因成本停止，非完整基准）。这只能说明当时网络/执行组合更差，不证明任何16卡都慢。S4单8卡100测量步的中位为1.6903s（P95 1.7710s，源974c5c6）；[原profile](evidence/probes/NEW_MODEL_PROFILE_SUMMARY_v1.json)已补入。这些历史任务不包含G的在线GeoBEV/Bt/FGTR和训练期真实10步proposal，且精度/输入协议需按各自身份解释。

G完整profile显示proposal约0.38s、backward约1.62s；Qwen processor约268ms，视觉GPU约0.29s。P0把processor移到CPU worker后减少host gap，但视觉计算没有消失；普通数据等待已是毫秒级。冻结视觉缓存实测每场景含全部DeepStack为27MiB，101,592场景约2.62TiB，batch1与batch4输出有数值差；只做小样probe，未构建或采用全量缓存。

| 执行尝试 | 已执行证据 | 正式采用状态 |
|---|---|---|
| io_preserving_v1 | 完整原生继续训练与已接受IO路径 | G正式run仍用此模式 |
| engineering_fast P0 | Qwen长度元数据、CPU预处理、批量传输/日志、MLN/Fuser公共表达式、query/body复用；24个交替窗口 | 3.198–3.272s→2.847–2.857s，吞吐+12–15%；完整数值/规划验收未闭合，不能称无损已采用 |
| time vectorization | 真实Bt比较超过登记容差 | 关闭，原失败保留 |
| 局部BF16 precision_fast | 60-update筛选约2.608s | E3候选；非正式结果 |
| proposal K/V复用 | 局部321→304ms且逐位 | 小于整步1%，未采用 |
| no-recompute / selective checkpoint | 原失败guard与A/A分析保留 | 不以重命名解除；无正式承接 |
| host-only / 16GPU | CPU接口、源桥接、环境资格；16卡正式对照未完成 | 不宣称已加速或可无损换world |

24窗口共5280更新、进程成本41.2302GPUh，峰值allocated约31.90→30.58GiB。实际IO/P0窗口源是1e5b109，不能把它说成未经修改0c54f1c的直接A/B；源桥接也不是完整GPU数值证明。G1完成500步A/AA/B并有自然浮点漂移分析，仍不等于所有G1/G3早/中期、恢复与完整dev验收完成。详见[性能实测](../../structured_world/execution_20261011/PERFORMANCE_EVIDENCE_20261011_1405_ZH.md)、[实施与原失败](../../structured_world/execution_20261011/IMPLEMENTATION_AND_ACCEPTANCE_ZH.md)、[host-only/16卡/缓存后续](../../structured_world/execution_20261012/NO_REGRESSION_HOST_FAST_AND_16GPU_ZH.md)。

本次快照训练日志累计GPUh：NAVSIM G0/G1/G2/G3约610.80/614.76/553.76/708.67，nuScenes G1/G3约88.96/174.23。这些是已接受续段累计计数，**不含所有被丢弃的重放、预处理、几何准备、评测与性能pilot**；不能相加后称完整研究成本。最新32步的共享主机观测也不是公平隔离benchmark。精确计数、时间范围及实际LR见[TRAINING_STATUS](data/TRAINING_STATUS.csv)。

## 目前能支持什么、仍缺什么

可以支持：当前C分辨率扩展没有明确收益；S视频/动作memory尚未稳定超过S0；A_ACTION和S4的具体接口有负面证据；r1标签存在进度与安全取舍；激进/v2新目标的当前联合配方不值得按原方案继续；G同模型FGTR修正确有部署贡献，但新增监督并未稳定抬高历史规划上限。

不能支持：事件目标始终优于完整occupancy；局部监督一定有效；G−S的差值全由辅助监督造成；nuScenes已经证明跨数据集收益；仅凭辅助loss、单checkpoint或一个训练seed宣称结论稳定；未经闭合数值验收就称执行优化无损。

当前行动保持用户最新决定：只完成既有NAVSIM G2及已登记80/90/100k Navtest；G0/G1、nuScenes、r1/V12和新目标研究不自动恢复。若后续重启研究，最少需要关键第二训练seed、同容量current-only修正器、仅GT几何正则控制，以及去H_A/语言或置换未来场的依赖诊断；这些只是后续证据缺口，本次未启动。

## 实际训练源与报告源

| 范围 | 实际源码 SHA | 证据 |
|---|---|---|
| C正式 | d1d40854299b9599b2accc382bcfc4b676dd7623 | C/S COMPLETED_RUNS |
| S正式 | 1493deda247107efe076b9294863a6753391298a | 同上 |
| A_ACTION / A_NO_MAE | f0ed4c3608b8ae78ddb1a0e7a5cca78f9a4b9882 | 旧configs/av登记 |
| V_QUERY | 7c5231c514f2a327dabf934fbe3bd9a30f9b7c6c | 旧configs/av登记 |
| r1 S0 | ee2f086d7fcb94a6ee442ff8747e378d8be016e6 | 旧configs/optimized |
| r1 R/S3 | 32de37db070f0e4df2ce66f25e850bed2561828e | 同上 |
| 激进端点目标正式四组 | b3cefdd994146526cf70f259010fce5077fc5692 | evidence/objective_formal |
| v2冷启动四组 | 6ec45b0e8705c7919fb555c20324f91580add976 | 旧source_versions；a1e38a3只是后续序列化修复 |
| C1推理步数扫 | 771c802021e24d7ad3c3ff4f77c6de3824669edd | 非训练源；evidence/fm_steps/registration |
| G NAVSIM正式及模型推理 | 0c54f1c49e7ed2a76b22826b654d6c72625f5b6d | SOURCE_VERSIONS；canonical评分代码另有锁定 |
| G nuScenes正式 | 41da7c198cf0ac4f799aee2a27a0a8126de30230 | SOURCE_VERSIONS |

C/S完整结果源 e8b31ed、10月8日归档 e5c12fa、G旧报告267701f、前次汇总ed0725c均是报告/归档身份，不冒充训练源码。原始证据文件 hash 见 [SOURCE_INDEX](evidence/SOURCE_INDEX.json)，本次提交只在归档分支，不合并、不热改运行源。
