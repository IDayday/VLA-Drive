# 当前有效Qwen-Flow上的安全改进：论文复核与迁移边界

复核日期2026-10-24。此前检索并非只有“增加安全权重”：已有文档涉及DriveWLA、GDPO、Constrained GRPO、Dr. GRPO/DAPO及Flow采样一致性。入口为[旧多来源分析](../ddp_simwam_x_anchor_20261015/flow_design_review/multi_source_analysis_20261016.md)和[旧归一化研究](../ddp_flow_grpo_frozen_improvement/research.md)。本次另外锁定Plan-R1及Run-then-Walk公开源码，区分已经进入当前基线的设计与真正的新改进。

当前参照为原生Qwen＋源3维FM/Flow-GRPO：IL53k Navtest87.0160；最新CANONICAL46.2k为90.3808，SOURCE39.6k为90.0141。固定seed2101、单候选shift5 ODE、canonical v1.1。既有约3点净收益不等于安全保持，CANONICAL新增114个零分，同时恢复516个；因此现在适合检验安全方法。不能由此前无效直接证明其唯一原因是基础链路，也不能预设迁移后必然有效。[当前完整实测](../qwen_sourceflow_native_20261018/latest_navtest_20261024T113127Z/README.md)

## 1. 对应论文与当前实现

|来源|可核实的方法|与当前问题的关系|本轮处理|
|---|---|---|---|
|[DriveWLA v1](https://arxiv.org/html/2610.17903v1)|逐分项优势、组合归一化、PDMS组方差收缩|避免总分掩盖单项差异，降低微小差异驱动|正式迁移优势估计器，保留有效SimWAM采样/更新基础|
|[Constrained GRPO v4](https://arxiv.org/html/2602.05863v4)|先标准化分项，再用乘子组合；约束期望违规率|避免先合成奖励再归一化削弱乘子语义|正式迁移此前独立乘子ScAdv适配；并非原论文全部设置|
|[GDPO](https://github.com/NVlabs/GDPO/tree/4ad86b4fbfc5db594f3a2750ff9c39fdc8ee6115)|各reward独立归一化再组合；发布实现还有最终batch whitening|分项信号保留，与约束乘子设计相关|复用既有组件；约束版不保留最终whitening，明确为适配|
|[Run-then-Walk v1](https://arxiv.org/html/2609.25831v1)|进度阶段→安全/终点修复，阶段间切换reference|最直接讨论进度—安全调度|已核实公式与公开源码；独立后续，不伪装成当前常数λ方法|
|[Plan-R1 v3](https://arxiv.org/html/2505.17659v3)、[Dr. GRPO](https://github.com/sail-sg/understand-r1-zero/tree/dfca49dd460ee7cc8e4a5a162c876a7fd6993b87)|去逐组std，改为固定尺度或中心化|防止微小安全场景差异和高方差危险组被同等缩放|当前已用跨rank共同std；固定c仍不同，暂不列另一条长训|
|[DAPO](https://github.com/BytedTsinghua-SIA/DAPO)|动态组过滤等训练技巧|针对无信号浪费，不自动针对安全|当前非零PG约99.6%以上，无证据需优先重新补样|

## 2. DriveWLA：本轮正式组WLA_ADV

按论文Eq8/9及附录C：对NC/DAC/EP/TTC/Comfort逐组标准化，权重[2,2,5,5,2]；加权和除以该组合的组标准差，再乘s²/(s²+0.1²)，s²为原canonical PDMS组方差。两层方差都只在同scene的G8内计算。零PDMS方差会把该组更新压为零，即使其他分项仍有差异；它不能凭空修复全失败组。[原公式](https://arxiv.org/html/2610.17903v1#S4.SS3)

明确只迁移优势估计器：论文完整配方还有10个随机步骤、discount0.8、BC0.1及8 epochs；当前保留SimWAM末三步随机、discount1、均值锚定50、无BC及50k预算。该组命名WLA_ADV，不称完整DriveWLA复现，不把其论文92.3当本项目结果。未核实到可执行的官方训练实现，本处为论文公式实现加数值oracle。epsilon未由论文给出，固定1e-8；FP64稳定总体方差是明确数值约定。

## 3. Constrained GRPO：本轮正式组SCADV3

沿用此前已实现的核心：奖励与NC/DAC/TTC事件成本分别求组中心及std；A=Z_PDMS−Σλ_j Z_cost_j。奖励floor0.01，成本仅数值epsilon；组合后不再归一化、不clip5。独立OmniSafe Adam乘子使用原负成本差损失、非负投影，每份fresh未筛选数据更新一次；四个inner固定同一λ快照、链、old概率及优势。

当前适配并非原文逐项复现：v4论文采用归一化乘子、WAT任务目标、NAVSIMv2的NC/DAC/DDC及0.01违规阈值；本组保持canonical v1.1 PDMS作为主reward，约束NC/DAC/TTC事件，使用独立未归一化乘子和固定SFT随机参考预算。该差异使它成为Constrained-GRPO思想的明确迁移，不能借用原论文保证。[论文方法及NAVSIM设置](https://arxiv.org/html/2602.05863v4#S4)

预算来自512个固定公开难例TRAIN场景、4096条原SFT真实随机链：NC/DAC/TTC违规率0.041748046875／0.13818359375／0.13916015625。它们是随机策略的期望预算，既不是ODE违规率，也不是逐场景或绝对安全保证。原始PDMS、分项和请求/模拟结果保留，GT地图/对象/cost不进入actor condition。

## 4. Run-then-Walk：为什么不能写成“加个安全loss”

论文Run采用PDMS；Walk使用GT终点L1误差的分段reward（Δ=10、η=0.2）及NC+DAC，gate条件是NC>0、DAC>0。故NC=0.5仍可通过这个论文gate，TTC也没有直接进入Walk reward。第二阶段reference变为选定Run策略。论文AutoDrive-P3使用3 Run+2 Walk epochs，ReCogDrive为5+1；其效果不能套作当前预算或加速倍数。[方法及附录](https://arxiv.org/html/2609.25831v1#S4)

实际公开仓库已锁定`haha-yuki-haha/AutoDrive-P3_with_Run-then-walk@2f99d85782995f175d5a117eb45356bff34e343a`，读取reward、get_pdm、两个launcher及Apache-2.0 LICENSE。reward中的endpoint函数与上述参数一致；但launcher默认RUN_EPOCHS/WALK_EPOCHS均15，不能把GitHub默认15+15与论文3+2拼为“官方一致配方”。[锁定reward源码](https://github.com/haha-yuki-haha/AutoDrive-P3_with_Run-then-walk/blob/2f99d85782995f175d5a117eb45356bff34e343a/rl/examples/reward_function/autodrive_navsim.py)

若下一项迁移它，必须独立登记Run父checkpoint、Walk reference、GT终点只作反馈、阶段预算和导出base；不能只切reward而仍错误地以原IL为reference，也不能把LoRA增量合并到错误的base。可以复用已有CANONICAL Run前缀以节省重复计算，但目前没有启动此第三条训练。

## 5. 归一化与其它trick的优先级

Plan-R1活跃`compute_ae_process_supervision`确实中心化后除以固定scaling_factor，默认0.1；源码中另有逐组std分支，但不是当前rollout调用。已锁定`XiaolongTang23/Plan-R1@23bc54433599002f275c18b88cf261ab5e87d775`。其物理过程时间信用分配和反应式world model不迁入Flow生成步骤。[源码](https://github.com/XiaolongTang23/Plan-R1/blob/23bc54433599002f275c18b88cf261ab5e87d775/model/PlanR1.py)

我们当前分母是fresh全rank原始reward共同std，已避免“每个组都被拉到单位方差”；它仍随batch变化，故不等价于固定0.1。固定尺度会改变PG相对锚定的强度，需要独立控制，不能当纯数值修复。DAPO补样对当前极少数等分组未必省算力；clip-higher也可能增强进度漂移。末步ODE/Flow-CPS/learned-noise会改变探索核，保留为有证据后的独立对照，不与本轮两个优势方法捆绑。

此前DDP的X1还涉及4维编码与z-score下的物理锚定各向异性；当前已使用原生3维及源归一化，旧理由不完全适用。因此初始X1方案只保留已完成的短验收，不占本轮正式方法名额。没有恢复旧DDP实验。

## 6. 正式训练合同与验证边界

两个新组各从同一IL53k和相同零B LoRA初始化；公开难例17037 TRAIN，seed42，G8/K10，四次inner，LR5e-5，clip0.005，mean锚定beta50；各50k真实optimizer updates、12500 fresh全局rollout、100000scene曝光、800000新候选。保持原canonical单候选shift5 ODE/seed2101作为主评价；0/5k/10k/20k/35k/50k DEV，固定50k和DEV选择分别报告，最终完整Navtest不用于调方法。

源码复用、CPU算术、真实8卡梯度和恢复分别验收；不会把短训loss或候选最高值当安全改进。首次SCADV3真实16步短训通过，但inner恢复发现诊断λ快照已搬CUDA而比较tensor仍在CPU；修复比较device后，连续与中断恢复结果完全一致。WLA恢复另发现CPU/CUDA派生FP64统计最大3.56e-15的舍入差，限定只对派生诊断使用1e-12容差，保存的FP32优势仍严格一致。两次失败及耗时保留，没有正式run受影响。两组均已通过真实8卡恢复、所有480个LoRA张量实际更新及原始图片合并ODE验收，随后才从原IL53k开始正式50k训练。[资格证据](QUALIFIED.json)、[启动快照](formal_start_verified.json)

已有Navtest曾参与用户指定IL53k父模型及本轮研究问题的形成，不能称本轮设计完全未看过Navtest。当前两组没有用Navtest拟合乘子预算或训练参数；后续方法选择使用登记的DEV，终点与选择锁定后再测Navtest。既有Navtest结果仅为历史依据，不会被新优势计算或恢复诊断修复重写。

本轮原始资料、论文HTML及上游读取文件留在私有工作目录；Git只保存链接、hash、必要代码和聚合证据。`paper_source_lock.json`记录版本及实际读取文件，不提交论文全文、模型、场景、权重或cache。
