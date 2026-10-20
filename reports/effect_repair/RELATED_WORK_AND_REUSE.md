# Action-effect repair: primary-source audit and reuse boundaries

审计时间：2026-10-20 UTC。代码起点：`36493208ba4d8c25bedfcc43935a4ce4527131cc`。本文是一手文献与公开源码核验，不是本项目的效果验收；无训练、模拟、权重下载或服务器任务。机器可读来源、实际读取文件及 SHA256 见 [PRIMARY_SOURCE_EVIDENCE.json](PRIMARY_SOURCE_EVIDENCE.json)。未识别的实现明确保留 `NOT_IDENTIFIED`。

## 可立即实施的结论

1. **“诊断失败、生成修正、把修正能力蒸馏为直接生成”已有明确先例。** Reflect-VLA 已做训练期特权反思与直接生成的自蒸馏，ELF-VLA 已做失败诊断、修正与 RL 回注。因此 EGR/IEG 的新贡献只能来自实证支持的成对干预效果、物理时间信用与匹配预算下的额外收益。
2. **终局 critic 梯度进入 FM 速度场也已有先例。** FlowDPG、DFQL 与 GuideFlow 各有不同实现与假设；不能把任意 learned effect gradient 接入原 DiT 就称为新方法，也不能继承它们在其他环境的正确性结论。
3. **原论文公式、公开代码和本地基线要分开核验。** 本报告没有将外部权重、奖励教师或代码加入模型。本轮默认继续使用本地已验证 FM/normalizer/Euler10；外部代码仅作阅读和独立实现参考。
4. **NAVSIM v1 的配对干预是固定日志条件下的模拟效果。** 它不是观测到真实世界的全部因果反事实。v2 可有 IDM 响应车辆，且 EPDMS 含跨帧/两阶段项；不能把单场景 v1 反馈重命名为 v2 完整 EPDMS。

## 核验身份与实现状态

日期来自论文提交记录或正式 proceedings，版本固定到阅读版本，不用搜索引擎的相对日期。`read` 指读取正文方法及相关附录/正式 PDF，而非只引用摘要。表中的公开源码状态只对本次锁定 SHA 成立。

| 工作 | 核实身份与版本 | 主要读取位置 | 官方实现与状态 |
|---|---|---|---|
| Reflect-VLA | Luo et al., arXiv **2610.13453v1**, 2026-10-09 | §3.1–3.4, Eq.1–10；Appendix D/E | [luo-yc17/Reflect-VLA](https://github.com/luo-yc17/Reflect-VLA/tree/693a45ebd91295c5eacb7c78a8908b78d914858c)，只有 README 和两张图；训练/推理代码未发布；未识别代码许可证 |
| ELF-VLA | Luo et al., *Unleashing VLA Potentials in Autonomous Driving via Explicit Learning from Failures*, **2603.01063v1**, 2026-03-01；CVPR 2026 | §3.2–3.4；反馈模板与 policy shaping | [luo-yc17/ELF-VLA](https://github.com/luo-yc17/ELF-VLA/tree/5eb04cc7484a435fa4c1e2fcd413a2766c910351)，4 文件、无算法源码；Apache-2.0 |
| GuideFlow | Liu et al., **2511.18729v3**, 2026-02-21；v1 为 2025-11-24 | §3–4, Eq.1–18 | 作者旧仓 [liulin815/GuideFlow](https://github.com/liulin815/GuideFlow/tree/3d2e36b85050918eade3e30f6a5aefae1dbbf109) 明确转向 [adept-thu/GuideFlow](https://github.com/adept-thu/GuideFlow/tree/878b1877a84f8c5305370468c9e9fbd3c91b38a1)；新仓有真实 Navsim 代码，但未发现其代码树许可证；旧仓 Apache-2.0 不能自动当新仓所有文件授权 |
| FlowDPG | Shi et al., **2606.22303v2**, 2026-09-30 camera-ready；v1 为 2026-06-21；CoRL 2026 | §3.1–3.2, Algorithm 1, Appendix A/B | [作者项目页](https://flowdpg.github.io/) 有论文/视频，没有定位到官方可执行算法仓；`CODE_NOT_IDENTIFIED` |
| WPT | Jiang et al., *World-to-Policy Transfer via Online World Model Distillation*, **2511.20095v2**, 2026-03-18；v1 为 2025-11-25；[CVPR 2026, 17842–17852](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_WPT_World-to-Policy_Transfer_via_Online_World_Model_Distillation_CVPR_2026_paper.html) | §3.1–3.4, §6.3, §7.2 | 正文未链接可执行官方仓；搜索到第三方复现不作为官方来源；`CODE_NOT_IDENTIFIED` |
| DFQL | Cao et al., *Direct Flow Q-Learning*, [ICML 2026/PMLR 306:11182–11205](https://proceedings.mlr.press/v306/cao26f.html), 2026-07-06–11；24 页正式 PDF | §2.2, §3.1–3.3, Eq.5–21, Appendix A/C | [2012060/Direct-Flow-Q-Learning](https://github.com/2012060/Direct-Flow-Q-Learning/tree/8b6b15803cc1e5605ad1524553a2c95e103b0bad)，MIT，`agents/dfql.py`；正文旧 URL `2012060/DFQL` 与当前仓名分开记录 |
| FPO | McAllister et al., *Flow Matching Policy Gradients*, **2507.21053v2**, 2025-08-01；2025-07-28 初稿；正式 ICLR 2026 | §3, Algorithm 1, Appendix A（尤其估计与优势处理） | [akanazawa/fpo](https://github.com/akanazawa/fpo/tree/418c2554f7cd22d52e14c07d951280929d73bf2f)，真实 PyTorch/JAX 实现；未识别全仓根许可证 |
| RFO | Zhong et al., *Reparameterization Flow Policy Optimization*, **2602.03501v1**, 2026-02-03 | §4.1–4.5, Eq.11–15, Appendix B/E/F | 论文链接的 Rewarped/mineral 是 simulator/baseline，不据此冒称作者 RFO 源码；`CODE_NOT_IDENTIFIED` |
| AWR | Peng et al., **1910.00177v3**, 2019-10-07；v1 为 2019-10-01 | §3, Eq.3–10, Algorithm 1 | [xbpeng/awr](https://github.com/xbpeng/awr/tree/831442fb8d4c24bd200667cbc5e458c7657effc2)，MIT；TensorFlow 实现 |
| DPO | Rafailov et al., **2305.18290v3**, 2024-07-29；v1 为 2023-05-29 | §4, Eq.4–7 | [eric-mitchell/direct-preference-optimization](https://github.com/eric-mitchell/direct-preference-optimization/tree/f8b8c0f49dc92a430bae41585f9d467d3618fe2f)，Apache-2.0；`trainers.py::preference_loss` |
| CHOMP | Ratliff et al., [ICRA 2009 author PDF](https://www.cs.cmu.edu/~mzucker/icra09-chomp.pdf)；扩展 IJRR 2013 | §II-A–F, Eq.1–4 | 此次只读作者论文，未将 ROS/MoveIt 第三方实现当原论文源码 |
| STOMP | Kalakrishnan et al., ICRA 2011:4569–4574, DOI **10.1109/ICRA.2011.5980280** | §III，平滑扰动、rollout cost、概率更新 | [作者实验仓](https://github.com/kalakris/stomp_motion_planner_icra2011/tree/2a30860e34136e9b9b7f18270ee37a5cf7837782)，真实 C++ 实现；已读 `policy_improvement.cpp` 的 BSD-3-clause 文件声明，其他文件仍按自身声明 |
| CPO | Achiam et al., [ICML 2017/PMLR70](https://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf), **1705.10528**, 2017-05-30 | §5, 约束回报/信赖域及近似求解 | 此次只读正式论文；不将其期望成本保证挪作单场景安全保证 |
| COMA / difference rewards | Foerster et al., **1705.08926**, 2017-05-24；AAAI 2018 | §4，替换单个行动、其余行动固定的 counterfactual baseline | 此次只读[作者论文](https://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/foersteraaai18.pdf)；不复用非官方 Python fork |
| World on Rails | Chen et al., **2105.00636**, 2021-05-03；[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Learning_To_Drive_From_a_World_on_Rails_ICCV_2021_paper.pdf) | §3，ego dynamics、日志环境、DP、policy distillation | 只读论文；无加载其教师/策略/权重 |
| NAVSIM v1 / v2 | Dauner et al., **2406.15349**, NeurIPS 2024；Cao et al., *Pseudo-Simulation*, **2506.04218v3**, 2026-03-20，CoRL 2025 | v1 非响应日志；v2 §3.1–3.2、Eq.1；官方 docs | [官方仓](https://github.com/autonomousvision/navsim)，Apache-2.0；本地实际 simulator SHA 由 `BASELINE_AUDIT.md` 绑定，不用此次上游 main 替换 |

## 学习对象、信息与成本的逐项区别

### Reflect-VLA：最直接的新颖性重合

其标准输入与反思输入分开；低质样本的诊断包含评估、专家对齐、改进指令。§3.2 使用 classical-planner/rule 候选及验证过的修正，§3.4 以分数路由高质自模仿与低质逆 KL。训练付出学生 rollout、评分、特权教师前向；部署只保留无反思学生。没有直接学习本文的同场景 `ΔF(z,a,δ)`，也没有通过真实局部动作干预确定物理 waypoint 的责任。它是离散轨迹 token 的 autoregressive 分布，token KL 不能原样用于本地连续 DiT。保留 B0 修正 FM 和高分 preserve，是必要对照。[正文](https://arxiv.org/html/2610.13453v1#S3)

### ELF-VLA：诊断修正不能算新发现

§3.3–3.4 将错误轨迹、GT、评分与任务信息用于训练期教师诊断；策略以反馈生成修正，回注 GRPO，并有针对低概率高优势修正的 policy shaping。它使用额外大教师产生文本诊断，本轮禁止把该教师/输出当自有物理差分监督。可借鉴失败类别和保留成功行为的实验设计，不能宣称自己第一次用 NAVSIM 反馈修复 VLA。部署时仍应区分带反馈修正路径与直接输入路径。[正文](https://arxiv.org/html/2603.01063v1#S3)

### GuideFlow：约束生成与本轮不相同

§4.3 的 CVF、CF、RFE 分别修改速度、积分状态、能量；Eq.14–18 不是对原 NAVSIM 碰撞函数的精确可微化。文中可用外部 GTRS 选约束 anchor，CF 使用 100 步/中途替换，RFE 允许额外精修；不能直接作为保持原 Euler10 的 IEG。实际 `V10TransfuserModel.forward_test` 读取 camera/lidar backbone，`ReflowTrajectoryHead.forward_test` 以 token 索引 anchor、100 步、CFG 双模型调用。变量 `gt_energy` 在该路径由**预测 BEV semantic map**计算，不能只凭变量名断言 GT 泄漏；anchor 来源仍须独立审计。默认不移植其网络或输入。[正文](https://arxiv.org/html/2511.18729v3#S4.SS3)、[实际函数](https://github.com/adept-thu/GuideFlow/blob/878b1877a84f8c5305370468c9e9fbd3c91b38a1/navsim_train/agents/flowdrive_unet/transfuser_model_v10.py)

### FlowDPG：可以借用推导边界，不能宣称精确优化

§3.2 先从共享 FM state 作一跳 clean-action estimate，在该点取 twin critic 梯度，归一化为相对演示速度的改变量，stop-gradient 后蒸馏目标，并用 consistency 约束估计点。Eq.10/Appendix A 明列近似：终点换一跳估计、传播 Jacobian 近似、积分换单时刻估计与尺度调整。critic 为动作价值，不是本轮事件差分教师。训练增加 critic/IQL 和梯度计算；部署保持多步 FM、不必保留 critic。默认不采用其 SARM reward、机器人视觉编码器或参数。[v2 正文](https://arxiv.org/html/2606.22303v2#S3)

### DFQL：Flow 时间不是物理时间责任

§3.2 Eq.13–18 把终点 Q 梯度投影为各速度步监督，放弃跨步 state sensitivity；Eq.16 权重为 `1/(K−k)`。`DFQLAgent.actor_loss` 实际每步 detach 积分 state，终点 Q 梯度广播，BC 则在相同离散 flow 时间密集训练。`k` 是 ODE 生成步，不是 8 个真实时间 waypoint。它不提供碰撞事件应归因哪段轨迹的证据。代码的 action clip、TD、固定均匀时间、梯度均值常数不能无审计移植到本地 Beta 时间分布。可独立实现 stop-gradient surrogate 对照，不能冒称精确 DPG。[正式全文](https://proceedings.mlr.press/v306/cao26f.html)、[实际代码](https://github.com/2012060/Direct-Flow-Q-Learning/blob/8b6b15803cc1e5605ad1524553a2c95e103b0bad/agents/dfql.py)

### WPT：世界知识蒸馏已有，但不等于真实干预差分

§3.3 的 reward model 对候选轨迹与 world-predicted dynamics 建模，§3.4 做 policy 与 world-reward distillation；教师/学生输入和推理成本分开。部署学生可丢弃 world model；该思想已覆盖“训练期世界/后果知识，部署轻量策略”的一般叙事。其学得未来与 reward 不能当官方模拟真值，也没有证明事件时间即行动责任。本文只复用训练/部署隔离与 matched student 对照原则，不增加新的 world model。[正文](https://arxiv.org/html/2511.20095v2#S3)

### FPO / RFO：与本轮的关系

FPO 以旧/新 CFM loss 差构造策略比率 surrogate，结合 PPO clip/优势；`FPO.learn` 保存相同噪声/时间，`DiffusionPolicy.compute_cfm_loss` 返回逐样本 CFM。不是精确 likelihood，不是细粒度效果预测。可将本项目已验证同预算直接 PDMS 组作为 RL comparator；不能因有奖励就把 EGR/IEG 称为 FPO。[论文](https://arxiv.org/html/2507.21053v2)、[源码](https://github.com/akanazawa/fpo/blob/418c2554f7cd22d52e14c07d951280929d73bf2f/gridworld/models/fpo.py)

RFO §4.1–4.4 则直接穿过可微 dynamics 与 FM 积分做 RPG，并加近期行为 CFM/均匀探索正则。NAVSIM 官方离散责任与 Shapely 几何不是此类可微 simulator，不能用 learned proxy 替换后仍声称原 RFO 保证。暂不移植其额外探索或 tanh action 定义。[正文](https://arxiv.org/html/2602.03501v1#S4)

### AWR / DPO：常规加权与偏好基线

AWR §3 把 `exp(advantage/temperature)` 作为行为监督权重；源码 `_calc_adv_weights` 有 clipping。将较好修正轨迹按价值加权训练，是成熟思路，并非本轮独立贡献。它无法学习观测数据支持之外不存在的修正。[正文](https://arxiv.org/html/1910.00177)、[源码](https://github.com/xbpeng/awr/blob/831442fb8d4c24bd200667cbc5e458c7657effc2/learning/awr_agent.py)

DPO §4 把偏好转为 reference-relative log-probability 差；`preference_loss` 的 chosen/rejected log-ratio 有明确含义。FM 的 MSE 差并不天然等于这些 log-probabilities。可借鉴独立 pair 标签和 reference 保持，不能直接换变量名声称得到 exact DPO。[正文](https://arxiv.org/html/2305.18290v3)、[源码](https://github.com/eric-mitchell/direct-preference-optimization/blob/f8b8c0f49dc92a430bae41585f9d467d3618fe2f/trainers.py)

### 轨迹优化、约束与反事实

CHOMP §II 以导数平滑性矩阵给轨迹改动定义 metric，整段传播局部 obstacle gradient；它支持平滑干预的设计，不支持用中心点距离代替官方整车/控制执行判定。[作者全文](https://www.cs.cmu.edu/~mzucker/icra09-chomp.pdf)

STOMP §III 用有相关性的平滑 rollout 扰动与 cost 概率更新处理不可微 cost。G0 在低维可执行支持上真实模拟，继承这一成熟搜索原则；不是“首次无需梯度进行修复”。每个样本及组合后的平均修正仍需重模拟，不能平均两条各自安全轨迹就认定安全。[作者论文身份](https://is.mpg.de/ics/de/publications/kalakrishnan_raiic_2011)、[原文镜像](https://whiteoak.umd.edu/roswiki/attachments/Papers%282f%29ICRA2011_Kalakrishnan/kalakrishnan_icra2011.pdf)

CPO 给的是假设下的策略期望约束及近似更新，不是逐案例零违规保证。本轮安全优先、进度/舒适约束、最小变化量的字典序检索应保持可审计，不用若干启发式门槛借用 CPO 的证明。[正式论文](https://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf)

COMA/difference rewards 把其余因素固定后替换某一行动，说明 credit 需要比较条件。这里可对应固定场景/背景日志/seed 后仅修改某段自车轨迹；但 COMA 的多智能体 advantage 不能当 8 点轨迹的精确因果分解。有限差分依赖基、半径及控制器，事件与干预支持的时间差才是本轮应验证的量。[作者论文](https://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/foersteraaai18.pdf)

World on Rails 已使用“ego 可控 dynamics + 环境日志不响应”的分解，通过离线行动价值蒸馏视觉策略。它明确给出本轮非响应日志下结果的解释边界：对固定日志有用，不等于已验证真实交通参与者会如何响应。[正式论文](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Learning_To_Drive_From_a_World_on_Rails_ICCV_2021_paper.pdf)

## 适配本地代码的数学卡片（实现推导，非外部论文效果结论）

### 成对效果目标与可检验恒等式

令 `a,b` 为同一场景、同一版本/seed/背景响应协议下成功模拟的两条轨迹。按实际有效字段定义 `ΔF = F(b)−F(a)`，PDMS/舒适等“越大越好”与碰撞事件等“越大越坏”分开，不先压成一个无量纲求和。

- 零干预：`D(z,a,0)=0`。
- 交换 pair：`D(z,b,a−b)=−D(z,a,b−a)`；首次事件不存在须保留 censored/valid mask，不能把“未发生”写成时刻 0。
- 连续幅度、离散事件转移和 rank 分项分别按有效量归约。排名指向已模拟候选，不把网络导数称作官方碰撞梯度。
- 预测不确定性须在未见 log 校准；高不确定/超支持输入可以回退原轨迹，回退比例也报告。
- 训练/评测同时包含有益、无效、有害以及修好旧事件却制造新事件的 pair。

这些是建议的工程契约，不是声称从文献得到未验证新方法。需要在匹配场景/参数量/标签计算预算下比较：`absolute(b)−absolute(a)`、终局 score、完整 absolute consequence、显式 pair-delta，以及 trajectory-only/shuffled-scene。

### 时间信用的可识别范围

同时保留 3 个时间轴：8 点提交轨迹时间、控制器执行时间、Flow ODE 时间。对基函数 `b_k` 在支持区间 `I_k` 的已模拟干预，记录事件变化 `ΔF_{k,t}` 与新生事件。以“原事件修复且无新生核心失败”定义正例；不是把 `event_at(t)` 当 `credit(t)`。

至少固定一组 held-out intervention coefficients/组合，在未见 log 验证；训练和测试只复用同一小组幅度容易退化为扰动模板分类。`NO_VERIFIED_REPAIR_WITHIN_BUDGET` 只说明已查询支持内未找到，不能解释为无解。G0 应同时输出预算曲线与高分破坏率，后者不能从失败集推断。

### B0 / B1 的共享路径要明确

以下仅适用于实际审计确认 `x_u=(1−u)ε+u a`、target `a−ε` 的本地参数化。时间采样仍继承原 Beta 与离散 bucket；若实际不符，先重新推导。

1. B0：在 `x_u^+=(1−u)ε+u a+` 上，用原 FM 监督 `a+−ε`，混合合法高质量 replay。这就是直接修正轨迹 SFT 对照。
2. B1-trajectory：明确选择共享 source bridge `x_u^-=(1−u)ε+u a−`，在**同一状态**给 residual 目标 `r*=a+−ε−stopgrad[v_base(x_u^-,u,z)]`，只能明确称为 **off-path 改进 surrogate**，不是原 source bridge 的真实 CFM 导数。只有原速度恰为 `a−−ε` 时，它才等于 `a+−a−`。训练 residual 常数差与补偿原场误差，是不同定义，不能混称。若改用从 source state 直达 `a+` 的一跳速度，目标是 `(a+−x_u^-)/(1−u)`，会出现终点奇异；它也不能在未登记 clipping/time 截断的情况下替换原目标。
3. 可选的 pair velocity relation `v_new(x_u^+)−v_base(x_u^-)≈a+−a−` 比较的是**两个不同 state**，不等于在单一路径任意 state 的 residual 目标。若使用，另立明确损失与对照。
4. B1-effect：冻结 effect teacher 只能提供训练期差分/事件/rank 监督；它不能读取本次样本真实反馈再作为部署条件。对 source-state 的一跳 clean estimate 或最终 ODE rollout 应指定实际哪一个接受 effect loss；不得在 noisy trajectory 上查询只训练过 clean pairs 的教师而不报告 OOD。
5. `alpha(u)` 与零初始化输出使 adapter-off 初始路径保持；这不提供训练后安全保证。必须验证指定参数有梯度、Base 冻结、共享噪声/时间、完整 Euler10 末端输出及 independent-process 导出。

### EGR 的效果引导不能退化为任意减速

普通 refiner A0 与 effect-guided A1 使用同样低维 basis、相同 frozen context、同样安全审计过的目标与 zero-repair 样本；只有冻结 effect 一致性/事件监督改变。监督目标优先按安全可行、合理进度/舒适、最小距离的顺序选，不靠加权总分补偿硬违规。

推理只读当前合法 context 与预测轨迹；真实 feedback、metric cache、GT、未来 actor 仅在离线标签侧。效果教师/不确定性模块若保留在部署图中，参数和调用数必须计入额外开销。

## NAVSIM 反馈复用与版本边界

官方仓明确 `main` 是 v2，`v1.1` 是 v1。v2 的 traffic docs 明确：ego 始终一次提交完整计划；可选 background log replay 或 reactive IDM，two-stage 默认 reactive。**不能将“ego 不重规划”误写为“所有背景交通也不响应”。** [官方 README](https://github.com/autonomousvision/navsim#pseudo-simulation-for-autonomous-driving)、[traffic 文档](https://github.com/autonomousvision/navsim/blob/main/docs/traffic_agents.md)

v2 增加 LK/EC、DDC/TLC multiplier、human false-positive filtering，EC 比较相邻输出帧；pseudo-simulation 用一阶段终点权重汇总二阶段合成观测。完整 EPDMS 不能从单条末端数值重建。v1 单候选分数与 v2 one-stage/完整 two-stage 必须显式分开。[指标文档](https://github.com/autonomousvision/navsim/blob/main/docs/metrics.md)、[v2 正文 §3](https://arxiv.org/html/2506.04218v3#S3)

本轮只在**本地锁定的官方版本**上提取 executed states、官方事件触发/责任、scorer 原有 per-time buffer；连续 clearance/jerk/执行跟踪误差等额外量标注 `derived`。必须保存提交轨迹和执行轨迹两个对象，背景响应逐候选重置；每条候选单独配固定 reference，避免候选池最大进度改变旧标签。最终能否重构官方分数由真实对齐测试证明，不由本报告宣布 PASS。

## 复用决策与最低可辨识实验

| 拟实现机制 | 已有先例/可读函数 | 本轮复用决策 | 必须用实验回答 |
|---|---|---|---|
| 低维平滑干预 | CHOMP smooth metric；STOMP rollout/probability update | 自行实现基础矩阵与运动学审计；不拷贝整套机器人 planner | 局部支持有多少 verified repairs；高分被破坏多少 |
| 成对 `ΔF`、rank、事件 | absolute critic/difference-reward 有相关基础；所审文献未确证完全相同合同 | 允许独立小模型；保留 absolute 差分同容量对照 | 同日志之外、未见幅度方向判断/top-k 真模拟是否更好 |
| B0 修正 FM | 监督模仿、Reflect/ELF 修正数据思想 | 直接复用本地原 FM loss/normalizer/sampler | 是否足以获得全部收益 |
| B1 residual velocity | FlowDPG target distill；DFQL velocity surrogate | 自行实现共享路径推导；不改采样器与动作维度 | 相同修正样本下是否胜过 B0/trajectory-only |
| 高质量 preserve | Reflect Eq.8；AWR/BC regularization | 同 noise/time 的 frozen S0 velocity preserve | 高分→低分/零分、新 NC/DAC/TTC 是否受控 |
| 训练期 world/effect、部署无模拟 | WPT、World on Rails | 复用信息边界，不加载其模型 | 无 teacher/score/GT/cache 新进程导出是否同输出 |
| 单标量直接 PDMS | 已有本地 C1/FPO-style 真实研究 | 复用匹配口径的结果与成熟代码，不能用历史不同人口作减法 | 额外标签成本是否换来可解释收益 |

当前只能说：文献身份、相关机制和实现边界已核实。**G0/G1/G2/G3 的科学命题均未由这次调研验证。** 若 G1 相对 absolute scorer 或 B1-effect 相对 B0 没有收益，保留真实负结果；“多了过程反馈”和“训练可运行”均不足以支持方法贡献。

## 本次取证边界

- 论文正文通过 arXiv HTML / 作者或 proceedings PDF 读取。DFQL 正式 PDF 经 PMLR 链接下载；浏览器因 `application/octet-stream` 无法解析，改用独立 `pypdf==5.1.0` wheel 只读解析 24 页，**未安装/修改训练环境**。STOMP 部分镜像 TLS/超时失败如实保留，使用可访问作者论文镜像/作者资料及源码核验。
- GitHub API 读取 commit/tree/license，raw URL 固定 SHA 读取必要函数。完整源码、PDF、wheel 只留在 `/mnt/project/action-effect-literature-audit-20261020/`，不入 Git。报告 JSON 仅保存小型来源/hash/状态，未保存大权重或私有数据。
- 搜索碰到的其他同名产品、非官方复现、错误 arXiv 候选均未作为支持来源。`NOT_IDENTIFIED` 指这次限定的一手搜索未识别，不声称世界上不存在实现。
- 本次没有直接复用任何外部算法文件或权重。若后续确需复制代码，必须再绑定实际文件/许可证和修改声明，而不是把“公开可见”当作复制许可。
