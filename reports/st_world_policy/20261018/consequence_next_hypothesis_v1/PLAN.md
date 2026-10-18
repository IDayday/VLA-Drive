# C1 boundary3584：优势来源与终局聚合的信息损失

状态：LEAD_AUTHORIZED_BOUNDED_READ_ONLY_ANALYSIS，尚未执行研究统计；不是训练批准。
独立证据Agent `/root/consequence_next_hypothesis`；起点 `336e41cda741b58d1a82f798ccd0902d17e88d94`。
唯一 owned path 为本目录。Lead 于本轮消息授权先登记、再实现并执行本分析。

唯一问题：同一 C1 实际梯度诊断人口中的非零优势来自哪些 canonical 分项，终局相同的候选是否保留不同事件过程？先回答这个问题，再决定 C5、G4/G8 或代理训练是否有必要。

固定 C1 `boundary_003584` 四个原 rank 的全部32 TRAIN场景、128候选、32 log。使用每场景原4条10步轨迹；不存在筛选、补样或删零分。已核实完整账本128/128有效、每条均保留道路41个原生时点与8段、责任碰撞/TTC事件与8段、raw/参考进度及v2固定参考合同。只核查了字段与完整性，未计算以下研究统计。道路方向的32场景来自 boundary4095，是另一个人口，不能把其全DAC=1套到3584。

真实入口：

`/mnt/project/st-world-policy-20261016/formal/consequence/runs/C1_fulltrain_seed42_official_4096_v1/official_feedback/a53a189d6603e316f5eaf2254fe85d3903e82ec27dce9167b143e6eddfff33c0/attempt_00/RESULTS.pt`

3,441,888 bytes，SHA256 `8019dd29b17efa79c1fd1a7211bdf09a0b5be93931db75cd4db95ad9a3fb122f`。REQUESTS、ADMITTED、COMPLETE和四rank的小JSON回执共同证明身份；不打开训练全manifest、checkpoint、图片、metric cache、DEV/Navtest或模型。源记录只读。

先锁定计算规则：

1. 核对128个request/result逐项身份及4×8×4排序，与四rank rollout_id、policy state hash、policy version3584、numeric runtime、参考input/context、保存的float32 reward对应。验证所有字段完整、有效，事件归约恢复DAC、NC、TTC。合法零分保留；任一身份/归约错误停止并报告。
2. 官方分数为 `R = M*(5*EP + 5*TTC + 2*Comfort)/12`，`M=NC*DAC`，DDC权重0。用完整分项重建R，绝对误差上限1e-12；这只复核保存结果，不调用scorer。EP保留每条 `[固定官方参考, 一条候选]` 的原归一化，禁止在四候选间重归一化。
3. 训练真正使用 `r=float32(R)`、`A=r-mean_4(r)`。非零优势以该实际float32计算精确判定。数学描述采用float64项 `b_EP=5*M*EP/12`、`b_TTC=5*M*TTC/12`、`b_Comfort=2*M*Comfort/12`，各自在scene内中心化为a。保留舍入残差 `a_round=A-(a_EP+a_TTC+a_Comfort)`，所以分解精确。报告 `V=sum A^2` 与 `S_k=sum(A*a_k)/V`（含round项）；V=0时S为undefined。S可为负或超过1，不能裁剪，也不是梯度份额/因果贡献。M变化同时影响所有项，禁止将乘积TTC项变化一律归因为TTC变化。
4. 对每scene报告NC/DAC/TTC/EP/Comfort是否变化，及C1/C2/C3/C4非零组优势。`EP-only`限定NC/DAC/TTC/Comfort在4候选间都相同、M>0、EP有差异且实际C1优势非零；它才支持这个batch的总分反馈只由进度区分。其余按分项变化标志如实报告，不强制单因归属。分母始终32scene/128candidate；各项零优势、all-safe/all-failed、零分门控分别保留。
5. 每scene枚举6个无序候选对，共192对。分别报告：(a) 实际训练C1 float32 reward精确tie；(b) 安全终局向量 `(DAC,NC,TTC)` 精确tie；(c) 每个D/C/T终局分项tie。对tie对检查对应的原生事件时间、8段数组及事件严重性时间序列是否不同。先报告各条件分母，再报告事件非空分母与差异数；全空事件pair不称有过程信号。原始float64总分tie另列，防止float32量化造成的假同分解释。
6. 道路使用41点布尔序列及8段；责任碰撞使用at_fault事件的 `(state_index,nc_contribution)` 和segment_nc_min（保留0.5/0与后续更严重事件）；TTC使用violation事件的state_index，以及单独的 `projection_offset_steps/observation_index`，不能把TTC预测offset当成真实碰撞时刻。时序主判据只使用物理state_index序列；对象ID差异不是时序差异。`None`表示无事件，不能用4秒/inf/0填充；8段右闭分箱 `[0,.5],(.5,1],...,(3.5,4]`。
7. 这个诊断只能证明信号支持与终局聚合是否丢失已记录的时间信息，不能证明过程监督的因果方向。合成反例：十步共同生成两个物理轨迹点，终端事件依赖前后两点；验证对全部十步均有非零导数，反驳按物理时刻只更新同编号Flow步/最近trajectory token。另验同终局不同时间、全空事件、先0.5后0严重性、跨段边界、常数reward和float32量化tie。

预登记假设与停止条件：

- H1：此固定batch中C1的非零优势主要来自EP；这是待检假设，不能由61.75%历史曝光率推定。只有实际EP-only计数与协方差分解支持才作局部描述；若安全/Comfort变化则保留反例。
- H2：终局安全同分掩盖不同事件时序；只要存在相应非空tie且时序不同即可证明该batch有信息损失，但不能称改进。若事件全空或非空tie=0，结论为本人口无可识别过程对照，不补采样、不改reward、不训练预测器。
- 当前梯度分解仍待Lead提供；本分析不推断BC压过policy。届时只按同batch读结果，不能把优势方差当参数梯度强弱，更不能当ZeRO2/Adam更新。
- 任一身份/分母/时序归约错误、超过读取/CPU/输出上限立即停止；不重试评分、不替换scene、不扩大预算。

最小资源：一CPU进程、线程1、GPU0、模型/梯度/优化/采样/新评分均0；最大120 CPU秒，累计任务输入逻辑读取32MiB，输出4MiB。共享环境不安装依赖。源文本与小回执/已有反馈是唯一输入；第三方库代码/动态链接器页不属于数据资产读取计数。记录实际CPU/wall、peak RSS与读取账本。

后续边界：优先完成本次只读事实；无条件新增C5/G8/DFQL都不成立。G4/G8须新的固定snapshot配对信息收益诊断，且区分G与FPO MC50；当前API原生G4。DFQL须事件相关人群的真实执行方向证据以及共享DiT参数适配验证；已有道路7/7代理下降、官方PDMS轻降且道路修复分母0未满足该目的。所有正式训练仍由Lead登记并独立正确性/实测效率双审。
