# Flow-RL来源、DDP基础设计与采样迁移复核

完整整理见 [2026-10-16 多来源分析详细报告](multi_source_analysis_20261016.md)：包含来源版本、配方与公式、LoRA/动作编码核查、20k正式结果、4096次诊断评分、两组各40,000次训练曝光的安全信号、成本和失败证据，以及后续单因素研究的边界。读者可直接从主文阅读索引定位问题，无需按开发记录顺序阅读。

14:03 UTC增补：[LAST_ODE固定5k完整结果](../../ddp_simwam_terminal_ode_20261016/stage005000/README.md)已完成DEV1696和定向TRAIN512。随机候选更安全，但原ODE分别为93.5608/79.6007，低于同期Canonical94.3121/84.4715。主文第13.5节同时保留这一反面证据；它不是Navtest或20k终点。当前文档的数字、链接和源码保护核验见[阶段交付校验](../../ddp_simwam_terminal_ode_20261016/stage005000/validation.json)。

2026-10-16最终补充：[X1完整Navtest](../final_navtest_20261016/README.md)原单候选ODE89.3817，相对SFT+0.4701分、相对Canonical+0.4300分；相对SFT的NC/TTC仍退化，92目标未完成。详细报告第13.4节已加入这一证据。下文的原SOURCE/CANONICAL及TRAIN采样诊断保持其原始比较口径。

初版[文档校验记录](documentation_validation_20261016.json)核对了124个JSON表格数值、53处本地链接、三个实际CLI帮助入口和既有源码hash；它绑定初版文档，不代表后续增补版本。先前整合见[历史校验](consolidation_validation_20261016.json)，含第8.4节的`b9207db`交付版见[当时校验](delivery_validation_20261016.json)。这些都是文档复核，不是新增算法验收或效果实验。

后续[完整TRAIN安全信号分析](../../ddp_simwam_terminal_ode_20261016/TRAIN_safety_signal/README.md)核对两组各40,000次实际曝光及全部违规正优势组：有安全替代的340/270条违规正优势全部低于组内最好安全候选，固定SFT的TTC失败人口占曝光1.84%。此次新增证据及当前文档hash见[后续校验](../../ddp_simwam_terminal_ode_20261016/TRAIN_safety_signal/validation.json)；前述校验保留为对应历史文档版本。

这是已有实验的源码分析与有限TRAIN诊断，不是新的算法训练矩阵。已经完成的正式X1使用锁定源码`368aba3`；以下检查不改变其actor、loss、随机窗口、奖励或部署。R_SOURCE/R_CANONICAL/X1均已完成20k及完整Navtest；本报告不把工程验收当作提分证据。

## 多来源对照的实际边界

| 来源 | 核实的采样/更新设置 | 对DDP有用的证据与限制 |
|---|---|---|
| [Flow-GRPO论文v1](https://arxiv.org/html/2505.05470v1) | SD3.5-M训练10步、评价40步；LoRA r32/alpha64 | “训练和推理都十步”不是通用配方。图像模型的0.7噪声不能直接解释成驾驶动作的合理扰动。 |
| [Flow-GRPO当前代码](https://github.com/yifan123/flow_grpo/blob/879042cf5707f8b90daa98d147d7deac2317c5da/config/grpo.py) | SD3 Fast部分配置10/40、前三至五步范围内选择3个随机步骤；Qwen-image部分配置10/50、窗口2 | 按具体配置区分版本。Fast的ODE前后缀和随机窗口与我们的固定末三步不同，不能只比较总步数。 |
| [SimWAM锁定配置](https://github.com/H-EmbodVis/SimWAM/blob/1433ee7e06a62bd8b89da74455df0d82bda7b726/configs/task/navsim_grpo_action_pdm_384x672_flowgrpo_lora.yaml) | 10步shift5、固定[7,8,9]随机，末步也随机；LoRA、global std、均值MSE锚定 | 我们迁移的是这一明确版本。当前默认配置与论文/发布权重的完整训练身份没有因此自动等价；不能用发布分数证明DDP移植必然有效。 |
| [DriveWLA v1](https://arxiv.org/html/2610.17903v1) | 论文描述G8、10个随机生成步骤、10步ODE部署；动作LoRA、折扣0.8、BC0.1、困难场景 | 原Flow-GRPO消融90.8→92.1，分项优势加shrinkage版本92.3；同时NC 99.2→99.1、TTC 97.2→97.1/97.0。增益主要在DAC/EP，不是所有安全分项同时改善。其5/10/15/20步SFT消融为90.6/90.8/90.8/90.7，更多步并非必然更好。未找到可核实的官方训练代码，具体sigma/clip不能补猜。 |
| [EponaV2 v1](https://arxiv.org/html/2605.14696v1) | 冻结上游、Flow规划器RL；奖励基于GT与实际模拟轨迹差异，含模仿项 | 消融89.4→90.4，TTC 95.3→95.7；并非我们的PDMS训练奖励。核实的官方仓库仅README/LICENSE，不能声称已核实LoRA范围或采样超参数。 |
| [MixGRPO v3](https://arxiv.org/html/2507.21802v3)及[实际采样代码](https://github.com/Tencent-Hunyuan/MixGRPO/blob/19946facaefa33eca1b565ec87c6fc7cca14e339/fastvideo/utils/sampling_utils.py) | FLUX主要实验总25步、随机窗口4，窗口外ODE，逐步移动；代码另有CPS分支 | 可借鉴“随机发生在哪里”的实验分离。锁定启动脚本实际`sde_type=sde`、`noise_level=None`回退eta=0.7；其注释“None for CPS”不等于实际启用CPS。论文文字中的shift公式与代码也不能混读，应以实际函数为准。 |
| [Flow-CPS v1](https://arxiv.org/html/2509.05952v1)及[官方kernel](https://github.com/IamCreateAI/FlowCPS/blob/b402c7a3d5b2539e6c42fed6c6c042bab9a1be05/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py) | 讨论少步离散SDE的噪声系数误差；CPS末端sigma归零 | 代码同时用负均方误差作为logprob代理，省略方差归一化。不能把它当精确Gaussian概率，或仅换采样器后沿用旧概率声称等价。本文公式的符号与实际代码也需区分。 |
| [GRPO-Guard](https://arxiv.org/html/2510.22319v1)及[实现](https://github.com/yifan123/flow_grpo/blob/879042cf5707f8b90daa98d147d7deac2317c5da/scripts/train_sd3_GRPO_Guard.py) | RatioNorm与生成步骤的梯度重加权，研究图像优化中的clip不平衡 | 提供逐步检查的理由，不证明我们也存在相同故障。精确密度比的旧分布期望为1；期望log-ratio为负本身不是bug。我们的H/D mean加clamp已明确是代理。 |

公开文件的commit、路径、SHA见[sources.json](sources.json)。未下载外部模型权重，未引入这些算法的新训练实现。SimWAM原有vendor和许可证保持不变。

## DDP代码确认与尚待证明的设计限制

1. **FM方向与实际动作编码一致。** 原SFT为`x_t=(1-t)*epsilon+t*y`，监督`y-epsilon`；原ODE按正dt积分。SimWAM适配采用`t=1-sigma`和相反velocity符号；已有上游oracle、实际采样mean/std对照支持这一点。没有新的正负号错误证据。
2. **原动作是当前ego坐标下的8个未来位置，不是逐点位移累加。** XY做z-score，sin/cos不另做z-score，解码使用atan2；四维中朝向径向幅值不改变物理角度。新增小测试确认对sin/cos同时乘正数会改变原动作空间误差而保持请求轨迹不变。这是表示中的冗余，不等于FM不能学习。
3. **十步的实际生成时间相差很大。** 原ODE访问bucket 0/100/…/900；shift5训练有梯度的三步访问318/444/642，dt约0.1263/0.1984/0.3571。共享LoRA也影响没有直接PG项的前缀及原ODE其他时间点。部分随机步骤是源经验代理，不是完整混合链的无偏梯度。
4. **末步噪声没有后续去噪。** 源标准差已经包含sqrt(dt)，三步分别0.052016/0.049801/0.044544。按DDP XY标准差8.805105m、2.277741m，末步单独注入约0.3922m纵向、0.1015m横向尺度；朝向影响取决于sin/cos半径。该分析是最后一次创新的条件扰动，不是整条SDE分布或最终仿真事故率。
5. **锚定与PG的时间尺度不同。** 相同velocity偏移在第9步造成的均值MSE约为第7步的8倍；`dmu/dv/std`约2.44/4.00/8.06。它说明相同beta不对应各步相同的velocity约束，但不能仅凭系数推出真实网络梯度方向。已有四阶段saved-batch证据显示冲突随batch变化。
6. **SFT时间分布和RL状态人口需要区分。** 原SFT使用Beta(1.5,1)变换，不是十个均匀离散时刻。连续训练与离散推理不要求分布完全相同。`noise_s=.999`带来约0.15%的轻微负时间尾部，是继承的SFT约定；没有证据将它列为当前主要原因，不在本轮顺手改掉。
7. **已有结果支持目标与泛化问题。** 全程有效PG约99.6%，零优势已不是主要瓶颈；困难清单偏低进度安全场景，原SOURCE/CANONICAL约66%/71%的组仍是progress-only。Canonical原ODE测试集仅+0.0401分，TTC新增219、恢复39；后续X1提高到+0.4701分，TTC新增87、恢复37。已有部分部署泛化收益，安全保持仍未解决。

另一个实际适配差异是**物理尺度下的锚定各向异性**。核实[SimWAM数据类](https://github.com/H-EmbodVis/SimWAM/blob/1433ee7e06a62bd8b89da74455df0d82bda7b726/src/simwam/datasets/navsim/navsim_dataset.py)发现，锁定absolute模式的实际反归一化尺度为XY **33.37m/21m**，并明确忽略配置中的stats文件；DDP为**8.805105m/2.277741m**。对相同1m的均值坐标偏移，原权重0.2对应纵向/横向锚定系数比约**1:12.63（源）**和**1:74.72（DDP）**。因此照搬0.2并没有保留同样的物理方向平衡；已登记X1约为1:14.94。这里比较的是相对方向权重，不是说DDP的绝对锚定更弱，也不是实际网络参数梯度之比。

反过来，源版本同一末步sigma对应的物理随机尺度约1.486m/0.935m，实际上比DDP更大。不能编造“DDP比源模型加了更大物理噪声”的解释。源模型、训练人口与发布权重的版本差异仍需分别看待。详见[action_scale_comparison.json](action_scale_comparison.json)；这是X1启动后的分析，没有用它改动运行权重，也不把事后解释写成事前选参依据。

以上源码hash见[local_source_hashes.json](local_source_hashes.json)。冻结前缀、LoRA覆盖、概率重算、评分隔离和恢复验收复用[已有审计](../../ddp_simwam_audit_recovery_20261015/README.md)，不重新包装成新算法发现。

## 有限配对诊断

从已登记TRAIN512用固定hash选64场景；SFT与R_CANONICAL20k各比较原10步ODE、shift5 ODE、真实末三步SDE。每场景各G8，匹配初始噪声，SDE创新使用独立CPU RNG并在两模型间配对。预算3072条候选，全部独立官方评分；不取best-of-N，不更新参数、不使用DEV/Navtest选择参数。

该诊断使用正式训练的FP32 root和源CPU RNG接口；不是历史正式单候选评价的BF16 root，不拿其绝对分数冒充Navtest。64场景/60 logs保留原诊断集的定向偏好，不作训练人口无偏估计。已完成两模型全部3072条评分，没有无效样本被丢弃。

| 同场景G8随机输出均值 | SFT PDMS | Canonical20k PDMS | RL−SFT / log聚类95%区间 | SFT→RL联合安全比例 |
|---|---:|---:|---|---|
| 原10步ODE | 85.2075 | 88.7307 | +3.5231 [−0.7573,+8.3235] | 90.04%→92.58% |
| shift5十步ODE | 85.6901 | 88.5427 | +2.8526 [+0.9740,+5.0608] | 90.63%→92.77% |
| shift5末三步SDE | 78.5166 | 80.0370 | +1.5204 [−0.3164,+3.4745] | 83.20%→83.40% |

**该TRAIN探针不支持“只有SDE学会、ODE完全未继承”。** 两种ODE也吸收了训练中的收益。但它不推翻完整Navtest上原ODE仅+0.0401及TTC退化的事实；训练人口上的吸收与测试泛化是两件事。

**当前SDE探索明显改变了样本质量。** 同一模型、相同root、相同G，SDE比shift5 ODE低7.1735分（SFT）/8.5057分（RL）；RL的DAC为89.06%而对应ODE为96.68%，TTC为93.55%而ODE为96.09%。这不是把SDE best-of-8和ODE单样本混比。

请求轨迹jerk诊断（m/s³）为SFT原ODE/shift5ODE/SDE **1.060/2.618/10.599**，RL为**1.711/3.017/10.805**。它不是官方Comfort，不能解释成同倍数的实际车身抖动。SFT pairwise ADE从原ODE的0.064m增至SDE的0.501m，这种几何多样性不能自动称为有用驾驶探索。正式模拟评分已显示其安全代价。

最后一次创新相对同一保存状态的SDE均值，实际请求XY RMS约**0.3943m/0.1010m**，heading RMS约**0.0441rad**。两模型使用相同创新，数值一致。完整SDE−ODE差异还包含之前的噪声与score correction，不能全部归因于最后一步。

朝向径向误差的均值在原ODE约0.0025（SFT）/0.0032（RL），所有被检查输出都没有sin/cos半径小于0.25；当前没有证据把“朝向半径塌缩”当主因。表示冗余是后续可测的限制，不能仅凭数学上存在就解释全部失败。

完整分项、配对安全转移、连续收益和代价见[sampler_transfer_train64.json](sampler_transfer_train64.json)。登记见[公开摘要](probe_registration_public.json)，原私有token和链留在本地。有效诊断实测82秒job跨度、613.6秒累计worker wall，含16个GPU worker的加载与评分等待；不是613.6秒端到端耗时或独占GPU时间。正式训练没有因此改变更新计划。

新诊断适配的三个smoke分别在评分前因输出目录、CPU/CUDA generator接口、配置返回值接口失败；已修复并保留[证据摘要](probe_smoke_failures.json)。这些错误只在新增探针中，原正式trainer已正确使用CPU generator和SFT配置，不属于旧实验失效证据。失败开销与成功诊断分开，首次输出目录失败前的精确worker耗时无法追溯。

## 仅移除最后一次创新的反事实

观察上述结果后，单独登记了[1024次附加评分](last_innovation_registration.json)：复用全部64场景、两模型、每场景8条已保存链，不筛选成功场景。保留前九步及最后的score-corrected均值，只将最终输出从`mu+std*epsilon`换成保存的`mu`。没有新actor前向、rollout或训练，不给这些输出附上伪造的on-policy概率。

| 固定保存链 | 原SDE PDMS | 去末次创新后的PDMS | 配对增益 / log聚类95%区间 | 联合安全通过率 | 请求jerk RMS变化 |
|---|---:|---:|---|---|---|
| SFT | 78.5166 | 85.9609 | +7.4443 [+2.9940,+12.0683] | 83.20%→91.21% | 10.599→3.191 |
| Canonical20k | 80.0370 | 88.8202 | +8.7832 [+4.6984,+13.0644] | 83.40%→93.16% | 10.805→3.536 |

这给出了**末次创新损害本批候选质量的直接配对证据**。RL模型中，移除创新后DAC新失败/恢复为4/44，TTC为8/22；仍有个别退化，不能称逐场景安全保证。它没有证明重新训练后一定改善单候选ODE，更不是新模型88.82的Navtest成绩。保留的SDE均值也不是原始ODE的最后一步，score correction并没有被移除。

结果见[last_innovation_counterfactual.json](last_innovation_counterfactual.json)。附加评分1024条，0新模型前向，136.1秒累计CPU worker wall、49.3秒job跨度（含先验证一个shard再并行其余shard）。合计本次有效诊断4096条真实评分，不包含已披露的评分前失败成本。

## 当前结论与已登记后续

当前判断：没有发现FM插值方向或正式概率重算的新错误；有证据表明本DDP上的末步随机扰动破坏了训练候选，且源归一化权重迁移后物理方向保持不同。X1已有真实原ODE测试集收益，相对SFT的NC/TTC仍退化。不能把这些不同层次的问题合并成“Flow本身不适合RL”，也不能将当前正结果说成92或全面安全成功。

X1只改变anchor原生权重，已完成固定20k及评价；没有混入新的采样器、奖励、课程或训练seed。另立的`R_CANONICAL_LAST_ODE`已登记、验收并启动：**保持G8/K10/shift5与三个随机转移，令随机窗口前移为[6,7,8]，保留末步Euler ODE**，不叠加X1。概率只覆盖真正随机步骤，保存链反事实没有被当作on-policy链用于训练。本报告尚无该变体完整部署结果；首100更新随机候选变好不能替代它。详见[独立登记及执行证据](../../ddp_simwam_terminal_ode_20261016/README.md)。没有引入CPS、GRPO-Guard或多seed矩阵。

## 已执行入口

以下worker必须位于实时授权的Tsinghua主机；注册和汇总是CPU管理操作。用新输出目录复跑，不能覆盖已锁定登记。原始私有根目录为`/mnt/project/simwam-x-anchor-20261015`。

```bash
python -m scripts.analysis.simwam_recipe.probe_sampler_transfer register --root /path/to/new_probe
python -m scripts.analysis.simwam_recipe.dispatch_sampler_probe --root /path/to/new_probe --source /path/to/verified/source --host training-vlawm-tsinghua-zt3 --model SFT --gpus 0 1 2 3 4 5 6 7
python -m scripts.analysis.simwam_recipe.dispatch_sampler_probe --root /path/to/new_probe --source /path/to/verified/source --host training-vlawm-tsinghua-zt4 --model R_CANONICAL_20000 --gpus 0 1 2 3 4 5 6 7
python -m scripts.analysis.simwam_recipe.probe_sampler_transfer summarize --root /path/to/new_probe
python -m scripts.analysis.simwam_recipe.probe_last_innovation register --root /path/to/new_counterfactual --parent /path/to/new_probe
python -m scripts.analysis.simwam_recipe.probe_last_innovation worker --root /path/to/new_counterfactual --model SFT --shard 0
python -m scripts.analysis.simwam_recipe.probe_last_innovation worker --root /path/to/new_counterfactual --model R_CANONICAL_20000 --shard 0
python -m scripts.analysis.simwam_recipe.probe_last_innovation summarize --root /path/to/new_counterfactual
```

counterfactual的worker命令在每个模型上执行shard 0—7；完整128个scene记录齐备后才能汇总。评分重入只复用hash一致的记录，旧失败和有效结果均保留。没有上传链、权重、缓存或场景图。
