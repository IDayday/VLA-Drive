# Flow-RL 未获得有效 Navtest 提升：完成的原因诊断

本轮完成了真实诊断：读取保存的训练链与完整评价记录，重新生成并官方评分6,144条TRAIN候选，计算同链奖励反事实及真实参数梯度，并执行锁定SimWAM/DDP解码函数核对动作单位。**没有改变正式模型、旧配方或Navtest轨迹，没有新增训练seed，也没有启动新的长训。** [完成收据](COMPLETE.json)绑定全部结果文件；三个GPU诊断worker均已退出。

当前最强的证据是：**RL在目标困难人口上学到了收益，但损坏了原有的安全高分行为；这种退化已出现在TRAIN诊断集，不只是Navtest泛化失败。** 训练奖励与评价目标、DDP与源动作单位之间也存在已验证的差异。新证据不支持把失败简单归结为“没有优势”“LoRA不更新”或“只有SDE学会、ODE完全没继承”。尚未完成因果干预，不能宣布某一个根因已被独立证明。

完整Navtest主结论不变：A原生v1.1由88.9272降至87.8476；B由88.8244到88.7864，未达到92。B换头后的SFT能力没有严重损失，换头本身也没有解决安全保持，详见[固定终点报告](../ddp_simwam_fm_transfer_20261016/final_navtest_results_20261018.md)。

## 1. 安全高分人口的损失抵消了困难场景收益

使用**SFT自身结果**分组，门槛仍为原困难清单的0.9；不按RL结果重划集合。安全是NC/DAC/TTC全部等于1，NC=0.5不能算通过。下表为原生ODE、v1.1；增量为百分点。

| Navtest固定SFT人口 | A场景数 | A的RL−SFT | B场景数 | B的RL−SFT |
|---|---:|---:|---:|---:|
| 安全、PDMS≥0.9 | 8,887 | −2.5474 | 8,807 | −1.3901 |
| 安全、PDMS<0.9 | 2,310 | +1.1724 | 2,384 | +1.0552 |
| NC/DAC/TTC至少一项未完全通过 | 949 | +7.1844 | 955 | +9.7019 |

高分安全人口的配对95% log-cluster区间为A [−3.1740,−1.9073]、B [−1.7770,−1.0133]，均低于零。其中A新增526个联合违规，B新增334个。B这部分约占Navtest人口72.51%，其损失不能被少量困难场景均分上涨掩盖。

这种分化也发生在TRAIN诊断集：A的288个SFT安全高分场景平均下降4.3380分，新增22个联合违规；B的305个场景下降3.9854分，新增21个，B区间[−6.1691,−2.1421]。相应安全低分人口分别上涨3.6475/3.7090分。TRAIN诊断含定向样本，不能把其均分外推成完整TRAIN收益；但它说明问题不只在训练集之外。

完整DEV中B安全高分人口只下降0.1499分，区间跨零，困难人口收益更容易掩盖退化；这解释了DEV总体+1.1816而Navtest总体−0.0380的部分差别，不能据此重新挑选DEV或Navtest人口。

代码路径也吻合这一覆盖缺口：固定hard清单只含SFT score<0.9，`collection.collect`返回该场景的链，`SourceLoss._policy_loss`在同一链/条件上计算永久SFT均值anchor。没有GT replay或额外全局保持人口。**被排除的安全高分TRAIN场景，也没有直接进入这个anchor的人口。** 共享LoRA参数仍会改变它们的输出。

这是数据/保持覆盖的强线索，不是已经完成的“增加保持即可修复”的因果证明。这些集合由固定单seed的SFT结果定义，不代表所有噪声下稳定安全；按基线高分分组也存在分数上限和条件选择效应，不能单凭高分组下降就证明遗忘。实际新增官方违规、TRAIN上的对应现象和anchor数据路径共同支持优先检验覆盖问题。它也不说明安全失败完全没有训练信号；后面的优势统计明确显示违规候选仍承担了显著权重。

证据：[A分人口分析](A_fixed_cohorts.json)、[B分人口分析](B_fixed_cohorts.json)。

## 2. 损失集中在新增安全失败，并伴随更多前进

完整Navtest原生评价中，A新增的562个原SFT联合安全→不安全场景，贡献了全部负向得分损失G−的88.80%；B相应368个场景贡献83.46%。这些场景按**原10:5:2训练奖励也全部下降**。不能将它们描述为模型找到了逐轨迹奖励漏洞、或源奖励直接认为这些具体失败更好。

| 原生Navtest新增TTC失败 | A | B |
|---|---:|---:|
| 场景数 | 530 | 343 |
| 在首次官方违规时比自身SFT前进更多 | 529 | 340 |
| 同时刻前进位移增量均值 | +0.8443m | +0.6982m |
| 同时刻之前0.1秒平均段速度增加 | 528 | 342 |
| 该段速度增量均值 | +0.5854m/s | +0.4932m/s |

两轨迹共享初始状态和物理时间网格。首次TTC违规时间是官方判据首次失败的模拟时刻，不是预计碰撞倒计时，也不能映射为Flow第几个生成步骤。源/正式分数与请求/模拟后果分别保留。

两种ODE网格也有大量共同失败：B新增TTC集合中279个场景两种网格共有，native独有64、shift5独有30。这削弱了“仅换回shift5即可解决安全”的解释。

证据：[A完整安全事件](A_NAVTEST_safety.json)、[B完整安全事件](B_NAVTEST_safety.json)。

## 3. 奖励偏好确实改变总体目标，但不是只改公式就有保证

下表对**同一v1.1仿真反馈**做两种读出；这是10:5:2与5:5:2的差别，**不是v1.0与v1.1评分版本的差别**。

| 路线/网格 | 源10:5:2读出均值变化 | canonical5:5:2变化 |
|---|---:|---:|
| A原生 | −0.3841 | −1.0796 |
| A shift5 | +0.2713 | −0.1824 |
| B原生 | +0.4742 | −0.0380 |
| B shift5 | +0.5103 | +0.0472 |

即使每条新增失败都低于自己的SFT，源目标仍可能让大量进度收益在总体上抵消安全损失。B原生就出现源读出上涨、canonical基本持平。这是总体权衡失配的直接证据，不等于每条候选的排序都反了。

从每条路线500份fresh全局rollout抽取原8个rank、全部G8候选：固定collection 1、11、…、4991，共4,000组/32,000候选。这是预先固定的10% collection样本，不是全量训练；场景有重复曝光。保存链、old概率、优势与journal摘要逐项核验，原优势精确重建。

| 固定训练链反事实 | A | B |
|---|---:|---:|
| progress-only组 | 2,720/4,000 | 2,752/4,000 |
| 改为canonical后最高分候选索引变化 | 1组 | 0组 |
| progress-only中的稳健优势正负翻转 | 0 | 0 |
| 全部候选稳健正负翻转 | 509/32,000 | 552/32,000 |
| 正优势但仍违规的候选，source→canonical | 79→61 | 53→38 |
| 有安全备选却给违规候选正优势 | 52→34 | 39→24 |
| source/canonical优势向量余弦均值 | 0.9845 | 0.9817 |

奖励改变主要重新分配不同候选/组的更新权重。B的progress-only组占68.8%，但只占源裁剪后优势绝对质量的27.76%，canonical后为20.72%；不能用组占比声称68.8%的参数梯度都来自进度。违规候选的源优势绝对质量仍约32.33%，并非没有安全负反馈。优势质量也不等于参数梯度。

因此又运行了B的5k/20k真实保存批次参数梯度对照：相同模型、链、old概率、源PPO loss及anchor，只重新计算canonical优势；所有LoRA参数保持不变，没有optimizer step。

| B实际参数梯度 | 5k批次 | 20k批次 |
|---|---:|---:|
| source/canonical PG余弦 | 0.9282 | 0.8615 |
| 含anchor的总梯度余弦 | 0.9010 | 0.8745 |
| source PG与anchor余弦 | −0.5546 | −0.4506 |
| canonical PG与anchor余弦 | −0.5210 | −0.1646 |

优势向量很接近不代表参数梯度完全相同。奖励对齐有实际作用，值得保留为独立控制；但这只有两个固定全局批次，不能用它预测一条新的20k策略。此前原F-SFT上的历史R_CANONICAL完整Navtest原生88.9517，相对自身SFT仅+0.0401、区间跨零，也提示奖励改回canonical尚不足以单独证明RL成功；该旧控制不等于新B上的匹配训练。

证据：[A反事实](A_reward_counterfactual.json)、[B反事实](B_reward_counterfactual.json)、[同链真实梯度](B_reward_gradient.json)。

## 4. 匹配SDE/ODE诊断：没有发现“只在随机策略上学会”的明确证据

从原512场景TRAIN诊断清单按预登记hash规则取64场景/59 logs。每场景G8，SFT与RL、三种sampler共享初始噪声；SDE创新使用独立CPU RNG且在两检查点间一致。共6,144条新候选均经过独立canonical官方评分。这里是诊断均值，**不是新的单候选Navtest主成绩**；没有将best-of8当部署分数。

| 路线/采样器 | SFT均值 | RL均值 | 增量，百分点 [95% log-cluster CI] |
|---|---:|---:|---:|
| A原生ODE | 83.0549 | 86.3056 | +3.2507 [−1.0229,+8.4672] |
| A shift5 ODE | 77.0469 | 86.7784 | +9.7316 [+2.6263,+17.9140] |
| A源SDE | 71.4632 | 78.4876 | +7.0244 [+3.3331,+11.9549] |
| B原生ODE | 87.7618 | 91.7670 | +4.0052 [−0.1005,+8.9290] |
| B shift5 ODE | 88.1031 | 91.4519 | +3.3488 [−0.3233,+8.1712] |
| B源SDE | 78.1437 | 82.9936 | +4.8499 [+1.8322,+8.6210] |

先对G8取场景均值，再按log聚类；不能把512个候选当512个独立场景。B两个ODE增量区间仍跨零。B的“SDE增量−原生ODE增量”为+0.8447分，区间[−3.3443,+5.4052]；与shift5的差为+1.5011，区间[−2.6614,+5.8798]。没有足够证据确认收益主要丢在SDE→ODE转换，不能据此继续盲换sampler。

随机分布本身与ODE仍明显不同：B终点SDE均值比原生ODE低8.7734分，安全候选比例82.42%而原生ODE为92.58%。SDE有更多候选多样性，不等于更高的单候选质量。其有限G8可行场景数还从SFT的64/64变为RL的62/64，均值上涨不代表每个场景都改善。

全部保存链已核对：两checkpoint共享root，同checkpoint的SDE与shift5 ODE前七步完全一致。虽然只有后3步参与源PG，RL权重确实改变前7步的确定性前缀；这是共享LoRA的预期影响，不是新的代码错误。仅有中间latent漂移不能证明哪部分漂移造成安全失败。

证据：[A匹配采样](A_sampler_transfer.json)、[B匹配采样](B_sampler_transfer.json)、[事前登记](registration.json)。

## 5. 相同RL数字不等于相同物理配方：DDP动作单位仍不同

直接执行锁定SimWAM `1433ee7...` 的`denorm_odo`和本项目实际`deal_action_1225`，核对resolved源配置为absolute、normalize_action=true，以及DDP实际act_norm=1。

| 每1个归一化单位 | 本项目A/B保留的DDP编码 | 锁定SimWAM编码 |
|---|---:|---:|
| X坐标，米 | 8.805105 | 33.37 |
| Y坐标，米 | 2.277741 | 21.0 |
| 朝向 | sin/cos两个通道 | 归一化角度一个通道 |

对XY线性坐标，均值anchor的每米平方系数与`w_axis / scale_axis²`成正比。即便都使用纵向0.2、横向1，**同样1米变化的横/纵惩罚比，本项目为74.7189，源为12.6254**。这是相对各向异性差异。与此同时，本项目每米绝对惩罚在X/Y上分别是源的14.36/85.00倍；不能误写成“本项目的绝对纵向anchor比源更弱”。共同beta和权重归约分母在此XY比值中抵消。

最后一个随机步骤归一化std同为0.04454354，映射到请求XY的直接创新标准差却是：本项目 **0.3922m/0.1015m**，源 **1.4864m/0.9354m**。前两个随机步骤也有同类差异。这里没有评判哪种噪声更好，更没有把距离当碰撞概率。

shift5的最后3步不等长。相同保存状态下，源均值差为`1.005 * delta_sigma * delta_velocity`，因此均值anchor相对于velocity差的平方系数分别约0.01610、0.03976、0.12883，最后一步约为第7步的8倍。这能解释保持与PG作用为何不均匀分布在三个生成步骤，但不是漏实现；它正是保留的源公式。

这些证据说明：**换成源ActionDiT，仍不等于迁移了源动作坐标中的全部探索/锚定语义。** 本轮保留DDP表达是用户要求，不能中途改归一化破坏SFT身份。要改变这些单位或做物理匹配，必须另立配置与对照；当前没有证据保证这样改就能提分。中间状态anchor也不是最终物理轨迹的KL或安全保证。

证据：[原函数执行及hash](action_units.json)。

## 6. 当前判断与下一项单因素验证

| 候选解释 | 当前证据判断 |
|---|---|
| 大量无优势导致没有学习 | 不是这轮主瓶颈；有效PG>99.5%，真实梯度/输出均改变 |
| 换头严重损坏SFT | 不支持；B完整SFT Navtest与原F-SFT基本保持 |
| 奖励目标与canonical不一致 | 已观察到总体收益背离及真实梯度变化，是实质因素 |
| 学会困难人口，但原安全能力退化 | 目前最强线索，TRAIN/DEV/Navtest固定SFT分组均有证据 |
| 只有SDE学会，ODE完全未迁移 | 匹配小样本不支持这一单一解释；尚不能排除部分迁移损失 |
| anchor把PG全部压没 | 不支持；有冲突但总梯度仍有显著PG方向，不能据两个批次就减小beta |
| 直接搬参数已物理等价于SimWAM | 不成立；动作单位、朝向表示、探索/均值保持的物理尺度不同 |
| 已证明Flow本身不适合RL | 没有这种证据 |

**下一项优先验证保持项的人口覆盖。** 在B同一个已完成SFT、同一PG困难清单/数据顺序、源reward、G8/K10、LoRA、四次inner、原optimizer和固定预算下，仅把固定数量anchor场景中的一部分替换为原SFT安全高分TRAIN场景。仍使用同一种冻结SFT转移均值MSE，明确参考状态来自永久SFT，不能伪造成新PG候选。先做同链/同状态有限数值与梯度验收，再开展这一条独立配置的匹配学习；额外参考前向和缓存成本必须单列。

其目的很具体：检验能否保留原来安全高分人口，同时保留困难场景收益。门槛继续来自固定SFT TRAIN筛选，不从Navtest失败场景反向选样。不能同时改reward、纵向权重、sigma或LR，否则无法归因。奖励改为canonical应作为独立目标对齐控制；动作单位匹配是另一个已明确但尚未验证的因素，均不在本次诊断中混成一条新训练线。

这一选择是对新人口证据的研究优先级判断，不是成功保证。本轮**未启动**新的长训或恢复已暂停的旧追加实验。完成的是原因诊断，主协议有效且安全的RL提升仍未完成。

## 7. 实现、失败证据与成本

所有新增代码位于`scripts/analysis/fm_transfer/`；只连接已有actor、source loss、condition、decoder和评分器。CPU资格测试8项通过；真实GPU验证了源采样、首场景两checkpoint与既有shift5部署采样器输出一致、完整共同root/前缀、同链参数梯度及LoRA只读性。原正式源码与权重未修改。

新采样诊断第一次调用错误地向源`_randn`传了CUDA generator，而上游明确在CPU上生成再搬到GPU；正式trainer一直使用CPU generator，**这不是本轮发现的旧训练bug**。两个新诊断都在第一个SDE创新处退出，未完成场景；此前各完成16次ODE候选评分。失败目录和日志保留，[修复收据](sampler_rng_fix_receipt.json)记录额外32次评分开销。独立v2修正接口、通过新增RNG合同测试后完成，旧失败未覆盖。

有效新候选/官方评分6,144，额外失败前缀评分32；新optimizer updates为0。A/B采样worker主函数wall分别291.712/546.508秒，B真实梯度73.073秒；它们并行运行于授权tsinghua-zt三个空闲GPU。这些是含加载/等待的worker wall，不是独占GPU算力小时，也不应相加称端到端wall。失败尝试完整wall未捕获，保留为未知，不补0。CPU反事实约52.76/53.97秒，六个人口配对分析按路线约274.87/276.82秒。原始数据、链、condition与权重均未上传。

以下入口均已真实执行。再次复核必须使用新的输出路径；已有结果不可覆盖。GPU命令需先核对当前授权与空闲卡，不能仅凭本报告复用当时GPU编号。

```bash
# 以本分支工作树为cwd；这是实际采用的Python环境。
PYTHONPATH=. /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.fm_transfer.analyze_saved_reward_counterfactual \
  --run /mnt/project/fm-transfer-20261016/runs/B_SOURCE_ACTION_DIT_R_SOURCE_seed42_v1 \
  --output /mnt/project/fm-rl-failure-analysis-20261018/analysis/B_reward_counterfactual.json

PYTHONPATH=. /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.fm_transfer.probe_sampler_transfer \
  --run /mnt/project/fm-transfer-20261016/runs/B_SOURCE_ACTION_DIT_R_SOURCE_seed42_v1 \
  --registration /mnt/project/fm-rl-failure-analysis-20261018/registration.json \
  --output /mnt/project/fm-rl-failure-analysis-20261018/probe_B_SOURCE_ACTION_DIT_v2

PYTHONPATH=. /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.fm_transfer.diagnose_reward_gradient \
  --run /mnt/project/fm-transfer-20261016/runs/B_SOURCE_ACTION_DIT_R_SOURCE_seed42_v1 \
  --output /mnt/project/fm-rl-failure-analysis-20261018/analysis/B_reward_gradient.json

PYTHONPATH=. /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.fm_transfer.summarize_sampler_probe \
  --probe /mnt/project/fm-rl-failure-analysis-20261018/probe_B_SOURCE_ACTION_DIT_v2 \
  --output /mnt/project/fm-rl-failure-analysis-20261018/analysis/B_sampler_transfer.json

PYTHONPATH=. /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.fm_transfer.analyze_fixed_sft_cohorts --variant B_SOURCE_ACTION_DIT \
  --output /mnt/project/fm-rl-failure-analysis-20261018/analysis/B_fixed_cohorts.json

PYTHONPATH=. /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.fm_transfer.analyze_action_units \
  --output /mnt/project/fm-rl-failure-analysis-20261018/analysis/action_units.json
```

采样v2与GPU梯度的不可变执行源码为`f5ebd2a58c755afe9ce967189aa1ae250ec26a02`；CPU汇总文件各自记录分析源码hash。原先正式训练/部署身份仍由原报告绑定，不因为新增分析而改写。
