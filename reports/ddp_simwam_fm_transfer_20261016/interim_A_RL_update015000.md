# A组RL15k：部署均分保持增益，TTC与DAC出现不同方向变化

A_DDP_TIME的15,000 optimizer updates固定观察点已完成完整DEV与固定TRAIN诊断。参考是本组20k SFT终点；DEV为1696场景/16 logs，TRAIN诊断为512场景/343 logs。使用原始观测、seed2101、单候选10步ODE、canonical NAVSIM v1.1；原生均匀网格和shift5分别报告。

**DEV原生PDMS为94.474666，相对自己的SFT提高0.754574个百分点；shift5为94.513221，提高1.052911个百分点。** 两项对自身SFT的log聚类配对区间均高于零。相对10k，原生略降、shift5略升，但两项区间均跨零。TTC新增违规减少，原生DAC退化增加，不能将均分增益解释成全面安全提升。继续固定20k终点，不选择中期最高点；最终Navtest尚未执行。

## 单候选部署成绩

分数用0–100，差值为百分点。置信区间按log聚类重采样，保持场景权重；只覆盖固定训练seed42、推理seed2101下的数据不确定性，不含训练或推理seed的不确定性。

| 人口/网格 | 自身SFT | RL5k | RL10k | RL15k | 15k−SFT，95%区间 | 15k−10k，95%区间 |
|---|---:|---:|---:|---:|---|---|
| DEV 原生 | 93.720093 | 94.332347 | 94.627902 | **94.474666** | **+0.754574** [+0.365713,+1.130394] | −0.153235 [−0.567143,+0.186611] |
| DEV shift5 | 93.460310 | 94.384140 | 94.381735 | **94.513221** | **+1.052911** [+0.586089,+1.724847] | +0.131486 [−0.049446,+0.464312] |
| TRAIN诊断 原生 | 81.835774 | 84.254122 | 86.575159 | 85.890533 | +4.054759 [+1.920457,+6.207336] | −0.684626 [−2.168881,+0.752765] |
| TRAIN诊断 shift5 | 80.607056 | 83.998308 | 86.701362 | 85.443626 | +4.836570 [+2.128305,+7.665553] | −1.257736 [−3.057119,+0.618026] |

TRAIN诊断包含定向困难样本，不是完整TRAIN无偏估计。DEV未用于RL更新；原SFT可能见过DEV，不能称基模训练完全未见。

同一15k权重的DEV原生−shift5为−0.038554个百分点，区间[−0.687052,+0.286855]，不足以判定网格优劣。原生仍是主要部署协议，不能把两列较高值作为单一成绩。

## 安全保持与新增失败

| DEV分项，0–100 | 自身SFT 原生 | RL10k 原生 | RL15k 原生 | 自身SFT shift5 | RL10k shift5 | RL15k shift5 |
|---|---:|---:|---:|---:|---:|---:|
| NC | 100.000000 | 100.000000 | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| DAC | 99.292453 | 99.764151 | 99.469340 | 99.056604 | 99.469340 | 99.528302 |
| TTC | 99.764151 | 98.761792 | 99.233491 | 99.587264 | 99.233491 | 99.587264 |
| EP | 86.142845 | 88.746115 | 88.319388 | 86.050026 | 88.096352 | 87.975597 |
| Comfort | 99.882075 | 99.823113 | 99.823113 | 99.823113 | 99.823113 | 99.823113 |
| DDC | 98.879717 | 98.850236 | 98.879717 | 98.879717 | 98.850236 | 98.909198 |

下表为**15k相对自己的SFT**。安全通过要求NC/DAC/TTC分别为1，NC=0.5不计完全通过。TTC违规不能直接写作碰撞。

| 人口/网格 | NC新增/恢复 | DAC新增/恢复 | TTC新增/恢复 | 非零→零/零→非零 | 固定SFT-safe中新联合违规 |
|---|---|---|---|---|---|
| DEV 原生 | 0/0 | 6/9 | **13/4** | 6/9 | **19/1681** |
| DEV shift5 | 0/0 | 4/12 | **3/3** | 4/12 | **7/1674** |
| TRAIN诊断 原生 | 1/0 | 5/24 | 12/4 | 5/24 | 16/440 |
| TRAIN诊断 shift5 | 1/0 | 14/36 | 9/7 | 14/36 | 22/432 |

相对10k，DEV原生TTC新增1个、恢复9个，但DAC新增5个、恢复0个，零分从4增至9。联合安全场景1671→1674，仍低于SFT的1681；原生1%分位仍为58.333333。总安全数改善与零分增加可以同时发生，不能用单一计数替代各分项转移。

shift5相对10k的TTC新增2个、恢复8个，DAC新增3个、恢复4个，零分9→8，联合安全场景1674→1681。虽然TTC均值恢复至自身SFT水平，仍有3个原通过场景变为违规、另3个恢复，不能声称逐场景安全保持。原生和shift5当前零分分别9/8，固定单seed口径不涉及“任意seed失败”等其他定义。

DEV原生G+=23.544982、G−=10.747408，shift5为23.025904/5.168533；单位为原始0–1分差的场景求和。原生任意改善1038、任意退化55；达到±0.005实用幅度的改善466、退化20。shift5相应为1003/89和438/7。任意小负差不是事故计数。

## 已保存请求与模拟轨迹的诊断

只读分析既有轨迹，新增推理、评分和参数更新均为0。DEV原生13个新增TTC来自3个logs，首次官方违规分别在物理3.8秒3个、3.9秒5个、4.0秒5个；shift5的3个来自1个log，均在4.0秒。初始模拟状态严格相同，全部新增TTC场景的NC完全通过。

在首次违规的同一物理时刻，原生13/13、shift5 3/3相对SFT前进更多，之前0.1秒的平均段速度也更高：

| 新增TTC子集，配对均值 | 原生 | shift5 |
|---|---:|---:|
| 同时刻前进位移差 | +0.661150 m | +0.395937 m |
| 前0.1秒段速度差 | +0.507257 m/s | +0.371447 m/s |
| 请求点离散jerk RMS，SFT→RL | 0.486299→2.235715 m/s³ | 0.861881→1.263296 m/s³ |

请求点jerk是几何诊断，不是官方Comfort或模拟车辆瞬时jerk。物理3.8秒也不是Flow生成步骤。两网格共同新增TTC有3个，另有10个仅原生新增；这是固定seed描述，不构成随机稳定性证明。

固定SFT-safe中新联合违规贡献原生G−的98.31%、shift5的98.57%。同时，454/424个场景在两策略均安全且PDMS至少提高0.005，其模拟终点前进量平均增加0.678979/0.525305米。问题是场景相关的安全进度权衡，不能将更多前进全部认定为无益。

对同一仿真反馈重读源10:5:2奖励，全DEV平均变化为+1.172800/+1.309576个百分点；新增TTC子集的源奖励全部低于SFT，没有源奖励升而canonical降的配对场景。这排除了这些具体ODE失败被源奖励直接评价为改善的解释，但不排除源权重在G8组内排序和共享参数更新中的作用。

四个人口/网格组合的全部分项均值、PDMS增量、G+/G−及安全转移与独立配对汇总核对一致，数值容差1e−12。结果条件化子集只支持描述性判断，不据此中途改变reward、anchor或噪声。

## 更新信号、真实人口和梯度

八rank已提交前缀与原生15k恢复点一致：15,000次全局optimizer更新、3,750份fresh全局rollout、30,000次scene曝光、240,000条新候选。覆盖全部21,367个困难场景/1064 logs；8,633次重复曝光中1次是源末批补齐，8,632次为正常第二epoch重访。全部30,000个组的链摘要不同，重访重新采样，不复用上epoch旧链。

- 有PG的组29,856，占99.52%；144个等分组由22个全满分、37个全零分和85个其他等分组成。全局anchor-only optimizer更新为0，不能用某个rank的局部计数冒充全局跳步。
- progress-only组20,110，占67.0333%；安全分项有差异的组7,430，占24.7667%。有效优势多并不意味着安全探索已充分。
- 正优势候选130,638个完全安全、676个仍违规，其中536个组内有完全安全替代。这是候选计数，不是梯度贡献比例，也不能直接对应DEV失败。
- 第一inner ratio最大误差和PPO clip比例均为0；第2/3/4个inner平均clip比例0.304722%/2.475000%/6.176111%。仍使用源clamped dimension-mean代理，不称完整轨迹精确概率比。

训练困难人口上的随机策略canonical均值82.836678、源奖励78.819832；有限G8安全候选最高值均值90.761439，仅定义于29,892个存在安全候选的组。这些训练统计不与完整DEV单候选成绩混成一张策略排名表。

对第3750份已保存批次、inner4后15k权重重新分解梯度：

| Flow步骤 | PG norm | beta50 anchor norm | cosine | 当前相对旧概率PPO clip |
|---|---:|---:|---:|---:|
| 7 | 0.016916 | 0.000525 | −0.044380 | 0% |
| 8 | 0.030993 | 0.001903 | −0.145817 | 0% |
| 9 | 0.044541 | 0.016090 | −0.471652 | 10.9375% |

最后一步anchor约为PG norm的36.1%，方向部分冲突；前两步较弱。10k对应批次为46.2%，但场景和链不同，不能据此解释整个训练走势。分解与源总loss梯度最大绝对差4.22e−10；29项训练依赖hash匹配，全部adapter保持不变。没有据此调低anchor或放宽PPO clip。

## 成本、源码与执行

15k训练循环wall为12,753.792秒（212.563分钟），模型/DDP准备31.985秒另计；10k→15k为4104.547秒（68.409分钟）。累计循环吞吐1.1761 updates/s。rank0累计checkpoint保存78.970秒、更新7664.238秒、rollout1949.758秒、旧概率609.265秒、评分等待467.775秒；scorer elapsed227.391秒不是CPU核秒。各rank时间重叠，不能相加成独占GPU-hours。

| 15k评价 | 两网格逻辑候选 | 实际推理含入口检查 | job wall秒 | 累计worker wall秒 |
|---|---:|---:|---:|---:|
| DEV | 3392 | 3400 | 372.167 | 2957.848 |
| TRAIN诊断 | 1024 | 1032 | 199.613 | 1582.491 |

两项分别在tsinghua-zt和vla-zt并行，不能将job wall相加充当端到端耗时。16个worker均已核实退出、16项原入口检查通过，完整分母和账本保留。此次没有手工释放占卡进程，其他真实/未知任务保持。保存批次梯度诊断另用38.826秒、峰值7,387,997,696 bytes，0新候选/评分/optimizer更新。

训练仍为`225b9af660cb0210208e2c46fc4216d0f151275b`，数值评价仍为`45d2318d2dec265583860a26f891d8b79bb6068a`，保存批次诊断为`fd440275edc3fdf40b2aaa46c38da1d038068902`。原生checkpoint `update015000_inner0`为world8、phase READY，不是20k完成。

合并权重SHA256为`4fa4bb8d60b38cb23dd20a51378599e3505ad1343c98b0346b530b51f2a31ea6`；96个attention投影改变，冻结前缀保持。恢复仍须使用原生LoRA/Adam/RNG checkpoint。里程碑调度接入本机是执行调度变化，没有改变两组配方或数值入口。

20k保存批次诊断的单次等待任务已登记，只在封存终点后读取固定链，不新增候选；发布本报告时仍等待。完整终点预算汇总由独立的已登记CPU等待任务负责，不能将等待状态写成已完成实验。

## 本阶段判断与可复现证据

A组已连续三个观察点在真实单候选ODE上取得相对自身SFT的净收益，不能再用“没有有效更新”或“只有候选库提高”解释当前结果。10k→15k没有可靠的均分方向证据，安全变化也不单调：TTC部分恢复、原生DAC退化增加。主要剩余问题是保住原安全场景，而非制造更多非零优势。

仍不能把收益单独归因于时间分布或ActionDiT：A包含新增SFT曝光，B仍未完成自身SFT和RL。继续既定终点和匹配评价，不新增算法或seed。DEV94.5不代表Navtest已经超过92。

[DEV对自身SFT](A_RL_update015000_DEV_vs_own_SFT.json)、[DEV对10k](A_RL_update015000_DEV_vs10k.json)、[TRAIN对自身SFT](A_RL_update015000_TRAIN_DIAGNOSTIC_vs_own_SFT.json)、[TRAIN对10k](A_RL_update015000_TRAIN_DIAGNOSTIC_vs10k.json)、[DEV安全转移](A_RL_update015000_DEV_safety_transitions.json)、[TRAIN安全转移](A_RL_update015000_TRAIN_DIAGNOSTIC_safety_transitions.json)、[网格差](A_RL15k_DEV_native_minus_shift5.json)、[训练前缀](A_RL_update015000_training_prefix.json)、[人口审计](A_RL_update015000_population.json)、[梯度](A_RL_update015000_saved_batch.json)、[执行与成本](A_RL15k_execution.json)。

以下入口均实际执行，输出已存在时不要覆盖；重复只读分析应使用新的输出身份。TRAIN对应命令使用`TRAIN_DIAGNOSTIC`路径。所有权重、原始轨迹、缓存与私有完整日志保留在资产目录，未上传仓库。

```bash
python /mnt/project/fm-transfer-20261016/resources/analyze_A_RL_15k_prefix.py
python -m scripts.analysis.fm_transfer.analyze_rl_population --run /mnt/project/fm-transfer-20261016/runs/A_DDP_TIME_R_SOURCE_seed42_v1 --manifest /mnt/project/fm-transfer-20261016/hard_manifests/A_DDP_TIME_SFT_update020000.json --updates 15000 --output /mnt/project/fm-transfer-20261016/analysis/A_DDP_TIME_RL_update015000_population.json
python -m scripts.analysis.fm_transfer.diagnose_saved_batch --run /mnt/project/fm-transfer-20261016/runs/A_DDP_TIME_R_SOURCE_seed42_v1 --checkpoint /mnt/project/fm-transfer-20261016/runs/A_DDP_TIME_R_SOURCE_seed42_v1/checkpoints/update015000_inner0 --collection 3750 --hard-manifest /mnt/project/fm-transfer-20261016/hard_manifests/A_DDP_TIME_SFT_update020000.json --sft-screen /mnt/project/fm-transfer-20261016/screens/A_DDP_TIME_SFT_update020000 --output /mnt/project/fm-transfer-20261016/analysis/A_DDP_TIME_RL_update015000_saved_batch.json
python -m scripts.analysis.fm_transfer.analyze_evaluation --evaluation /mnt/project/fm-transfer-20261016/evaluations/A_DDP_TIME_RL_update015000_DEV --reference /mnt/project/fm-transfer-20261016/evaluations/A_DDP_TIME_SFT_update020000_DEV --output /mnt/project/fm-transfer-20261016/analysis/A_DDP_TIME_RL_update015000_DEV_paired_own_SFT.json
python -m scripts.analysis.fm_transfer.analyze_evaluation --evaluation /mnt/project/fm-transfer-20261016/evaluations/A_DDP_TIME_RL_update015000_DEV --reference /mnt/project/fm-transfer-20261016/evaluations/A_DDP_TIME_RL_update010000_DEV --output /mnt/project/fm-transfer-20261016/analysis/A_DDP_TIME_RL_update015000_DEV_vs10k.json
python -m scripts.analysis.fm_transfer.analyze_safety_transitions --evaluation /mnt/project/fm-transfer-20261016/evaluations/A_DDP_TIME_RL_update015000_DEV --reference /mnt/project/fm-transfer-20261016/evaluations/A_DDP_TIME_SFT_update020000_DEV --output /mnt/project/fm-transfer-20261016/analysis/A_DDP_TIME_RL_update015000_DEV_safety_transitions.json
python -m scripts.analysis.fm_transfer.analyze_grid_gap --evaluation /mnt/project/fm-transfer-20261016/evaluations/A_DDP_TIME_RL_update015000_DEV --output /mnt/project/fm-transfer-20261016/analysis/A_RL15k_DEV_native_minus_shift5.json
```

训练前缀的实际脚本副本：[reproduce_A_RL15k_training_prefix.py](reproduce_A_RL15k_training_prefix.py)。
