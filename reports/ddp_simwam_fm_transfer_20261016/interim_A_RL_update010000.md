# A组RL10k：原生ODE收益继续增加，TTC退化有所收窄

A_DDP_TIME完成固定10,000 optimizer updates观察点。自己的参考仍为20k SFT终点；完整DEV为1696场景/16 logs，固定TRAIN诊断为512场景/343 logs。均为原始观测、seed2101、单候选10步ODE及canonical NAVSIM v1.1。原生均匀网格与shift5分开报告，不做best-of-two。

**DEV原生PDMS从自身SFT的93.720093，经RL5k的94.332347，提高至RL10k的94.627902。** 相对SFT和5k的配对区间都高于零。shift5为94.381735，与5k基本持平。原生新增TTC违规相对SFT从28降至20，但安全保持仍未达成。A继续原20k RL终点，B继续原100k SFT；这不是终点模型选择或Navtest结果。

## 部署成绩与匹配增量

所有分数及差值以下用0–100分/百分点表示；置信区间按log聚类重采样，保持场景权重，仅覆盖固定训练seed42、推理seed2101下的数据不确定性。

| 人口/网格 | 自身SFT | RL5k | RL10k | 10k−SFT，95%区间 | 10k−5k，95%区间 |
|---|---:|---:|---:|---|---|
| DEV 原生 | 93.720093 | 94.332347 | **94.627902** | **+0.907809** [+0.606600,+1.218724] | **+0.295554** [+0.126913,+0.453765] |
| DEV shift5 | 93.460310 | 94.384140 | **94.381735** | **+0.921425** [+0.509944,+1.435392] | −0.002405 [−0.266182,+0.396849] |
| TRAIN诊断 原生 | 81.835774 | 84.254122 | 86.575159 | +4.739385 [+2.646521,+6.905020] | +2.321037 [+0.962961,+3.814698] |
| TRAIN诊断 shift5 | 80.607056 | 83.998308 | 86.701362 | +6.094306 [+3.428868,+8.810224] | +2.703054 [+1.115683,+4.499709] |

TRAIN诊断包含预选困难场景，不能当完整TRAIN无偏均分。其RL曝光覆盖见[执行记录](A_RL10k_execution.json)；完整DEV没有用于参数更新。原SFT可能见过这些人口，不把“RL未更新过”解释成基模训练完全未见。

同一10k权重的DEV原生−shift5为+0.246167个百分点，区间[+0.043200,+0.358176]。但两种网格相对各自SFT的RL增量接近（+0.907809/+0.921425）：不能把网格起点差异全部归给RL，也不能据此临时选择较高读出。原生仍是登记的主要部署协议。

## 安全：相对5k有恢复，相对SFT仍有新增失败

| DEV分项，0–100 | SFT原生 | RL5k原生 | RL10k原生 | SFT shift5 | RL5k shift5 | RL10k shift5 |
|---|---:|---:|---:|---:|---:|---:|
| NC | 100.000000 | 100.000000 | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| DAC | 99.292453 | 99.705189 | 99.764151 | 99.056604 | 99.528302 | 99.469340 |
| TTC | 99.764151 | 98.290094 | **98.761792** | 99.587264 | 99.233491 | **99.233491** |
| EP | 86.142845 | 88.567445 | 88.746115 | 86.050026 | 88.019577 | 88.096352 |
| Comfort | 99.882075 | 99.882075 | 99.823113 | 99.823113 | 99.823113 | 99.823113 |
| DDC | 98.879717 | 98.879717 | 98.850236 | 98.879717 | 98.879717 | 98.850236 |

下表均是**10k相对自己的SFT**，违规指指标未完全通过，NC=0.5也算违规；它不等于每一项都发生碰撞。

| 人口/网格 | NC新增/恢复 | DAC新增/恢复 | TTC新增/恢复 | 非零→零/零→非零 | 固定SFT-safe中新联合违规 |
|---|---|---|---|---|---|
| DEV 原生 | 0/0 | 2/10 | **20/3** | 2/10 | **22/1681** |
| DEV shift5 | 0/0 | 5/12 | **8/2** | 5/12 | **13/1674** |
| TRAIN诊断 原生 | 4/0 | 6/28 | 13/5 | 6/28 | 17/440 |
| TRAIN诊断 shift5 | 1/0 | 9/37 | 7/4 | 9/37 | 14/432 |

若改为**10k相对5k**，DEV原生TTC新增1个、恢复9个，DAC新增0个、恢复1个；联合安全场景1662→1671，零分5→4。相对SFT的联合安全场景仍是1681→1671，所以只能说退化有所收窄。原生1%分位仍为58.333333，低于自身SFT的80.302305；5%分位则升至85.289659。

shift5的TTC均值虽未变，相对5k却新增6个、恢复6个，失败场景成员发生变化；DAC新增4个、恢复3个，零分8→9。联合安全场景1675→1674，等于自身SFT的总数，但仍有13个原安全场景退化，被另一批恢复抵消。不能用总数不变宣称逐场景安全保持。

DEV原生G+=25.299776、G−=9.903334；shift5为23.709392、8.082021，单位为原始0–1分差的场景求和。相比5k，原生G−从14.078394下降；shift5从5.176395上升。完整正负变化与±0.005实用阈值计数均保留在配对JSON中，不把任意小负差解释成事故。

## 保存轨迹揭示的残余问题

复用已验收的安全转移分析器读取同一批真实请求与模拟轨迹，没有新推理或评分。DEV原生20个新增TTC场景来自4个logs，19个首次官方违规发生于物理3.6–4.0秒；shift5的8个来自5个logs。它们的NC均完全通过，不能写成20/8次碰撞，也不能将物理违规时间对应到Flow步骤。

在各自首次官方TTC违规的**同一物理时刻**，原生20/20、shift5 8/8均比SFT前进更多，之前0.1秒的平均段速度也更高：

| 新增TTC子集，配对均值 | 原生 | shift5 |
|---|---:|---:|
| 该时刻前进位移差 | +0.7843 m | +0.5855 m |
| 前0.1秒段速度差 | +0.5276 m/s | +0.4543 m/s |
| 请求点离散jerk RMS，SFT→RL | 0.4758→2.6151 m/s³ | 1.2870→2.3254 m/s³ |

请求点jerk不是官方Comfort，也不是模拟车辆的瞬时jerk。所有初始模拟状态、评分cache与协议均已配对核验。

原SFT-safe中新联合违规贡献了全部G−的97.74%（原生）/99.81%（shift5）；新增TTC贡献78.59%/40.56%。同时，420/400个场景在两策略均安全且PDMS至少提高0.005，它们的模拟终点前进量平均增加0.8308/0.6131米。因此有益进度与安全退化都存在，不能一律要求更多前进或一律减速。

对同一已评分ODE反馈做源10:5:2读出，全部DEV均值分别提高1.406474/1.252278个百分点，但新增TTC子集的源reward全部低于SFT。没有“source提高而canonical下降”的配对场景。这排除了这些具体失败轨迹被源reward直接评为更好的解释，仍未排除源权重对随机G8组内梯度权衡的影响。

以上按结果定义的子集仅作描述，不构成因果识别。相对5k，原生TTC恢复增加，而shift5失败成员更替，表明安全变化并非简单随更新次数单调恶化；也没有证明某个网格或anchor单独负责。

## 真实更新与保存批次梯度

八rank日志及原生10k checkpoint匹配：10,000次全局optimizer更新、2,500份fresh全局rollout、20,000次场景曝光、160,000条新候选，覆盖20,000个独立场景/1062 logs。四次inner复用不计为四倍新数据。

- 19,905/20,000组有PG，比例99.525%；95个等分组中15个全满分、23个全零分、57个其他等分。全局anchor-only更新为0。
- progress-only为13,504组（67.52%），安全分项有差异为4,918组（24.59%）。解决等分浪费不等于已获得足够安全探索。
- 正优势候选86,833个完全安全、441个仍违规，其中366个所在组有完全安全替代。它们是候选计数，不是梯度权重或具体DEV事故归因。
- 第一inner的最大ratio误差和clip比例均为0；第2/3/4个inner平均clip比例为0.289375%/2.465625%/6.190208%。保留源clamped dimension-mean代理，不称整条轨迹精确概率比。

对第2500份保存批次、inner4后的10k权重，8场景/64候选重算得到：

| Flow步骤 | PG梯度norm | beta50 anchor梯度norm | cosine | 当前相对保存旧概率的PPO clip |
|---|---:|---:|---:|---:|
| 7 | 0.017618 | 0.000713 | −0.145417 | 0% |
| 8 | 0.026910 | 0.002295 | +0.026995 | 0% |
| 9 | 0.045587 | 0.021052 | −0.252637 | 17.1875% |

最后一步anchor约为PG norm的46.2%，且部分反向；前两步较弱。5k对应批次约63.7%，但两个milestone的场景/链不同，不能将此差值当成控制实验。分解梯度与实际源总loss梯度的最大绝对差3.49e−10；29项实际训练依赖hash匹配，全部adapter保持不变。这批36个正优势候选均安全，不代表全训练正优势都安全。

## 成本、资源与身份

10k训练循环wall为8649.245秒（144.154分钟），模型/DDP准备另计31.985秒。吞吐1.1562 updates/s。与已保存5k边界相减，后5k实际用时约69.036分钟，未用累计更新数除以局部进程时间。

本次在10k原生恢复点发布时同步保存了八rank invocation账本：rank0累计checkpoint保存59.153秒；rank0更新计算5117.500秒、rollout1300.128秒、旧概率406.253秒、评分等待312.543秒。scorer elapsed151.905秒不是CPU核秒。训练循环wall包含未单独拆出的调度、I/O等时间，累计rank wall不能当独占GPU计算时间。精确成本快照只使用已核定的保存/准备字段，其余逻辑预算取封存checkpoint与日志前缀。

| 本次评价 | 两网格逻辑候选 | 实际推理含入口检查 | job wall秒 | 累计worker wall秒 |
|---|---:|---:|---:|---:|
| DEV | 3392 | 3400 | 362.519 | 2867.578 |
| TRAIN诊断 | 1024 | 1032 | 188.294 | 1496.312 |

两项在tsinghua-zt/rl-zt2并行，不能相加job wall充当总耗时。16个worker均已核实退出，16项原入口检查通过，完整分母及payload hash核验通过。派发前重新检查授权和资源，只释放rl-zt2 GPU5/6/7上3个已核实PID/启动时间/argv/UID/ready/NVML映射的占卡worker，保留agent及其他真实/未知任务。

保存批次分析另用38.803秒，峰值显存7,387,997,696 bytes，0新候选、0评分、0参数更新。安全转移分析main耗时DEV18.877秒、TRAIN6.502秒，不含Python导入；配对均分、安全转移、G+/G−与原完整评价分析在四个人口/网格组合均一致。

正式训练源码仍为`225b9af660cb0210208e2c46fc4216d0f151275b`，评价为`45d2318d2dec265583860a26f891d8b79bb6068a`，保存批次诊断为`fd440275edc3fdf40b2aaa46c38da1d038068902`，本次只读分析为`487bf3e13d784b90fbf4c2e5789d38d47c88a9d8`。10k原生checkpoint为`update010000_inner0`，world8、phase READY；不表示20k训练结束。

合并后原入口权重SHA256：`289aa5a4b1442fd46f84cc35626c73a1fcd5710a6d40305610ab8a047c14e129`。相对自身SFT实际改变96个attention投影，冻结前缀保持；恢复继续使用原生LoRA/Adam/RNG checkpoint。

## 本阶段结论与下一观察

A组已在两个固定观察点把RL收益传到真实单候选ODE。原生10k较5k有进一步净收益，TTC部分恢复；这不支持“策略没有更新”或“只候选oracle变好”的解释。shift5的DEV均分在5k→10k基本持平，而TRAIN继续提高，不能只展示训练曲线。

安全保持仍是主要未解决问题：原安全场景的新违规几乎解释了全部负向退化，原生低尾尚未恢复。当前证据不足以单独归因于FM时间采样、额外SFT曝光、困难人口、源reward、anchor或动作结构；这些因素不是本轮已隔离的单因素对照。

继续固定15k/20k观察，B完成自身100k SFT后按既定筛选和八卡验收接同一RL。保持两组正式配方与预算。最终四模型Navtest仍等待两组终点和完整里程碑；不将当前DEV 94.6当作最终Navtest超过92。

## 已执行命令与机器证据

[DEV对自身SFT](A_RL_update010000_DEV_vs_own_SFT.json)、[DEV对5k](A_RL_update010000_DEV_vs5k.json)、[TRAIN对自身SFT](A_RL_update010000_TRAIN_DIAGNOSTIC_vs_own_SFT.json)、[TRAIN对5k](A_RL_update010000_TRAIN_DIAGNOSTIC_vs5k.json)、[DEV安全转移](A_RL_update010000_DEV_safety_transitions.json)、[TRAIN安全转移](A_RL_update010000_TRAIN_DIAGNOSTIC_safety_transitions.json)、[网格差](A_RL10k_DEV_native_minus_shift5.json)、[训练前缀](A_RL_update010000_training_prefix.json)、[梯度](A_RL_update010000_saved_batch.json)、[执行/成本](A_RL10k_execution.json)。

以下命令均实际执行。输出已存在时入口拒绝覆盖；重复分析应另给输出身份，不重新生成策略轨迹。

```bash
python /mnt/project/fm-transfer-20261016/resources/analyze_A_RL_10k_prefix.py
python -m scripts.analysis.fm_transfer.diagnose_saved_batch --run /mnt/project/fm-transfer-20261016/runs/A_DDP_TIME_R_SOURCE_seed42_v1 --checkpoint /mnt/project/fm-transfer-20261016/runs/A_DDP_TIME_R_SOURCE_seed42_v1/checkpoints/update010000_inner0 --collection 2500 --hard-manifest /mnt/project/fm-transfer-20261016/hard_manifests/A_DDP_TIME_SFT_update020000.json --sft-screen /mnt/project/fm-transfer-20261016/screens/A_DDP_TIME_SFT_update020000 --output /mnt/project/fm-transfer-20261016/analysis/A_DDP_TIME_RL_update010000_saved_batch.json
python -m scripts.analysis.fm_transfer.analyze_evaluation --evaluation /mnt/project/fm-transfer-20261016/evaluations/A_DDP_TIME_RL_update010000_DEV --reference /mnt/project/fm-transfer-20261016/evaluations/A_DDP_TIME_SFT_update020000_DEV --output /mnt/project/fm-transfer-20261016/analysis/A_DDP_TIME_RL_update010000_DEV_paired_own_SFT.json
python -m scripts.analysis.fm_transfer.analyze_evaluation --evaluation /mnt/project/fm-transfer-20261016/evaluations/A_DDP_TIME_RL_update010000_DEV --reference /mnt/project/fm-transfer-20261016/evaluations/A_DDP_TIME_RL_update005000_DEV --output /mnt/project/fm-transfer-20261016/analysis/A_DDP_TIME_RL_update010000_DEV_vs5k.json
python -m scripts.analysis.fm_transfer.analyze_safety_transitions --evaluation /mnt/project/fm-transfer-20261016/evaluations/A_DDP_TIME_RL_update010000_DEV --reference /mnt/project/fm-transfer-20261016/evaluations/A_DDP_TIME_SFT_update020000_DEV --output /mnt/project/fm-transfer-20261016/analysis/A_DDP_TIME_RL_update010000_DEV_safety_transitions.json
python -m scripts.analysis.fm_transfer.analyze_grid_gap --evaluation /mnt/project/fm-transfer-20261016/evaluations/A_DDP_TIME_RL_update010000_DEV --output /mnt/project/fm-transfer-20261016/analysis/A_RL10k_DEV_native_minus_shift5.json
```

TRAIN的三项配对/安全分析采用对应`TRAIN_DIAGNOSTIC`路径实际执行，结果链接见上。前缀核算脚本副本为[reproduce_A_RL10k_training_prefix.py](reproduce_A_RL10k_training_prefix.py)。所有原始轨迹、权重、缓存与完整私有日志保留在运行资产目录，未上传仓库。
