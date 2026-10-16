# A组首个5k RL：单候选DEV有净收益，TTC退化仍未解决

A_DDP_TIME完成首个登记的5,000 optimizer updates观察点。完整DEV为1696场景/16 logs，固定TRAIN诊断为512场景/343 logs。两种读出均从原始当前观测推理、seed2101、单候选、canonical NAVSIM v1.1；没有搜索、GT筛选或best-of-N。

参考固定为A自己的20k SFT终点。以下是`A_RL5k − A_SFT20k`，不是相对历史F-SFT，也不是随机训练候选均分。A继续既定20k RL，B继续既定100k SFT；本报告不是终点模型选择或Navtest确认。

## 真实ODE增益

| 人口/网格 | 自身SFT | RL5k | 差值，百分点 | 按log聚类95%区间，百分点 |
|---|---:|---:|---:|---|
| DEV 原生均匀10步 | 93.720093 | **94.332347** | **+0.612255** | **[+0.316837,+0.962323]** |
| DEV shift5 10步 | 93.460310 | **94.384140** | **+0.923830** | **[+0.694859,+1.262999]** |
| TRAIN诊断 原生均匀10步 | 81.835774 | 84.254122 | +2.418348 | [+0.270300,+4.649091] |
| TRAIN诊断 shift5 10步 | 80.607056 | 83.998308 | +3.391251 | [+0.954835,+5.740140] |

四个固定人口/网格的配对区间都高于零。这支持新SFT起点上的源Flow-GRPO已把部分训练收益传到单候选ODE，而且DEV均分也提高；不能再解释成“只有训练reward或候选最大值提高”。区间仅覆盖当前固定seed下的log重采样，不包含训练seed或推理seed不确定性。

TRAIN诊断有预选困难场景，不是完整TRAIN的无偏估计。其512个场景中，224个属于A的新固定困难集，只有107个进入前5k RL的实际更新，405个尚未进入该RL前缀；它们仍属于RL_TRAIN且可能被SFT见过。这些覆盖数不能单独证明泛化或排除记忆。

A自身SFT原生DEV相对旧F-SFT仅+0.100856个百分点、区间跨零，已经单列在[SFT终点报告](final_A_SFT_update020000.md)。本次RL增量与该SFT变化分开核算。当前结果仍不能把增益独立归因于“时间加权FM”：额外监督曝光、新SFT权重和新困难人口同时变化，旧C/N及旧R_SOURCE不是只改时间分布的匹配控制。

## 均分提高并不代表安全保持

| DEV分项，0–100 | SFT原生 | RL原生 | SFT shift5 | RL shift5 |
|---|---:|---:|---:|---:|
| NC | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| DAC | 99.292453 | 99.705189 | 99.056604 | 99.528302 |
| TTC | 99.764151 | **98.290094** | 99.587264 | **99.233491** |
| EP | 86.142845 | 88.567445 | 86.050026 | 88.019577 |
| Comfort | 99.882075 | 99.882075 | 99.823113 | 99.823113 |
| DDC | 98.879717 | 98.879717 | 98.879717 | 98.879717 |

| 人口/网格 | NC新增/恢复 | DAC新增/恢复 | TTC新增/恢复 | 非零→零/零→非零 | 固定SFT联合安全人口中新违规 |
|---|---|---|---|---|---|
| DEV 原生 | 0/0 | 3/10 | **28/3** | 3/10 | **31/1681** |
| DEV shift5 | 0/0 | 2/10 | **8/2** | 2/10 | **10/1674** |
| TRAIN诊断 原生 | 6/0 | 7/22 | 20/4 | 10/22 | 25/440 |
| TRAIN诊断 shift5 | 1/0 | 9/25 | 11/4 | 10/25 | 18/432 |

这里的违规指官方指标未完全通过，NC=0.5也不算完全安全；它不等于每一项都发生了碰撞。DEV原生零分12→5，shift5零分16→8，但原生NC/DAC/TTC联合安全场景数1681→1662；shift5为1674→1675。仅报告零分净减少，会掩盖新增TTC失败和原安全场景退化。

DEV原生G+=24.462237、G−=14.078394；shift5为20.844553、5.176395。这是0–1逐场景分差的总和，除1696再乘100才得到均分百分点变化。原生任意改善/退化为910/197场景，达到±0.005原始score幅度的改善/退化为470/31；shift5分别949/143和426/10。任意负差不能解释成事故。

原生1%低尾从80.302305降到58.333333，虽然5%分位从83.806183升到84.922731；shift5的1%分位从49.546998升到58.333333。低尾和新增违规必须与均值一起看。

## 网格和行为变化

同一RL5k权重的DEV原生减shift5为−0.051792个百分点，95%区间[−0.431807,+0.272513]，没有充分证据判定两种网格的总体PDMS不同。两个网格的TTC转移差异值得继续观察，但不能仅据这一次5k结果断定采样迁移是唯一原因或改用较高网格报告成功。原生ODE仍是主要部署协议。

DEV原生请求终点前向位移17.4869→17.7130 m，模拟终点前向位移17.5835→17.9716 m；实际原始进度18.5741→18.9682 m。shift5对应请求前向17.4622→17.6444 m、模拟前向17.6178→17.9054 m、原始进度18.6179→18.9153 m。这与EP增加一致，支持“更多向前进度”是当前收益的一部分。

请求轨迹平均分段速度：原生4.5258→4.5894 m/s，shift5 4.5231→4.5742 m/s。请求点离散差分jerk RMS：原生0.5668→1.6522 m/s³，shift5 1.2777→2.0118 m/s³。它们是请求轨迹描述，不能替代模拟轨迹上的官方Comfort；本次官方Comfort均值未变。到自身路径90%的时间也会随路径长度变化，不能直接解释成到达同一目标更快。

## 真实更新、有效组和源代理

八rank日志与5k原生checkpoint已匹配：5,000次全局optimizer更新、1,250次fresh collection、10,000次场景曝光、80,000条新候选。覆盖10,000个独立场景/989 logs，四次inner复用不计作四倍新数据。

- 9,951/10,000组有非零PG，比例99.51%。49个等分组中，4个全满分、18个全零分、27个其他等分。
- 所有1,250个全局batch均至少有一个rank有PG；全局anchor-only更新为0。不能把rank0的局部anchor-only计数当成全局没有RL更新。
- 6,810组为progress-only，2,416组有安全分项差异。非零优势多并不自动意味着有用的安全探索充分。
- 日志中正优势候选43,232个完全安全，228个仍有违规，其中185个所在组存在完全安全替代。这些是候选计数，不是梯度贡献，也不证明它们直接造成DEV的某次TTC失败。
- 训练随机策略canonical均分82.491789，源10:5:2训练reward为78.374548；其人口是困难TRAIN。有限G8安全最优均值90.589031，仅在9965个存在安全候选的组上定义。二者均不进入单候选部署主表。

第一inner的old/current ratio误差为0，clip fraction为0；第2/3/4个inner平均clip fraction分别为0.259167%、2.337917%、5.996667%。这支持数值起点一致及后续发生真实更新，不支持“PPO完全没有更新”的解释。主实验仍使用源逐维relative clamp之后的dimension-mean代理，未改成联合轨迹概率。

## 保存的真实批次梯度

复用原诊断内核，新增显式runtime连接自己的SFT与筛选记录，避免旧脚本硬编码历史F-SFT。15项CPU针对性测试、CLI smoke、真实SFT清单核查后，实际运行了5k最后一批8场景/64候选的只读诊断。29个实际导入训练依赖的hash与运行身份一致；8个SFT场景评分及payload核验通过。分解梯度与实际源总loss梯度最大绝对差2.73e−10，全部LoRA参数保持不变。

| Flow生成步骤 | PG梯度norm | beta50 anchor梯度norm | cosine | 四次更新后的PPO clip比例 |
|---|---:|---:|---:|---:|
| 7 | 0.022692 | 0.000956 | −0.448805 | 0% |
| 8 | 0.027720 | 0.002854 | +0.259045 | 0% |
| 9 | 0.037835 | 0.024090 | −0.384386 | 21.875% |

这是生成步骤，不是驾驶物理时间。该批最后一步的anchor约为PG norm的63.7%，且部分反向；前两步较弱。因此不能只比loss标量或beta数值认定anchor过强，也不能把这个批次推广到全部训练。该批36个正优势候选都完全安全，没有证明全训练的正优势都安全。

分析时点是inner4之后，对比保存的旧behavior；21.875%的末步clipping可以是真实更新的结果，不能与第一inner的数值误差混同。原Gaussian与源clamped代理分开记录；本批尾部clamp命中率约0.016276%，创新clamp命中率0。

## 执行成本和资源

5k训练循环wall为4,507.057秒（75.118分钟），约1.1094 updates/s；模型/DDP等准备另计31.985秒。rank0累计更新计算2,563.405秒、rollout650.379秒、旧概率重算203.252秒、评分等待158.068秒。这些计时不覆盖所有调度/I/O区间，跨rank时间也不能当作独占GPU计算时间。检查点保存期间的总耗时在训练循环wall内；5k边界没有另外归档完整save-only累计量，不补造该值。

| 本次评价 | 两网格逻辑候选 | 实际推理含入口检查 | job wall秒 | 累计worker wall秒 |
|---|---:|---:|---:|---:|
| DEV | 3392 | 3400 | 377.418 | 2997.367 |
| TRAIN诊断 | 1024 | 1032 | 174.945 | 1377.889 |

共16个worker在tsinghua-zt和rl-zt2并行，已全部正常退出；16项原入口一致性检查均完成，所有结果通过原analyzer的完整分母和payload检查。入口检查、两种网格和实际重试分别记账。保存批次分析另耗39.738秒、峰值显存7,387,997,696 bytes，新增候选/评分/optimizer更新均为0。

派发时重新检查授权和实际资源：只手动释放了本机GPU2及rl-zt2八张卡上已核实身份的9个占卡worker；tsinghua的占卡worker在手动处理前已自动退出，因此该主机未发送停止信号。其他真实训练和占卡监控agent均保持。A/B正式训练源码、预算和优化器状态未改变。

## 当前结论与继续条件

1. **更新可以传到部署。** 这条路线在固定DEV上有正的单候选RL净增益，两个网格的配对区间均高于零。仍需10k/15k/20k观察及锁定后的Navtest确认。
2. **安全保持尚未达成。** 收益伴随EP和DAC改善、TTC恶化，原生ODE还出现联合安全人口净减少及1%低尾下降。不能称稳定全面提升，也不能说所有安全分项都下降。
3. **不能把原因全部归于Flow源头错误或某一个模块。** 新SFT和困难人口、源配方完整性及采样网格都有关；本次局部梯度证据尚未隔离因果。现在不改reward、LR、sigma、anchor或clip，不增加新算法矩阵。
4. **继续原登记终点。** A到20k RL，B到100k SFT后建立自己的基线、筛选和终点验收，再接相同R_SOURCE。B尚无正式RL结果。最终Navtest仍未运行；DEV 94分不能替代最终RL/Navtest超过92的目标。

机器可读结果：[DEV配对](A_RL_update005000_DEV_vs_own_SFT.json)、[TRAIN配对](A_RL_update005000_TRAIN_DIAGNOSTIC_vs_own_SFT.json)、[训练前缀](A_RL_update005000_training_prefix.json)、[梯度](A_RL_update005000_saved_batch.json)、[网格差](A_RL5k_DEV_native_minus_shift5.json)、[执行身份与成本](A_RL5k_execution.json)。

## 已执行命令与身份

正式训练源码`225b9af660cb0210208e2c46fc4216d0f151275b`；评价源码`45d2318d2dec265583860a26f891d8b79bb6068a`；新只读梯度入口`fd440275edc3fdf40b2aaa46c38da1d038068902`。训练seed42，评价seed2101。

自身SFT模型SHA256：`ac52e9346080ce0e6bf27441935ac9aa6acbff239330b316bee36b560486579a`。
RL5k原入口导出SHA256：`496c7c37d5558b3e0d97628e79f523e7fc1d0a6cf29c107397533323abf4ceda`。
合并后实际改变96个attention投影，其余冻结部分经导出核查保持；恢复使用原生checkpoint，不使用merged export重新初始化Adam。

```bash
python -m scripts.analysis.fm_transfer.analyze_evaluation --evaluation /mnt/project/fm-transfer-20261016/evaluations/A_DDP_TIME_RL_update005000_DEV --reference /mnt/project/fm-transfer-20261016/evaluations/A_DDP_TIME_SFT_update020000_DEV --output /mnt/project/fm-transfer-20261016/analysis/A_DDP_TIME_RL_update005000_DEV_paired_own_SFT.json
python -m scripts.analysis.fm_transfer.analyze_evaluation --evaluation /mnt/project/fm-transfer-20261016/evaluations/A_DDP_TIME_RL_update005000_TRAIN_DIAGNOSTIC --reference /mnt/project/fm-transfer-20261016/evaluations/A_DDP_TIME_SFT_update020000_TRAIN_DIAGNOSTIC --output /mnt/project/fm-transfer-20261016/analysis/A_DDP_TIME_RL_update005000_TRAIN_DIAGNOSTIC_paired_own_SFT.json
python -m scripts.analysis.fm_transfer.diagnose_saved_batch --run /mnt/project/fm-transfer-20261016/runs/A_DDP_TIME_R_SOURCE_seed42_v1 --checkpoint /mnt/project/fm-transfer-20261016/runs/A_DDP_TIME_R_SOURCE_seed42_v1/checkpoints/update005000_inner0 --collection 1250 --hard-manifest /mnt/project/fm-transfer-20261016/hard_manifests/A_DDP_TIME_SFT_update020000.json --sft-screen /mnt/project/fm-transfer-20261016/screens/A_DDP_TIME_SFT_update020000 --output /mnt/project/fm-transfer-20261016/analysis/A_DDP_TIME_RL_update005000_saved_batch.json
python -m scripts.analysis.fm_transfer.analyze_grid_gap --evaluation /mnt/project/fm-transfer-20261016/evaluations/A_DDP_TIME_RL_update005000_DEV --output /mnt/project/fm-transfer-20261016/analysis/A_RL5k_DEV_native_minus_shift5.json
python /mnt/project/fm-transfer-20261016/resources/analyze_A_RL_5k_prefix.py
```

前缀核算的实际脚本副本见[reproduce_A_RL5k_training_prefix.py](reproduce_A_RL5k_training_prefix.py)。已存在的输出禁止覆盖；重新分析需使用新的输出身份。正式训练继续由原owner运行，本次没有执行恢复或重复派发训练。
