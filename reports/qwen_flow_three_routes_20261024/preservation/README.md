# 路线三：固定成功存档与成本约束

截至本记录，已实现P0—P5并通过实际原生模型资格；完整初筛和部署效果仍等待共同候选库、TRAIN校准及主agent的统一GPU排程。**当前没有本路线的正式PDMS增益或能力保留结论。** 单条DEV资格轨迹仅证明更新后的ODE能执行，不作为研究效果。

## 实现与共同合同

起点是已完整评测的成熟CANONICAL50k合并模型，Navtest90.513164；永久reference是该起点，不是旧IL53k。原生Qwen全token、源30层ActionDiT、源8×3表示及shift5保持。

|臂|策略更新|额外监督|
|---|---|---|
|P0|共同源GRPO|无|
|P1|同P0，仅有限TRAIN漂移校准后的较低LR|无|
|P2|同P0|同初始池的单最佳轨迹|
|P3|同P0|同初始池的最多四条合格、不同轨迹|
|P4|PDMS与独立成本优势、投影乘子|无|
|P5|同P4|与P3完全相同的固定多轨存档|

全部采用源PPO clip0.005、永久转移均值MSE anchor50、四次真实inner更新；新研究共同使用fresh全局原始reward std的0.01 floor且关闭优势±5截断。这是相对历史配方的共同研究设置，不是旧源码修复。P0—P3 mask复用公共`group_advantages`；P4/P5按最终组合优势判断整组是否有信号，避免PDMS等分屏蔽真实成本差异。

固定存档只取共同基策略的TRAIN候选，要求真实分数≥0.9及官方NC/DAC/TTC全部完全通过。最多四条，保留最高分和合格固定参考，再选近最优且有区别的请求轨迹；近优容差0.01原始score，几何去重ADE0.15米、heading RMS0.03弧度。P2/P3/P5使用相同可用场景人口和相同监督scene曝光，scene、目标选择、FM分别有独立RNG，防止单轨省略抽样导致后续scene顺序不同。

监督使用原始请求轨迹的源3维编码。源`sample_training_t/add_noise/training_target/training_weight`及模型FM前向保留，独立FM噪声不耦合候选root；非负场景内目标抽样，不平均成新轨迹，也不伪造SDE old logprob。旧DDP `noise_s=.999`和repeat8的提取损失没有搬来。

本轮成本明确为`1-NC,1-DAC,1-TTC`，NC=0.5对应成本0.5，与之前SCADV3的事件成本版本不同。预算由同核未更新base、固定128个TRAIN场景×G16的2048条真实SDE候选估计，epsilon0；不从ODE库冒充同核预算。三个投影SGD乘子初始1、lr0.01，每份未筛选fresh collection更新一次，四个inner保留采样时快照。成本先按scene中心化，再除同一个与λ无关的共同reward尺度；组合后不再whitening，不宣称精确CMDP或逐场景安全保证。

## 已执行证据

- [CPU v5](qualification_CPU_v5.json)：17项通过。包括固定库/去重/空库不造成功样本、单多轨相同scene曝光、成本与λ符号、冻结预算、P1有限TRAIN漂移匹配、零梯度cosine未定义；还用真实tiny原生ActionDiT/LoRA分别执行六臂四次更新，P5 inner1恢复后参数、Adam、dual及RNG完全一致。
- [实际原生模型资格](actual_native_profile_v1.json)：主agent派发vla-zt GPU3，B2/G4，148.79秒，峰值约27.21GiB。四次逻辑更新加三次恢复重演，480个LoRA张量都有非零Adam梯度一阶矩；Qwen/provider/base/reference参数与buffers hash不变，恢复后的下一条chain、old probability和reward完全一致。
- 实际更新后的独立DEV单候选ODE可运行，未调用archive或reference路径；该一条轨迹不构成部署收益、泛化或安全证据。
- CPU v2/v3曾因新RNG构造少传`seed_for`第三个参数而在更新前失败；原始结果保留，修复后才发布资格。[失败记录](qualification_issues.json)。没有正式run受影响。
- 真实资格后的归档准入补丁：同ID且同base/scorer/sampler/请求/编码/分项/奖励的重复候选合并频次和来源，重复次数不会提高该轨迹的回放概率；冲突仍拒绝。补丁未改actor/sampler/FM/dual/update。[18项CPU v6通过](qualification_CPU_v6_archive_dedup.json)，并用原GPU资格的8条真实记录与旧实现比较，入库与128次抽样目标逐值相同，[等价收据](archive_admission_equivalence_v1.json)。

资格的更新阶段31.05秒/7个物理inner，B2/G4；不能把它当B16/G16的每步时间或宣称免费加速。完整初筛仍需profile与实际账本确认成本，资格权重不能进入正式初始化。

## 可运行入口

实际执行过的CPU命令：

```bash
CUDA_VISIBLE_DEVICES='' /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.qwen_flow_exploration.preservation_validate \
  --output /mnt/project/qwen-flow-three-routes-20261024/preservation/qualification_CPU_v5.json
```

主agent已执行真实资格：

```bash
CUDA_VISIBLE_DEVICES=3 /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.qwen_flow_exploration.preservation_profile \
  --protocol /mnt/project/qwen-flow-three-routes-20261024/common_protocol.json \
  --output /mnt/project/qwen-flow-three-routes-20261024/preservation/profile_real_v1 --B 2 --G 4 --seed 42
```

后续真实命令已实现，**由主agent按独占study-GPU租约执行，不在现有profile目录重跑**：

```bash
MODEL_PY=/mnt/project/flow-safe-improvement-20261013/venv/bin/python
$MODEL_PY -m scripts.analysis.qwen_flow_exploration.preservation_build_archive \
  --candidates <共同TRAIN候选JSONL> --population <原log隔离population.json> --output <固定archive.json>
$MODEL_PY -m scripts.analysis.qwen_flow_exploration.preservation_calibrate \
  --protocol <common_protocol.json> --archive <固定archive.json> --output <新TRAIN校准目录> --planned-rounds 64
$MODEL_PY -m scripts.analysis.qwen_flow_exploration.preservation_train \
  --protocol <common_protocol.json> --output <独立P5正式run> --arm P5 --seed 42 \
  --registration <BUDGET_LOCK.json> --calibration <TRAIN校准目录/calibration.json>
```

P0无需calibration，其他臂使用同一个锁定校准身份。P1最多两个LR影子试验；P5四步与P1两次四步共12个校准optimizer更新单列，权重丢弃。若漂移不匹配，报告`NOT_MATCHED`，不能排除“仅学得更慢”。

训练保存完整pending batch、old链/概率/优势、next_inner、LoRA/Adam/scheduler、dual及三个replay RNG；复用公共合法phase原子checkpoint。`--resume <同run合法checkpoint>`恢复剩余inner。collection失败不推进latest，失败算力和回滚记录保留；SIGUSR1在合法inner边界保存退出。

## 当前研究判断

**证据不足**，仅针对目前尚未完成的正式实验，不是对存档或约束方法的否定。必须等待P0/P3/P4/P5因子对照、P2单轨和P1幅度匹配控制，再以自然评估人口上的净收益、G+/G−及固定高分场景退化判断。

最有信息量的下一项是完成同预算P0/P3/P4/P5，而不是增加archive容量或新的dual算法；若P0本身几乎不退化，还应明确这轮保留问题的统计判别力不足。
