# BASE：同root的SDE/ODE有限候选机会

**本次K16内，SDE均分低22.8201点，安全候选最高分的固定人口代理均值却高1.5128点。** 当前证据指向候选概率质量差异，尚未发现有限安全候选上界更弱；候选中部也退化，不能仅用极少数尾部离群点解释。

对象：成熟共同BASE，固定TRAIN128场景/117个log，每scene16条候选，两种采样均为十步shift5，原生未合并LoRA数值路径；ODE和SDE的x0逐值相同。安全条件仅指官方NC=1、DAC=1、TTC=1。这里全是有限候选诊断，**不是单轨部署成绩、Navtest成绩或学会这些轨迹的证据**。

|量|SDE16|ODE16|
|---|---:|---:|
|全部候选均值PDMS|70.8449|93.6650|
|feasible@16|127/128（99.22%）|127/128（99.22%）|
|安全候选比例|76.27%|98.34%|
|零分候选比例|16.46%|0.78%|
|候选中位数PDMS|83.3333|96.9661|
|overall max的scene均值|96.1592|94.6464|
|safe max的零代理scene均值（分母128）|96.1592|94.6464|
|safe max仅可用场景条件均值（分母127）|96.9163|95.3916|

两种采样在同一个场景都未找到可行候选。该scene的真正safe-max及其安全收益均记为缺失；主固定分母表只将safe-max替代为0，明确是一项保守统计代理，不能解释为真实安全零分轨迹。127个有可行候选的scene中，两种采样的safe max都等于overall max；缺失scene不参与这个等值断言。

SDE−ODE safe-max零代理差为+1.5128点，按log聚类的配对95%区间[1.0951,1.9549]。SDE安全best比ODE安全best至少高0.5点的scene有61个，反方向有3个。区间仅反映这组固定K16结果的log抽样，不覆盖新生成噪声、训练seed或更大搜索预算；它不是理论oracle上界。

## 安全改进发现率的两个不同基线

以下阈值均为原始score增益≥0.005，即百分制0.5点；分母始终128，缺失安全候选不计成功。

|固定逐scene参照|SDE发现至少一条安全改进|ODE发现至少一条安全改进|
|---|---:|---:|
|共同BASE同root ODE16均值|75/128（58.59%）|68/128（53.12%）|
|独立五reference均值|74/128（57.81%）|59/128（46.09%）|

第一行参照是本审计固定BASE的16条ODE均值；对BASE的ODE max来说并非独立参照。第二行来自共同bank预先固定的独立五个reference种子11001–11005，其人口与评分身份不变。两类参照不混用，不用全局均值替代各scene基线。

## 固定strata

|固定层|scene数|SDE/ODE候选均分|SDE/ODE安全max零代理均分|
|---|---:|---:|---:|
|稳定高分|98|73.8964 / 97.3863|98.8986 / 98.0369|
|稳定零分|1|0.0000 / 0.0000|safe-max缺失；代理0 / 0|
|低分非稳定零分|1|39.0625 / 71.3542|100.0000 / 100.0000|
|其他/波动|28|63.8299 / 84.7823|89.8684 / 85.9685|

分层固定于原五reference，不按本次SDE/ODE重划。稳定零分和低分层各只有1个scene，不能据此推广失败类型的总体可修复性。有限探针未找到可行轨迹不证明不可能解决。

## 读取与成本

新增CPU分析入口与分母/缺失单测已执行，无新模型前向、rollout或评分。输入4096条已有记录：2048条历史SDE校准记录、2048条BASE审计新ODE记录；不隐去缓存构建成本。原result、COMPLETE和候选文件均保持不变。聚合详情见[JSON](base_matched_root_opportunities_v1.json)，逐scene均值和缺失safe-max仅保存在私有CSV。

```bash
CUDA_VISIBLE_DEVICES='' /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.qwen_flow_exploration.preservation_analyze_sampler \
  --protocol /mnt/project/qwen-flow-three-routes-20261024/common_protocol.json \
  --audit /mnt/project/qwen-flow-three-routes-20261024/preservation/terminal_sampler_audit_v1/BASE \
  --reference-pool /mnt/project/qwen-flow-three-routes-20261024/preservation/calibration_v1/reference_pool.pt \
  --reference-strata /mnt/project/qwen-flow-three-routes-20261024/common_bank/reference_strata.json \
  --output /mnt/project/qwen-flow-three-routes-20261024/preservation/analysis/BASE_sampler_opportunity_v1.json
```

P0–P5复用同入口时，额外传入`--base-audit <固定BASE审计目录>`，避免把各自当前ODE均值悄悄改成改进基线。原示例输出已存在，复跑需另选结果路径。
