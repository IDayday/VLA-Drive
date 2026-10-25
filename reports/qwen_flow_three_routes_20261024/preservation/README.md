# 路线三：固定成功存档与成本约束

截至本记录，已实现P0—P5并通过实际原生模型资格；共同候选库和固定存档已经真实完成，正式训练由主agent按统一GPU排程执行。**当前没有本路线已完成的正式PDMS增益或能力保留结论。** 单条DEV资格轨迹仅证明更新后的ODE能执行，不作为研究效果。

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
- [纯部署CPU验收](qualification_deployment_CPU_v5.json)：9项通过。使用真实tiny ActionDiT和非零LoRA，直接FP32 `B@A*2`合并与源wrapper/源merged_state_dict一致，完整十步ODE误差在已登记容差内；最终模型没有LoRA wrapper或reference实例。对P组以及S1/S2/S3各自真实checkpoint schema核验后，只导出adapter字典；删除合成训练父目录后仍可加载纯部署包，不读取teacher、archive或dual。
- 同一CPU验收还只读检查已有真实L0场景的seed、canonical分项、请求与模拟后果关联，错误seed、reward或已评分请求均拒绝；没有新增模型推理/评分。[一次资格接口失败及修复](qualification_deployment_issue_v4.json)单独保留，未改变已有结果。
- [薄分析封装算术验收](qualification_analysis_CPU_v1.json)：仅调用公共`common_analysis`完成seed先归scene、log bootstrap；已验证P5−P3−P4+P0的已知数值、缺臂显式标记和不同训练seed拒绝，不另写一套统计器。

固定存档现覆盖512个TRAIN场景中的435个（84.96%），共1667条目标；1/2/3/4条目标的场景数分别为10/15/13/397。来源为每scene的32条IID候选及5条独立reference，共18944条已评分请求；77个场景没有合格目标，保持无标签。[实际存档摘要](fixed_archive_initial_summary.json)。P2/P3/P5的监督回放人口因准入而有偏，全部512场景仍进入PG人口。这不是新策略单轨成绩。

### P0首次正式派发的资源故障

2026-10-24约19:02 UTC，P0在首个inner概率重算时发生共享GPU显存OOM：本进程约26.88GiB，另两进程分别约38.06及14.27GiB，79.25GiB卡仅余约20.5MiB。随后异常证据写入因父目录尚未创建而二次抛错，掩盖了原OOM。[完整分类收据](formal_failure_P0_initial.json)。这不是评分或算法无效的证据；没有optimizer更新完成。

只读严格核验确认latest仍是合法`ROLLOUT_COMMITTED/update0/next_inner0`，全部16×G16=256条轨迹的评分、old概率、优势与chain摘要完整，Adam state尚为空。恢复应继续这同一批链，不能重新采样后沿用旧概率。原日志、checkpoint和run文件未被本次修复覆盖。

开发补丁仅在私有trainer启动时创建`failure_evidence`目录，不修改actor、sampler、loss或optimizer。[CPU故障注入](qualification_failure_CPU_v1.json)已验证：首inner OOM保留原始异常且不推进latest；第二个scene评分失败保留局部证据且latest仍READY。既有immutable快照可在预建目录后精确恢复，无需为这一目录修复迁移科学身份；但必须先由主agent安排足够稳定显存，目录修复不会解决容量OOM。本次没有自行派发GPU重试。

### 已完成的TRAIN校准与实际信号

`calibration_v1/calibration.json`已经完成并锁定。校准使用固定TRAIN前128场景、117个完整log的2048条初始source-SDE候选，未筛选；三个成本预算为`[0.036376953125, 0.1318359375, 0.12353515625]`，完全核对原始反馈后重新求均值一致。[只读分析及配对区间](completed_calibration_analysis_v1.json)。下表在同128场景上比较均值；没有使用best-of-N，也不把此表当DEV/Navtest部署结果。

|初始同一base的采样|PDMS|NC均值|DAC均值|TTC均值|Comfort均值|联合完全安全候选|
|---|---:|---:|---:|---:|---:|---:|
|训练source-SDE，G16|70.8449|96.3623|86.8164|87.6465|64.6484|76.2695%|
|部署ODE IID，K32均值|93.5387|99.9756|99.1943|98.9258|99.9023|98.1201%|

SDE−ODE均值PDMS差为−22.6938点，按log聚类的配对95%区间约[−25.7569,−19.8180]。采样RNG、候选数和两种转移核不同；该区间仅覆盖这128场景的log抽样，不含模型训练seed或新推理seed不确定性。它支持“训练随机链与部署ODE条件明显不同”的诊断，不能单凭此差值确定需要更换sampler。

source-SDE中127/128组（99.22%）有真实PDMS优势信号，仅1组全零、0组全满；107组的NC/DAC/TTC至少一项有候选差异，严格progress-only组为0。相同场景的ODE32则有42个等分组（41满分、1零分），有效组中89.53%仅进度变化。这两种统计不应混在一起继续解释为“当前PG仍因满分而没有更新”。128场景分成8个16-scene虚拟collection时，初始cost标量化没有进一步激活那一组全零；这是离线信号检查，不冒充正式collection。

成本预算参照的是初始**训练SDE**，不是初始部署ODE。SDE的NC/DAC/TTC平均成本分别为3.638/13.184/12.354%；ODE对应0.024/0.806/1.074%。因此即使P4/P5在训练人口上满足这些预算，也不能据此声称部署ODE不退化。NC成本是`1−NC`而不是事件率：7条NC=0.5的候选令SDE的NC完全通过违规率3.809%与平均成本3.638%不同。DAC/TTC本次取值下二者相同，语义仍分别保留。

回放系数按预登记上限锁为1，未再调参。固定4场景probe上PG梯度norm为0.034618、replay为0.001830，实际replay/PG梯度比约5.29%，低于25%校准目标，因为计算出的4.728系数触及上限1；梯度cosine为+0.1845。不能仅看beta=1就认定回放很强，也不能从这一批次推断所有后续梯度关系。

P1仅进行了两个允许的TRAIN LR试验，最终倍率0.5但匹配状态明确为`NOT_MATCHED`。P5四步公共root轨迹漂移ADE约0.006708米，P1约0.003414米，最大相对误差49.10%，超过20%容差。后续P5−P1不能声称已经排除“只是更新更保守”的解释；不为修饰匹配结果继续调参或重跑。

校准物理账本含2048条新评分、12次影子optimizer更新、约1354.67秒累计worker占用；其中生成约901.96秒、更新约270.87秒、评分等待约26.89秒。校准权重已丢弃，不进入正式P组初始化；这些成本与候选库构建成本单列。当前仅能定位训练信号与部署/保持边界，尚不能给出P0–P5的实际部署收益结论。

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

新增纯部署入口同样支持S1/S2/S3，统一使用`native_qwen_sourceflow_ODE`的CPU scene-bound标准高斯root，每个固定seed仅一条shift5十步ODE。默认DEV_SELECT固定256场景×5seed，B1推理；DEV_AUDIT仍由公共gate封存。旧反馈学生自动评价的不同RNG namespace仅保留辅助身份，不混入本入口的主配对比较。以下入口已执行`--help`及CPU验收，实际GPU评价由主agent在相应训练COMPLETE后派发：

```bash
$MODEL_PY -m scripts.analysis.qwen_flow_exploration.preservation_evaluate \
  --protocol <common_protocol.json> --arm P5 --seed 42 \
  --checkpoint <独立P5正式run或其COMPLETE checkpoint> --output <新的单轨部署评价目录>
$MODEL_PY -m scripts.analysis.qwen_flow_exploration.preservation_analyze \
  --protocol <common_protocol.json> --reference <统一L0_sample评价目录> \
  --reference-strata <common_bank/reference_strata.json> \
  --evaluation P0=<P0评价目录> --evaluation P3=<P3评价目录> \
  --evaluation P4=<P4评价目录> --evaluation P5=<P5评价目录> \
  --archive <固定archive.json> --calibration <TRAIN calibration.json> --output <新分析目录>
```

最终评价只依赖完整base和纯adapter包，加载训练checkpoint是一次独立离线提取。推理时不构建永久reference，不加载存档或teacher，不把dual作为actor输入。未更新权重的BASE与共同L0_sample可以复用，避免重复评分。分析入口拒绝qualification、不完整scene/seed人口和重复候选池；P3−P2、P5−P1及存档×约束对照会显式列出缺失实验。

### 单独登记的终点SDE→ODE迁移审计

主部署仍是前述pure-merged、DEV固定五seed、每seed单候选ODE。另实现了只读终点审计，用于区分“训练SDE均值提高”与“同root的ODE也提高”。这是评估预算，不增加算法训练臂。

固定TRAIN128、每scene16条：六个P终点分别恢复实际LoRA，在全参数冻结、eval/no_grad状态下重跑原source SDE；新SDE的全部x0必须逐值等于初始reference_pool保存的x0，完整CPU RNG流也必须一致。随后使用同一当前未合并专家、同16个x0生成ODE并评分。BASE复用初始2048条SDE反馈，只补2048条同root ODE；六臂各新增4096条评分，因此合计26624条新评分。初始SDE的历史2048条成本只有一个共享来源，不能从总账中消失，也不能重复计为新评分。

输出分别是SDE初始均值、SDE终点均值、ODE初始均值、ODE终点均值和配对差值；不使用best-of-N。审计专家保留LoRA wrapper，数值路径与训练采样一致，明确区别于主表的纯合并部署，不能将两种人口或精度路径混成一个分数。

[CPU六项资格](qualification_sampler_audit_CPU_v2.json)通过，包括实际tiny源actor的adapter恢复、冻结hash/无梯度、源原始随机数调用、部分结果原子恢复、候选均值非oracle、预算身份/缺候选拒绝；还只读核对真实reference_pool的128×G16根噪声和源CPU RNG边界全部逐值一致。没有新GPU推理或评分。每scene完整请求、实际模拟后果、SDE chain/概率和RNG边界仅在私有artifact目录保存。

CLI已执行`--help`及CPU验收；主agent必须先封存独立`TERMINAL_AUDIT_LOCKED`预算并集中派发，不修改运行中的immutable训练快照：

```bash
$MODEL_PY -m scripts.analysis.qwen_flow_exploration.preservation_audit_sampler \
  --protocol <common_protocol.json> --budget-lock <terminal_audit_budget.json> \
  --calibration <calibration_v1/calibration.json> --reference-pool <calibration_v1/reference_pool.pt> \
  --arm BASE --output <新BASE审计目录>
$MODEL_PY -m scripts.analysis.qwen_flow_exploration.preservation_audit_sampler \
  --protocol <common_protocol.json> --budget-lock <terminal_audit_budget.json> \
  --calibration <calibration_v1/calibration.json> --reference-pool <calibration_v1/reference_pool.pt> \
  --arm P5 --seed 42 --checkpoint <已COMPLETE的P5run> \
  --base-audit <已COMPLETE的BASE审计目录> --output <新P5审计目录>
```

## 当前研究判断

**证据不足**，仅针对目前尚未完成的正式实验，不是对存档或约束方法的否定。必须等待P0/P3/P4/P5因子对照、P2单轨和P1幅度匹配控制，再以自然评估人口上的净收益、G+/G−及固定高分场景退化判断。

最有信息量的下一项是完成同预算P0/P3/P4/P5，而不是增加archive容量或新的dual算法；若P0本身几乎不退化，还应明确这轮保留问题的统计判别力不足。


2026-10-24 23:24机制快照：已提交的新源SDE组PG有效率100%，不能继续把这组训练缓慢归因零优势；第一inner clip0属正常，后几inner clipping也仅约0.13–0.42%。三个成本乘子变化仍小，大部分约束作用来自初始化成本惩罚。[中期机制及修正成本](interim_mechanism_v1.md)。这不是终点或ODE收益结论。

P1完成一次合法mid-inner资源迁移：81→84的保存chain/old概率/奖励/优势/RNG逐值摘要不变，期间无新增评分；Adam与scheduler连续。暖inner约108.38→79.97秒，旧主机的P1–P5随后仅在原生边界退出并使用相同97源码恢复，P0保留本机分散CPU负载。[P1实际迁移证据](migration_P1_exact_boundary_v1.json)。


同x0的安全有限候选上界实际更高：SDE safe-max零代理96.1592，ODE94.6464；可行率均127/128。SDE候选中位数也显著较低，因此均值差并非只来自极少尾部，更不是完全没有好候选。[严格分母与机会分析](base_matched_root_opportunities_v1.md)。
