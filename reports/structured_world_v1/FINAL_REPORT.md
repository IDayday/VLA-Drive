# 结构化世界认知 V1 最终报告

`engineering_status = READY`（本次实现与明确支持的恢复边界）
`research_status = INCONCLUSIVE`

真实当前图像→Reader/几何 BEV→固定 scene/agent tokens→Qwen→当前框与未来轨迹监督→原 Flow-Matching DiT 的闭环已完成。关闭 world 可回到原检查点原推理路径。真实训练、推理、保存加载、外部特征缓存及两卡联合训练恢复均已验证。**READY 不代表感知已收敛、BEV 先验已充分适配或规划有效。** 有限预算内的目标召回很低，配对规划区间跨零，不能判为 POSITIVE，也不能据此否定这个方向。

## 基线、代码与输入

- 分支：`feature/structured-world-v1-20260926`。完整实现提交：`4ed9bbfe709e920c59e4c14f09272e565f2a7665`；最终报告另行提交，交付 HEAD/远端 SHA 见最终答复。
- 可加载基线源：`0ecd2ae1f616844641a6d94cfafb50e0f32c26fe`；接口参考 `9dd6b71b324a58de005dc669394295b9354d189c`，未整分支合并。
- 基线：本地 released DriveDreamer-Policy `pytorch_model.pt`，SHA256 `9445f9da577a8e3c6b7c636c60a98d714d602668f9a8033b0891c703bc40210f`。原始训练源码 SHA 无法确认；不将兼容源码冒充历史训练 SHA。完整 key 审计零缺失，1965 个未用 Wan key 明确列出。
- 输入统一为当前 CAM_F0/L0/R0，原自车历史状态和导航；无历史图像、无环视、无 GT 特征。保持原 prompt/辅助 query；没有用修改 prompt 的模型冒充 A0。
- 固定开发集1696场景/16完整log；增量训练8192场景，按log排除开发集。基础检查点是否见过这些log为 UNKNOWN/可能见过，不声称完全未见。详见 [BASELINE_MANIFEST.json](BASELINE_MANIFEST.json)、[SPLIT_AUDIT.json](SPLIT_AUDIT.json)。

## 实现与真实训练范围

固定64 scene＋64 agent queries，Reader维度256、两层；Qwen/DiT宽度从模型读取。新连续 token 插在原 action tokens 前，保留 native image token、DeepStack、mRoPE和因果mask。bbox/8步未来xy头读取经过Qwen的agent hidden；当前Hungarian一次匹配，未来沿同一track ID在ego(t0)坐标监督。当前框是感知，未来轨迹是条件运动预测；不声称动作干预下的反事实动力学。

核心组训练原DiT、Reader、queries/types、投影及相应结构化头。Qwen、视觉、原辅助模块固定；所有组Qwen LoRA均关闭。A1仅训练DiT，A2无世界监督，B/D关闭motion loss。零损失头无有效监督梯度；image-only Reader位置层不使用。日志区分梯度、optimizer成员和实际参数变化。视觉解冻开关经真实梯度及参数更新验证，但未展开独立训练矩阵。

D/E使用真实camera-only几何BEV：49×40个1m网格，x=[1,50)、y=[-20,20)，z={0,1,2}m，标定投影当前CNN特征并融合。它是**轻量、从零训练的provider，不是BEVDet预训练先验**。最终provider有效适配600步/64训练场景；另外800步provider排障成本同样计入总预算。当前修正标签下的独立感知核验：22/1043目标匹配，class-correct 2m recall **2.1093%**，误报40，匹配中心误差1.1664m；该最终重算为真实图像CPU FP32推理，原适配和在线路径已在GPU实测。显然不足以声称先验充分适配。

BEV特征确实进入Reader→Qwen→action queries；在线、离线缓存、ExternalBEVFeatures接口得到逐元素相同的动作。缓存验证源/权重、图像、标定、时间、sensor contract、grid及dtype。训练不复用最终VLM hidden缓存。部署仍需要provider；它的在线GPU耗时约4.04ms（20次、当前RGB tensor已在GPU），完整策略时间包含provider，详见成本表。

与WCog的差异：保留现有Qwen3-VL和DiT，采用独立轻量Reader、单模态周车xy预测和可选动作cross-attention；没有复现其完整训练配方/外部先验，也没有联合多车扩散、轨迹VAE、Game-CoT、未来RGB重建、RL、新scorer或路由。来源与license见 [THIRD_PARTY_SOURCES.md](THIRD_PARTY_SOURCES.md)。

## 全量开发集结果

所有行均为同一1696场景，1候选、10 FM步、8×0.5s、固定逐场景随机种子、原精度边界、无模型scorer。这里是NAVSIM **v1真实PDMS**，不是网络预测分数。单候选的候选均值/oracle/选中值相同，没有独立best-of-K部署增益。

| 组 | 训练/标签版本 | PDMS /100 | 零分比例 | 当前目标2m召回（类别无关） | 失败场景 |
|---|---|---:|---:|---:|---:|
| A0 | 原路径 | 93.1446 | 1.238% | — | 0 |
| A1_seed42 | 历史监督 /1000步 | 89.8049 | 4.658% | — | 0 |
| A2_seed42 | 历史监督 /1000步 | 88.3609 | 5.837% | 0.0000% | 0 |
| B_seed42 | 历史监督 /1000步 | 87.2141 | 6.899% | 0.6089% | 0 |
| C_seed42 | 历史监督 /1000步 | 88.5270 | 5.896% | 0.4410% | 0 |
| D_seed42 | 历史监督 /1000步 | 90.5701 | 3.833% | 0.1593% | 0 |
| E_seed42 | 历史监督 /1000步 | 89.5380 | 4.835% | 0.5179% | 0 |
| B_seed43 | 历史监督 /1000步 | 91.7761 | 2.771% | 0.0000% | 0 |
| C_seed43 | 历史监督 /1000步 | 89.9711 | 4.599% | 0.0085% | 0 |
| C_support_v6_seed42 | v6 修正后 /400步 | 90.7067 | 3.302% | 0.0000% | 0 |
| E_support_v6_seed42 | v6 修正后 /400步 | 90.4992 | 3.538% | 0.0313% | 0 |
| E_adapter_support_v6_seed42 | v6 修正后 /400步 | 91.2704 | 2.830% | 0.0000% | 0 |

历史B/C/D/E与第二种子训练完成后，边界审计发现畸变多项式非单调分支导致FOV误判。修正影响8192训练场景中的83个当前track集合（93个移除、12个因容量补位加入），21组标定合计54个唯一网格差异。旧数据不覆盖、旧结果不删除；这些历史行不冒充修正后完整矩阵。A1/A2的优化不依赖世界标签。新缓存v6有8192/1696完整覆盖、零生成失败；修正后C/E/adapter各400步从原基线独立启动。所有感知统计均用完整v6开发GT重算；规划轨迹和PDMS未因标签重算而修改。

配对log-cluster bootstrap（16log、10000重采样，单位百分点）：

- 历史C−B，seed42：+1.313，95%CI [−1.405,+2.922]；seed43：−1.805，[−5.235,+0.067]。方向不稳定。
- 历史E−D：−1.032，[−2.742,+0.481]。
- 修正后E−C：−0.208，[−1.191,+1.679]。
- 修正后adapter−E：+0.771，[−0.169,+1.705]。

这些区间只描述该训练种子下的log抽样不确定性，不能替代跨训练种子稳定性。修正后C/adapter检测覆盖为0，E仅约0.0313%；条件ADE/FDE在无匹配时记空值，不伪填0。全匹配assignment motion误差、端到端覆盖、类别正确召回、距离/相机观测支持分组和误报均在完整CSV。由于感知与provider明显欠拟合、完整修正矩阵未重跑，结论为 **INCONCLUSIVE**。

## 验证、修复与预算

- 8项聚焦CPU测试通过：坐标/静止目标、裁剪内参、token异常、几何投影、空目标/无效future、loss累计等。
- 真实GPU：FP32原路径中间输出最大差0；BF16世界关闭、目标污染、padding、保存加载通过。BF16预声明atol/rtol=.02，核心比较实测0。
- provider在物理不存在target目录的当前观测请求中可推理；未来时间、错标定/版本/相机/scene及关键missing/unexpected权重会拒绝。
- 冻结Qwen能回传新query梯度；视觉解冻发生实际参数更新；真实两场景8次FM重复条件/动作顺序一致。
- 两张真实GPU/NCCL：空目标rank及累计有效数归约与单卡参考最大梯度差1.49e−8；Qwen/world/原DiT联合训练的两rank恢复参数及query梯度差均为0。没有把CPU mock称为GPU结果。
- 单进程跨Python重启4步与2＋2：参数、optimizer、scheduler、随机状态和采样位置一致。
- 快速CUDA provider适配续训曾超过预声明1e−6容差（实测6.21e−6），失败记录保留。修复为显式`--deterministic`模式：训练时CPU确定性几何重采样保留跨设备autograd，GPU CNN继续训练，部署仍走GPU。真实4步与2＋2参数差0。Fast模式不宣称bitwise续训；不放宽原容差。

总计 **11639/12000 optimizer steps**，余361；包括失败、两轮收敛/超参排障、几何正确性修正、201步中断adapter和全部GPU更新测试。两轮超参数修复已用完；此后只做坐标/恢复正确性修复，未继续调loss权重或搜索测试成绩。初始归一化hidden语义错误的800步已计费并排除科学比较。没有OOM的新主实验，旧任务的OOM不属于本次证据。原工作区、原数据/缓存/检查点保持不变。

工程READY依据是修正后真实端到端训练/推理、真实BEV、严格边界及关键GPU回归已完成，不是以loss下降代替科学结论。详细记录见 [VALIDATION.json](VALIDATION.json)、[RUN_LEDGER.json](RUN_LEDGER.json)、[GEOMETRY_CORRECTION.json](GEOMETRY_CORRECTION.json)、[ARCHITECTURE_AND_GRADIENTS.md](ARCHITECTURE_AND_GRADIENTS.md)。

## 成本与交付

主训练batch1，A1可训练819,503,620参数，C为823,299,108，E为822,791,204，adapter为839,580,709。载入总参数包含保持的旧辅助模块，见机器可读ledger。C/E峰值约25.55/25.54GB（十进制），adapter25.81GB；观察到约1.48–1.50 samples/s。完整C/E/adapter推理平均约429/436/436ms，包含provider但不含图片读取；A0约294ms。设备均A800-80GB，但跨host/并发/缓存状态不同，这些是实测运行成本，不是严格隔离的延迟因果比较。

- [完整逐场景CSV：20352行，PDMS分项＋感知/运动/延迟](/mnt/project/structured-world-v1-artifacts/20260926/final_metrics/scene_metrics.csv)
- [完整配对差值CSV：28832行](/mnt/project/structured-world-v1-artifacts/20260926/final_metrics/paired_scenes.csv)
- [汇总指标](SUMMARY.json) 与 [全部配对区间](PAIRED_COMPARISONS.json)。
- [64个修正后真实场景图索引](/mnt/project/structured-world-v1-artifacts/20260926/visualizations_C_support_v6_seed42/index.json)：空目标3、缺失future41、静止48、转弯17、交汇6、离开当前FOV/ROI34；类别为明确的几何启发式标记，可重叠。这是事后诊断选图，未筛正式输入/评测样本。图像含当前框投影、ego(t0)框、同track future、mask、slot匹配。
- 权重、全部目标/特征缓存、图像与场景CSV留在授权存储，不推Git；[ARTIFACT_INDEX.json](ARTIFACT_INDEX.json)记录位置/校验。代码、配置、聚合指标、命令和小型证据进入本任务分支。

NOT_RUN：完整v6 A0/A1/A2/B/C/D/E 1000步重跑、视觉解冻大矩阵、LoRA训练、BEVDet依赖安装/公共权重引入、最终navtest。未选出可晋级模型，因此不消耗navtest用于选择或调参。轻量provider的低适配能力不能解释为“BEV注入无效”。正式checkpoint没有复用跨场景错配BEV；该类OOD干预不作因果证明。

## 一条恢复命令

以下命令已实测；已完成checkpoint返回`already_complete`且不新增optimizer step。未完成checkpoint只恢复原来预留的剩余步骤。

```bash
cd /mnt/project/VLA-Drive-structured-world-v1-20260926 && source /mnt/project/structured-world-v1-artifacts/20260926/campaign.env && "$WORLD_PYTHON" tools/structured_world/resume_run.py --checkpoint "$WORLD_ARTIFACTS/C_support_v6_seed42/checkpoint.pt"
```

完整数据、缓存、训练、分片评估、CPU打分、恢复和可视化命令见 [快速开始](../../docs/STRUCTURED_WORLD_V1_QUICKSTART.md)。本机GPU0–3及vla-zt2原占卡脚本已恢复；本机GPU4–7原任务未停止。没有实验训练/评估仍在后台继续扩大预算。
