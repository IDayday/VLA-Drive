# 原生Qwen-Flow安全RL：论文复核、真实验收与正式训练

2026-10-24。两组安全改进已完成实现与真实八卡验收，均从原IL53k重新启动50,000次optimizer更新。它们已进入实际学习；目前还没有两种新方法的完整DEV或Navtest结果，不能据此宣布安全退化已解决。

用户补充要求重新核实安全与进度平衡、GRPO改进论文后，正式方法由最初的SCADV3＋X1调整为WLA_ADV＋SCADV3。调整发生在正式启动前；旧登记与X1短验收保留，未额外启动第三条50k训练。详细来源、公式、代码锁定及适配边界见[论文复核](paper_review.md)、[来源锁定](paper_source_lock.json)和[最终登记](experiment_registration.json)。

## 已启动的两组

以下为2026-10-24 **14:28 UTC**的实际快照，更新数取八个rank已提交日志；恢复点另列，不把未落盘更新写成可恢复进度。[核实记录](formal_start_verified.json)

|方法|实际服务器/GPU|八rank实际更新|已封存恢复点|正式终点|
|---|---|---:|---:|---:|
|R_CANONICAL_WLA_ADV|vla-zt，0–7|237|200，READY|50,000|
|R_CANONICAL_SCADV3|tsinghua-zt4，0–7，与原SOURCE共享|160|100，READY|50,000|

单一CPU campaign拥有任务与评价队列，训练源码固定为`ee200bc11f03b026ee54c02e1997eb7db8a57861`，不可变目录为`/mnt/project/qwen-sourceflow-safe-rl-20261024/snapshots/formal_v3`。本报告后续提交不热改运行目录。正式run位于：

- `/mnt/project/qwen-sourceflow-safe-rl-20261024/runs/RL_WLA_ADV_seed42`
- `/mnt/project/qwen-sourceflow-safe-rl-20261024/runs/RL_SCADV3_seed42`

原CANONICAL已达到50k后释放资源，本任务没有中断它；原SOURCE、原IL及三时刻DINO版本保留。未知GPU任务不驱逐。`vlawm-zt`、`tsinghua-zt6`的授权仅用于Navtest，没有用于本轮安全训练；vla-zt3留给其既有临时授权任务。[资源范围](RESOURCE_AUTHORIZATION.json)

## 为什么选这两项

**WLA_ADV**迁移DriveWLA的优势估计器：NC/DAC/EP/TTC/Comfort分别标准化，权重`[2,2,5,5,2]`；组合再按组方差缩放，并乘原PDMS方差的收缩因子`var/(var+0.1²)`。用于检验分项信号与抑制微小差异是否减少无益漂移。它没有显式违规预算，且全零PDMS组仍无该优势信号。这里只迁移论文公式，不声称完整复现其模型、十步随机采样、BC或训练预算。

**SCADV3**迁移此前已实现的分项约束优势：分别标准化PDMS和NC/DAC/TTC事件成本，再计算`Z_PDMS − Σλ_j Z_cost_j`。三个独立OmniSafe式乘子依据未筛选fresh样本更新一次，四次inner固定同一λ快照；组合后不再whitening或clip5。它是Constrained-GRPO/GDPO思想的明确适配，独立乘子、成本与预算不同于完整原论文，不能借用其保证。

三个成本为`I[官方分项<1]`，包括NC=0.5。预算来自固定512个公开难例TRAIN场景、4096条原SFT真实随机轨迹，分别为4.1748%、13.8184%、13.9160%。这是有限TRAIN随机策略的参考均值，不能解释为ODE事故率、逐场景预算或硬安全保证。[校准聚合](calibration_summary.json)

Run-then-Walk是直接研究进度—安全平衡的相关工作，已核实论文和官方reward/launcher，并记录其阶段reference切换及论文/脚本预算差异；没有将它简化成当前λ方法或混入这两组。Plan-R1、Dr. GRPO、DAPO的适用边界同见[复核文档](paper_review.md)。

## 匹配合同

两个新组和既有CANONICAL从同一个完整IL53k开始，模型SHA256为`4c5188dec5674fc08f4b75b1a464c05348bf8a54623fd85f6598fb921ae5e786`。正式初始化与既有CANONICAL的全部480个LoRA张量、AdamW、scheduler严格相同；八rank第一份fresh collection的场景、完整十步chain、old概率及PDMS逐项一致。两组优势有意不同，后续策略应随学习分化。[实际初始化及首批对照](formal_initialization_and_first_chain_equivalence.json)

共同设置：公开困难TRAIN交集17,037场景；训练seed42、评价seed2101；G8/K10、shift5、仅步骤[7,8,9]随机；源rigorous SDE、创新±5及逐维relative logprob clamp；H/D mean代理、PPO clip0.005、inner4；LoRA rank16/alpha32/dropout0；AdamW LR5e-5、100步warmup后constant；有效gradient clipping仍为源配置的0；永久IL reference、均值MSE锚定beta50及原3维权重`[0.2,1,1]`。无GT replay、BC、learned-noise、critic、外部搜索或多候选部署。

每组固定50k更新＝12,500份fresh全局rollout＝100,000次scene曝光＝800,000条新候选。四次inner不重复计新数据。DEV观察0/5k/10k/20k/35k/50k，固定50k与登记的DEV选择结果分开报告；最终检查点锁定后测全量Navtest12146场景/136logs。主协议始终是seed2101、原始观测、单候选十步shift5 ODE、canonical v1.1。采样策略、候选上界与部署成绩分表。

既有Navtest已经被观察并参与用户指定IL53k，不能宣称本轮是完全盲测。预算校准没有使用DEV/Navtest。新方法是否有效将由重新采样后的ODE总分、NC/DAC/TTC新增违规及恢复、零分转移、G+/G−及配对log-bootstrap共同判断。

## 已执行的验证

|验证|实际结果与证据|
|---|---|
|分项优势、λ符号、floor、全等组、DriveWLA公式|CPU小张量及tiny ActionDiT四次真实更新通过；[记录](safe_math_papers_v4.json)|
|既有基础算术及合并速度输出|通过；[记录](baseline_math_v1.json)|
|真实八卡16次更新，每组256新候选|480个LoRA张量全部改变，A/B各240个Adam一阶矩非零；Qwen/provider/base/reference冻结hash不变；[记录](actual8GPU_continuous_profiles_v2.json)|
|实际采样分布与固定链重算|两组各32个scene groups的mean/std严格一致，第一inner最大ratio误差0；后续允许真实更新|
|中断后继续剩余inner及后续collection|两组adapter/optimizer/scheduler、八rank RNG/链/old概率/优势完全一致；[SCADV3](SCADV3_actual8GPU_resume_PASS_v2.json)、[WLA_ADV](WLA_actual8GPU_resume_PASS_v3.json)|
|合并LoRA后原始图片十步ODE|两组均通过；不依赖训练标签/cache/teacher/scorer；[SCADV3](scadv3_actual_ODE_export_parity_v2.json)、[WLA_ADV](wla_adv_actual_ODE_export_parity_v2.json)|

两次失败验收保留：SCADV恢复时诊断tensor的CPU/CUDA比较不一致；WLA恢复时派生FP64统计存在不超过3.56e-15的跨设备舍入差。修复只涉及验证device及诊断容差，保存的FP32优势仍严格核对。两组随后分别重跑并通过真实恢复；正式run没有承接失败或qualification权重。[失败与成本](ISSUES.json)、[只改验证的等价性收据](validation_only_FP64_receipt.json)、[最终资格](QUALIFIED.json)

资格通过证明实现及恢复可用，不等于证明规划收益。原始SimWAM的部分随机步骤及裁剪概率仍属于经验代理，本轮没有将其改称精确整链概率比。

## 成本与执行入口

真实16-update资格窗口中，最慢rank的rollout/评分/概率/更新合计分别为WLA_ADV 2.265秒/update、SCADV3 3.178秒/update，均与旧训练或评价共享GPU；峰值约35.65GiB/GPU。该窗口不包含模型加载、保存和资格hash检查，不能直接当完整训练ETA。失败验收的135.7秒与111.6秒另计，未从物理成本中删除。登记的约28–34小时/组是独占估算，共享时需按正式完整窗口更新，不能把共享worker wall当独占GPU计算时间。

已执行的CPU资格命令如下；可写入另一个输出文件复核，不会改变正式训练：

```bash
cd /mnt/project/qwen-sourceflow-safe-rl-20261024/snapshots/formal_v3
CUDA_VISIBLE_DEVICES='' /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.qwen_sourceflow.validate_safe_rl \
  --output /tmp/qwen_safe_math_check.json
```

当前已运行的单owner控制器入口如下，**不要在现有owner运行时重复启动**。它复用既有派发、正式训练、LoRA导出、DEV/最终Navtest和配对汇总；当前controller遇到worker失败会保留证据并停止自动重试，不盲信failure checkpoint：

```bash
CUDA_VISIBLE_DEVICES='' /mnt/project/flow-safe-improvement-20261013/venv/bin/python -u \
  -m scripts.analysis.qwen_sourceflow.safe_campaign \
  --training-source /mnt/project/qwen-sourceflow-safe-rl-20261024/snapshots/formal_v3 \
  --analysis-source /mnt/project/qwen-sourceflow-safe-rl-20261024/snapshots/formal_v3 \
  --evaluation-source /mnt/project/qwen-sourceflow-native-20261018/snapshots/RL_parent53k_user20261023
```

正式训练实际命令、主机、GPU UUID及启动回执在私有根目录的`campaign/jobs/{wla_adv,scadv3}_formal50k.json`，单一owner状态在`campaign/status.json`。两组均使用真实存在的`train_safe_rl`入口和对应[WLA配置](../../configs/qwen_sourceflow/rl_wla_adv.yaml)、[SCADV3配置](../../configs/qwen_sourceflow/rl_scadv3.yaml)。

需要恢复时，先确认旧worker已退出及当前授权，验证同run的`latest.json`所指COMPLETE合法边界，保留原命令、run输出和immutable源码，并增加`--resume <合法checkpoint目录>`。SCADV3必须保留原`--safety-budget`；跨源码恢复还必须提供经过等价性验收的`--migration-receipt`，不能通过重采链、重置Adam/λ或改身份续跑。这里不自动执行恢复，也不重复启动现有任务。

Git只保存源码、配置、许可证、论文链接/hash及聚合验证；完整论文、场景清单内容、图片、权重、chain及大cache仍在私有资产目录。
