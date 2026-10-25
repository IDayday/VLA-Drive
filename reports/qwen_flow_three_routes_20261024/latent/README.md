> **2026-10-25用户更新：方法有效性与性能提升只依据完整Navtest。本文既有DEV/VAL与定向TRAIN数值仅作工程、拟合和机制诊断，旧阶段价值标签不再作为当前结论。测试结果未完整前，一律待确认。**

# 路线一：冻结生成器的潜噪声策略

共同起点为成熟CANONICAL50k完整Qwen-Flow（固定Navtest90.513164），不是换回较弱IL来制造增益。VLM、provider、30层ActionDiT及全部buffers冻结；策略只改变原生ODE输入的24维root随机变量。原部署本来就是scene绑定的标准高斯root＋10步shift5 ODE，ODE内部没有额外随机xi。

## 实现与数学边界

- 条件L1/L3读取已有合法condition：text/current-state context和最后层视觉V分别masked mean，拼接得到7168维，进入小型LayerNorm＋MLP[128,64]。完整视觉memory仍由原ActionDiT读取，没有被本策略池化替换。L2只学习全局48个输出参数，不读取条件。
- 输出24维mean与有界log_std。log_std范围为±log2，std为[0.5,2]；最后层零初始化，使初始q严格等于N(0,I)。所有密度对8×3完整求和，不是dimension-mean denoising代理。
- 采样z、实际old mean/log_std/logq及PDMS优势detach；更新只重新计算固定z的current joint logq，不通过生成器或重参数化路径传播PPO梯度。完整joint PPO clip0.2、prior KL系数0.01为本路线设置，不能与旧dimension-mean clip0.005混为同一种尺度。
- 独立记录q_current到永久N(0,I)的joint KL与到本轮old q的joint KL，并另列每维平均。这里没有声称最终物理轨迹的精确density/KL，也没有逐场景安全保证。
- 噪声MLP始终逐scene batch1执行，采样和重算布局一致。实际采样参数与重算核对，避免只用同一条重算路径证明ratio1。
- 原始CPU scene-bound噪声可直接传入；训练root RNG和全局CPU/CUDA RNG分别保存。Qwen每scene只前向一次，候选按共同chunk4运行原ODE。

## 受控矩阵与部署

L0为原先验；L0-S仅比较登记的std0.7/1/1.3，选择仅用dev_select、成本全计。L1条件高斯＋prior KL，L2全局高斯＋相同KL，L3条件高斯无prior KL。首轮各在线学习臂B16/G16、64 fresh rounds、四次inner，即16384真实新候选及256实际optimizer更新；共同profile和预算已锁定，三个学习臂已由主agent按统一源码快照实际启动。

Adam峰值LR1e-4，按共同短窗采用3% warmup（256更新对应8）后constant。优势显式使用公共FP64组中心化／fresh raw共同std（floor0.01），不私下沿用旧逐组std或最终whitening。无成本乘子、搜索标签或轨迹replay混入本路线。

主读出为q采样单候选；mu-root确定性读出单独命名。后者不是平均轨迹，不把同一确定性输出重复五次解释为五个独立随机样本。任何候选库oracle只用于发现能力，不能充当部署分数。

## 已执行的资格

[CPU算术v2](cpu_qualification_v2.json)通过8项检查：完整Gaussian密度/解析KL、mean与sigma分别改变ratio、padding、零初始化、常数组、独立score-function方向、四次真实Adam与inner恢复、真实tiny source ActionDiT包装输出及参数/buffers冻结。此处tiny模型和二次reward仅验证算术，不是驾驶结果。

[实际CLI恢复v2](cpu_CLI_resume_v2.json)调用真正latent_train入口，两个round、八次真实Adam更新；inner1发布后注入中断，再通过CLI --resume继续，policy、Adam、scheduler、自定义root RNG、后续behavior batch和有效history严格一致。COLLECTING状态拒绝发布，latest不推进。评分器为明确标注的CPU mock；[原资格记录](cpu_CLI_resume.json)保留。

[STOP与恢复v3](cpu_CLI_resume_stop_v3.json)进一步实际执行了run/STOP_AFTER_BOUNDARY和SIGUSR1两种停止路径：仅在inner1已发布的合法边界退出，保留原old batch，恢复后八次更新与连续运行严格一致。SIGTERM使用相同只置flag的handler；不会在信号处理函数中触发CUDA、保存或随机抽样。收集期间的停止延至完整collection提交，不把半成品当可恢复rollout。

[真实完整模型profile](real_profile_L1_v1.json)已由主agent在GPU1集中执行：32个TRAIN场景、G4、128条真实canonical评分，22/32组有PG；四次真实小策略更新，初始ratio精确1，VLM/provider/DiT全部参数与named_buffers hash不变。峰值16,616,440,320 bytes（约15.48GiB）。总137.61秒，condition14.47秒、ODE生成27.04秒、评分7.46秒、噪声策略更新0.367秒；冻结hash另用28.83秒。第4个inner联合prior/old KL约0.002，ratio约0.8219—1.2887，clip比例0.0078。该资格并不证明驾驶提分，资格权重禁止作为正式初始化。

## 入口与恢复

均为真实存在的入口（主agent统一GPU路由，不重复启动）：

```bash
CUDA_VISIBLE_DEVICES='' /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.qwen_flow_exploration.latent_validate \
  --output reports/qwen_flow_three_routes_20261024/latent/cpu_qualification_v2.json

CUDA_VISIBLE_DEVICES='' /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.qwen_flow_exploration.latent_resume_validate \
  --output reports/qwen_flow_three_routes_20261024/latent/cpu_CLI_resume_v2.json
```

正式训练入口为`latent_train --protocol ... --output ... --arm L1|L2|L3 --seed 42`；调用公共`require_budget_lock`核对sealed BUDGET_LOCK与共同protocol SHA，只接受共同锁定预算。资格缩短窗口必须使用独立输出加`--qualification --limit N`，不能成为正式初始化。恢复增加`--resume <原run或合法checkpoint>`，保留同一output、arm、seed、protocol和源码身份；pending inner携带原roots/old logq/优势，不能重新生成。

评估入口为`latent_evaluate --protocol ... --output ... --arm ... --checkpoint ... --split DEV_SELECT --readout sample`。L0/L0-S不得加载学习checkpoint，L0-S显式`--scale 0.7|1|1.3`。默认采用预登记诊断人口，`--full-split`才是完整split；dev_audit受公共封存门禁控制。未完成记录不能从分母中消失。

终点统一CPU分析入口已实现并执行过未完成门禁：

```bash
CUDA_VISIBLE_DEVICES='' /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.qwen_flow_exploration.latent_analyze \
  --output /mnt/project/qwen-flow-three-routes-20261024/analysis/latent_terminal_v1
```

该命令必须等待L1/L2/L3的DEV随机、TRAIN随机、DEV均值读出及对应L0参考全部封存；缺失时输出`WAITING_FOR_SEALED_EVALUATIONS`且不创建分析结果。固定256更新终点、readout与seed集合有校验。所有配对及bootstrap调用公共分析器，额外按scene一次汇总真实root q的联合prior KL与std；已在现有L0/L0-S完整记录验证KL为0及2.44019865，后者与24维高斯解析式一致。此处root KL不是最终轨迹KL，也不将五次部署draw当K32搜索曲线。

终点薄入口最初只执行了缺失门禁，root审阅发现完整分支把`population/reference`误写为`population[reference]`。此分析编排错误已修复，不影响训练、轨迹或之前通过公共CLI取得的尺度对照。新增[完整分支验收](terminal_analysis_complete_case_v2.json)：14份完整sealed CPU合成评测实际执行全部13项比较、CSV发布和root诊断；另通过同一`write_pair`实际重算已有L0-S/L0各1280输出，结果严格一致。合成算术记录不属于驾驶结果。

## 研究判断进度

第一份已完成的部署对照见[固定尺度0.7报告](fixed_scale_control_07.md)：DEV_SELECT预登记256场景/8logs、五个评价噪声种子，L0-S为92.180332，L0为92.143105，净差+0.037228点，log-bootstrap区间[−0.103974,+0.181690]。其余固定尺度和学习臂未完成前不选择温度，不将此开发集92分写成Navtest达到92。

共同分析器[CPU资格](common_analysis_cpu_v4.json)通过12项检查，并已实际读取两臂全部各1280条记录。兼容latent与merged-adapter的readout别名时，会核对原生噪声namespace、CPU FP32、10步shift5、3维动作、source base及固定seed集合，还逐条复现实际root创新；不同namespace的早期feedback自动评测只作辅助结果。

L1/L2/L3均已完成64round/256update，每臂16384条真实新候选；[终点与条件性诊断](training_and_conditioning_diagnosis.md)核对了实际records、Adam/scheduler、四inner固定behavior、源码与冻结生成器hash。

原版完整13项终点比较已封存，详见[完整初筛研究判断](terminal_results_v1.md)及[aggregate](terminal_results_v1.json)。primary DEV随机单轨：L1为92.475394（相对L0 +0.332290点，log CI95[+0.262152,+0.397236]），L2为92.147917（+0.004812），L3为92.449639（+0.306534）。均为预登记256场景/8logs、五固定推理seeds，不能称Navtest达到92。TRAIN512的L1/L3增量仅+0.096243/+0.142066，区间均跨零，且新增NC/TTC违规；不是全面安全保持。固定std1.3已得DEV+0.181330，L1相对它的+0.150959区间跨零。原版研究标签为“条件支持”，限于冻结生成器root分布调整的小幅单轨证据。

但[全部256场景的参数记录](primary_DEV_conditionality_v1.json)确认L1/L3的root Gaussian几乎不随scene变化，首层tanh发生学到的饱和。L2在同LR下的分布位移又远小于L1，因此这些结果支持“冻结生成器上学习root分布可以产生小幅部署收益”，不支持“已学会有价值的场景条件化”，也不能据此宣布prior或条件化机制独立成立。

唯一修正[L1_CENTER_V1](CENTER_V1.md)只加入固定TRAIN均值中心化。CPU同输入/概率/梯度、默认旧类等价、B1布局和inner恢复已通过；随后真实GPU资格完成256条评分、4次更新且frozen hash不变。主agent在2026-10-24 20:53 UTC按独立批准锁、immutable7ec40a7启动fresh q0正式64round/256update，未复用资格θ；[运行索引](CENTER_execution_index_v1.json)绑定实际命令和三个待终点评测。它不热改或覆盖已有97/9ecd结果，也没有新增其他修正/seed/控制矩阵。此为启动快照，四步资格不等于PDMS提升。
