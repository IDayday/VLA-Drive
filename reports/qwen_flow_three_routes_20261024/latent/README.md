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

L0为原先验；L0-S仅比较登记的std0.7/1/1.3，选择仅用dev_select、成本全计。L1条件高斯＋prior KL，L2全局高斯＋相同KL，L3条件高斯无prior KL。首轮各在线学习臂B16/G16、64 fresh rounds、四次inner，即16384真实新候选及256实际optimizer更新；formal必须等待共同profile预算锁定。

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

## 研究判断进度

目前已证明接口与数值链可用，尚需完成同预算L1/L2/L3、固定尺度对照及新噪声单轨评价，才能判断条件化是否有独立价值。现阶段不因短资格的loss/候选表现而宣称路线成立，也不把尚未完成称为当前设计无效。
