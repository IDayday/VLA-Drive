# 时间模块LoRA：开训前的来源、映射与准确性分析

来源：[The Low-Rank Structure of VLA Reinforcement Learning, v1](https://arxiv.org/html/2609.34599v1)，
2026-09-28，Minjae Oh、Yoonah Park、Jongwon Lim、Yohan Jo。本文研究时间条件参数的RL更新结构，
附录C.5比较Time MLP＋AdaRMS targeted LoRA与attention＋FFN LoRA，使用rank32。
这不是只把现有attention LoRA换个名称。已读论文没有给出可逐字节迁移的专用训练仓库链接；
本项目复用已锁定SimWAM原LoRALinear及GRPO算术，不编造论文官方实现来源。
论文HTML版本hash及实际LoRA文件来源见 [upstream_lock.json](upstream_lock.json)。

## 当前World模型上的精确映射

| 当前执行模块 | 输入→输出 | LoRA位置 |
|---|---|---|
| `time_embedding.0` | 1024维连续sigma正余弦 → 1024 | Linear本身 |
| `time_embedding.2` | SiLU后的1024 → 1024 | Linear本身 |
| `blocks.0..19.modulation.1` | 1024 → 6144 | 生成self-attn/FFN各自shift、scale、gate的Linear |
| `output_modulation.1` | 1024 → 2048 | 最后归一化的shift、scale投影 |

共23个Linear、46个A/B张量、4816896参数，r32、alpha32、dropout0。
用完整模块路径安装，不按child name `0/1/2`递归误包FFN。
ego MLP、action输入/输出、self/cross-attention、FFN、普通LayerNorm仿射参数、位置编码保持冻结。
LayerNorm本身不是被称为AdaNorm-LoRA的目标；目标是产生其动态调制参数的投影。

## 保留的数学语义

每个目标线性层为 `base(x)+(alpha/r)*B*A*x`，base冻结；A沿用源Kaiming初始化，B严格0。
输入梯度和A/B梯度与显式FP64公式验证；B=0时首步A梯度为0符合数学，之后需真实证明更新。
模型仍预测clean-x，转换一次`v=(state-pred_x0)/sigma`；连续sigma*1000编码、shift5网格、
SDE均值/方差、创新±5、逐维relative clamp7、H/D mean ratio与四次inner保持原配方。
这不是引入新的噪声策略，也不修改sigma日程。

## 必须披露的适配边界

1. 论文pi0.5使用AdaRMS；本模型是AdaLayerNorm，既有拓扑不改。
2. 本模型实际condition=`TimeMLP(sigma)+EgoMLP(current_ego)`，然后SiLU/投影。
   因此block/output的LoRA增量可随合法ego变化，不能称“只由时间决定的向量”。
   共享TimeMLP本身仍只读时间；ego MLP没有解冻。
3. 包含最终output AdaLN的投影，因为它实际执行条件调制；不能把它遗漏为普通decoder。
4. 论文Table11只在其机器人环境比较对应范围，本实验是World clean-x/真实NAVSIM上的迁移。
   r32与原attention对照r16不同，alpha32为明确工程选择；现有对照只含attention而非attention＋FFN。
   不能把未来差异全部归因于同参数预算的模块位置，不能宣称复现其成功率。
5. 标准GRPO仍只更新保存状态上的随机步骤代理。时间参数也影响确定性前缀，
   本轮不新增跨整个ODE前缀的反传，不声称完整混合链的无偏策略梯度。

## 开训判据

CPU已验证：目标路径准确、全部实际forward调用、零初始化ODE不变、logprob ratio一致、
输入/A/B梯度公式一致、冻结范围、非零LoRA载荷导出、完整参数与Adam的inner恢复等价。
真实八卡资格必须进一步验证：原IL完整ODE逐值一致、步骤7/8/9采样与概率重算mean/std一致、
所有预期参数有合法梯度/实际任务动量、Qwen/memory/reference hash不变，以及inner2恢复。
通过后才允许本scope的独立正式50k训练；资格成绩不作为规划效果证据。
