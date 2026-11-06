# 官方AUG与D46迁移：区别、已发现的问题及修正

更新：2026-11-06。官方比较固定源码`edf3422304e4b2987e839a9212cf1176782e7380`。我们的训练父模型始终为完整D IL46 SHA `386a0a27bd197cb002501bcb90c343659001d2e6d4f7bad0cbd5e31a91f8477d`。官方模型诊断使用其独立SFT；不向D转移官方RL权重或Adam。

## 结论与证据强度

1. **确定的实现／分析偏差**：此前官方对照按通用verl样本std理解，实际发布Stage-I启用split分支，使用NumPy总体std、方差门限，不在有效分母加epsilon；官方诊断和反事实工具也有此偏差。本次修正，并直接执行固定官方函数逐值对照。
2. **确定的算法差异，尚未证明是退化原因**：D加入SafetyEP、全局标准化与cap5，缺少官方倒退门控。这改变奖励目标和场景权重，不只是优化实现。新独立AUG9恢复该动作奖励路径，实际部署效果须等完整评测。
3. **已验证的数值问题**：BF16文本/动作计算曾把微小参数更新变为较大输出跳变；FP32已修正真实TRAIN局部数值连续性。AUG7满3E提高0.24360，但不能把全部收益归因于此修正，尚未超过D历史最佳。
4. **已实测不支持的简单改法**：KL从0.1降为官方0.001，AUG8 0.5E新增74零分、修复41，安全恶化。不能继续把参考约束越小越好作为结论；也不能用0.5E断言其最终上限。
5. **尚未补齐的系统差异**：历史视频、主干与预测CoT训练、全参数更新、训练部署步数均不同。公式正确不等于完整迁移，不能把作者94.0/94.6归于当前D动作loss。

## 实际执行配置对照

| 项目 | 官方发布Stage I | D AUG7／AUG8 | 本次AUG9 |
|---|---|---|---|
| 初始化 | 官方SFT best，新RL optimizer | 同一D IL46，新Adam0 | 同一D IL46，新Adam0 |
| 观测 | 4历史帧、2Hz，三相机裁剪拼接2048×512；历史ego轨迹文本 | 当前帧三视图及ego/nav | 不变 |
| 主干 | Qwen3.5-2B，原生Transfusion时间调制 | Qwen3VL-2B，D自身预测／动作结构 | 不变 |
| 可训练参数 | 文本主干及动作机制，视觉冻结 | 7,010,307参数：text attention LoRA、time FiLM/LoRA、动作接口与位置 | 不变；不是仅动作头 |
| 采样 | 3步linear JaT SDE，独立随机根；噪声[.05,.05,.02] | 部署根；K10 shift5前7步ODE＋末3步JaT；TRAIN物理尺度校准 | 不变 |
| 部署 | SFT原生1步ODE | 原生K10 shift5，seed2101，单条 | 不变 |
| 动作奖励 | planning=5、PDMS=1、其它planning项0；reverse_gate=true | SafetyEP优先安全，安全组比较EP；官方PDMS仍用于评测 | 5×PDMS，倒退门控 |
| 优势 | `rl_split_advantage=true`；每组population std，无额外全局白化 | SafetyEP组内中心化，全B×G population std，cap5 | 逐组population std，std<1e-6为0，无分母epsilon或cap |
| KL | action0.001；CoT0.01 | action0.1／0.001 | action0.1：明确保留D适配，不称官方全配方 |
| CoT | prediction reward及token GRPO，独立优势 | 无CoT RL | 仍无；未伪造预测文本标签 |
| B/G/M | B32/G8；ppo mini-batch32；实际循环以原driver为准 | B64/G16/M2 | B64/G16/M2 |
| LR | YAML actor1e-6；其它调度继承原verl配置 | 1e-6；曝光warmup.08E/hold.60E/cosine4E/floor.05 | 不变 |
| 数值 | BF16路径／FSDP混合精度 | 文本动作FP32、冻结视觉BF16，AdamW(.9,.95) | 不变 |
| 训练数据 | 官方HF缓存；docs报告103287，精确排除token待核 | 经审计合法NAVTRAIN103288；与NAVTEST token/log隔离 | 不变 |
| 预算 | 公开Stage I 10 epochs | AUG7完成3E；AUG8在训 | 先登记3E，按效果扩展 |

公开奖励`reverse_gate`将轨迹从ego原点开始任一小于−1cm的x位移识别为倒退；触发后整个planning/action奖励为0。该bool门控没有GT倒退豁免，另一旧阈值门控才有；没有混淆两者。官方评分器记录仍保持原样，训练门控不修改canonical PDMS。

```text
官方：4帧 + SFT模型 → 3步JaT候选 → PDM + 倒退门控 → 动作组优势 → 动作JaT更新
                         └→ prediction-CoT → 预测/格式奖励 → CoT组优势 → token更新

D旧：当前帧 + D IL46 → K10末3步JaT → SafetyEP → 全batch std → 动作JaT更新
D新：当前帧 + D IL46 → 同一采样链 → PDM + 倒退门控 → 每组std → 动作JaT更新
```

## 为什么旧路线收益较弱

AUG7完成309864个有效场景曝光，9684次Adam；完整NAVTEST89.75409134。相同FP32u0提高0.24360，44-log配对95%区间[+0.11540,+0.36294]，但高分≥0.9群体贡献−0.06501；零分修复30、新增19。NC/TTC退化与EP改善并存，不能说只要延长就会达到92。AUG7后期仍有增益，未据此宣判长期上限；先验证学习信号差异更有信息价值。

AUG8 0.5E89.49181220，相同u0−0.01868，区间[−0.30627,+0.24500]；新增碰撞失败34、修复9，TTC新增失败97、修复23。它与AUG7的rank数不同，非逐采样配对对照；但该路线已实测到的安全退化不能被训练reward提高掩盖。

完整AUG6反馈审计8256不同TRAIN场景中，159组canonical PDMS有差异而SafetyEP无信号，54全零组里51组无SafetyEP信号且存在真实时间风险差异。此前约92.4%优势平方权重集中于仅10.39%的安全混合组。此为优势权重，不是参数梯度；原“官方PDMS组优势”统计实际用的是sample-std近似且未加倒退门控，现明确更正。新工具正扫描AUG7全3E实际反馈，分开记录曝光次数与唯一token，不以局部样本代替全程。

完整B64梯度回放u100参考KL/PG范数2.004、cos−0.924；u200为0.978、cos+0.649。参考项可以主导更新，但不是始终有害。既有梯度图中的角色范数在裁剪之后，不能推断Adam更新只发生在动作接口。

## 本次已完成与执行中

- 新模块`split_credit.py`复用原ActionCodec，从真实链终点取物理轨迹；official scorer不变。新增AUG9独立identity，继承原runtime与恢复系统。
- 固定源码函数直接执行对照：G8/G16/G32共1545组，1024条轨迹，归一化与倒退结果最大误差均0；25项定向测试通过，含padding、协议、原生material、报告归档状态。
- 同步纠正官方单步诊断、TRAIN反事实和梯度审计的归一化，旧实验结果保持历史标记，不重写成新方法收益。
- AUG9两节点16卡真实第一inner更新→保存pending→原生恢复第二inner；通过才从IL46和Adam0启动3E正式训练。每100更新保存、最近10份＋u0/最佳/评测中固定；后台事件触发完整NAVTEST。
- 官方SFT另外重跑修正的单次action-only JaT更新。此前原模型已能完成更新；仍不是B32联合CoT Stage-I复现，不报官方NAVTEST成绩。

## 下一阶段的决策依据

先看AUG9相同数值u0的完整分数、零分修复/新增、安全与EP，以及新规则在全量TRAIN上的有效组数。若改善，继续充分训练；若只提高EP而损伤安全，说明单纯恢复官方奖励仍不适合D，应优先改安全约束与部署轨迹吸收，而非继续盲目降低KL。

官方的时间信息、prediction-CoT和全文本更新是下一层需要隔离的系统差异。D没有可直接等价复用的官方AR预测头，不能给一个新辅助loss起CoT名字便声称复现。已取得真实P3 prediction标签，可用于独立原模型联合RL对照；原版SFT→RL曲线需单独建立。现阶段无证据认定LoRA容量不足或缺少CoT是唯一原因。

来源：[Stage-I配置](https://github.com/haha-yuki-haha/AUGUstus/blob/edf3422304e4b2987e839a9212cf1176782e7380/AUGUstus_GRPO/recipe/transfusion/config/stage1.yaml)、[split优势执行路径](https://github.com/haha-yuki-haha/AUGUstus/blob/edf3422304e4b2987e839a9212cf1176782e7380/AUGUstus_GRPO/recipe/transfusion/sync_driver.py)、[官方reward](https://github.com/haha-yuki-haha/AUGUstus/blob/edf3422304e4b2987e839a9212cf1176782e7380/AUGUstus_GRPO/recipe/transfusion/transfusion_reward.py)、[论文](https://arxiv.org/html/2610.16985v1)。论文数字是作者报告，不是本地复现。

## Checkpoint保留修复

审查发现后台旧规则只固定所有路线的全局最高点，而不是各路线的最高点。已改为每个独立实验保留best＋u0＋最近10份＋评测中权重，新增针对性测试通过。AUG8 u400的原生权重已被旧轮转规则删除，无法声称还能从该点原样恢复；完整分数和逐场景评测保留，当前AUG8最新原生状态完整。AUG7最高u9684仍在。该问题影响回退能力，不改变已报告分数。
