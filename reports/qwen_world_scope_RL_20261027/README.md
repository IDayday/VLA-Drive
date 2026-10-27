# Qwen-World IL40k：memory与时间调制的两组RL

用户2026-10-27明确授权新增两组，原动作attention-LoRA组继续作为对照。
两组都从同一完整World IL40k重新开始，不能使用对照组中期权重或资格验收权重。
父模型native model hash `f0c41d5ebb29ab3db59841fd96ea8d8ec4a804120ff4e474763e01ffe46027f4`；
已观察Navtest **87.972382**，如实登记用户指定父模型，不能称盲选。

| 组 | 训练范围 | 张量/参数 | 峰值LR |
|---|---|---:|---|
| 已有对照 | 160个attention投影LoRA，r16/alpha32 | 320 / 5242880 | 5e-5 |
| ATTN_LORA_MEMORY | 同一attention LoRA＋完整scene/predictive providers | 458 / 52707840 | attention5e-5，memory5e-6 |
| TIME_MOD_LORA | Time MLP＋20个block AdaLN投影＋末端AdaLN投影的LoRA | 46 / 4816896 | 5e-5 |

全部Qwen保持冻结。两组的G8/K10/shift5、随机[7,8,9]、source SDE/relative clamp、
global raw reward std、PPO clip0.005、beta50三维mean anchor、inner4、AdamW和有效clip1保持一致。
train42/eval2101；canonical v1.1 5:5:2；无GT replay/BC/critic/dual/新噪声模块。
每组50k optimizer updates，12500 fresh全局rollout、100000场景曝光、800000新候选评分。
同一World物理log隔离TRAIN公开hard-list交集15171场景，不含DEV/Navtest。

## Memory梯度与参考合同

- Qwen每scene一次前向，仅缓存其冻结的selected8/16/28、merger和文本特征。
- current scene＋predictive memory注册在actor中，每个inner重新建立有梯度的memory与KV。
  current路径没有`detach/no_grad`，同一次概率重算的3个随机步共用同一个memory计算图。
- 永久reference是原IL40k自己的冻结memory与冻结actor，不能读取current学习后的memory冒充固定参考。
- rollout/old density/奖励/优势固定，四次inner不刷新；行为版本hash包含全部可训练memory权重。
- 原生checkpoint的`adapters`字段在本组保存全部LoRA＋memory可训练张量，名称闭包严格核查。
  导出加载完整更新后的memory，不允许只导出LoRA丢失memory收益；部署仍无teacher/scorer/cache依赖。

## 时间组开训前分析

完整模块/公式/适配边界见 [time_module_transfer.md](time_module_transfer.md)。
本组是**时间模块targeted LoRA**，不叠加attention LoRA或memory更新。
采用论文的rank32；alpha32作为本项目明确工程设置。相较现有r16 attention对照，
target、rank与缩放存在差别，且其对照没有FFN LoRA，不能宣称逐项复现论文比较。

## 验收与启动

沿用同一个已验证trainer和原生事务/恢复框架，仅新增明确scope runtime与参数组接口。
新旧相关的CPU资格测试共61项PASS，包括：

- LoRA与显式FP64线性代数输出/输入/A/B梯度一致；正确的Time/AdaLN路径进入实际forward。
- 初始ODE逐值保留、采样与重算一致、合法梯度闭包；current memory有梯度，固定reference不漂移。
- 冻结Qwen只prefill一次，memory每inner重算；CPU中断inner2后恢复与连续执行的全部参数及Adam逐值一致。
- 非零memory/LoRA部署载荷完整加载；旧attention-only部署与训练合同回归通过。

正式启动前，每组还必须通过真实八卡：6次更新 → 保存inner2 → 原checkpoint恢复至16次更新。
检查32个scene/256新候选、相同链继续inner3/4、全部冻结hash、真实Adam任务动量、
每步采样mean/std、原IL初始完整ODE，以及source第一inner ratio。
验收权重不能初始化正式组，正式组重新从IL40k和新LoRA启动。

启动入口：

```bash
python -m tools.qwen_world_postcot.start_scope_rl --registration /absolute/scope/REGISTRATION.json
```

登记根：`/mnt/project/qwen-world-postcot-20261025/scope_rl_IL40k_20261027`，
各组独立owner/资格/正式run/原子checkpoint与失败证据；只用固定白名单空闲卡，不停止原对照。
所有CLI都已实际执行help smoke；真实启动状态与资格结果由后续运行收据记录，不能用本段代替已运行证据。

固定保存0/5k/10k/20k/35k/50k，恢复点每100更新；主评价单候选10步shift5 ODE、seed2101、
canonical v1.1全Navtest，报告相对IL和原LoRA对照的PDMS、NC/DAC/TTC、G+/G−与零分转移。
训练reward、资格batch、DEV或候选最优值都不替代Navtest有效性结论。
