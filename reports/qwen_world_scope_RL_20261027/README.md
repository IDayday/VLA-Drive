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

两组已实际通过真实八卡：6次更新 → 保存inner2 → 原checkpoint恢复至16次更新。
每组32个scene/256新候选；相同链继续inner3/4，初始完整ODE与IL逐值相同，
7/8/9步实际采样mean/std与重算逐值相同，Qwen和永久reference的冻结hash不变。
Memory组全部320个动作LoRA张量和138个memory张量、Time组全部46个LoRA张量
都有非零Adam任务一阶动量。峰值每卡分别14.816和14.531 GiB。
Memory资格从zt4的inner2保存点移到zt6继续，保持原链、old概率、优势、Adam及RNG。
正式两组均重新从原IL40k和相同新LoRA初始化；逐张量核对初始载荷，B全零、Adam为空，
未使用资格验收结束权重。证据见 [formal_start.json](formal_start.json)。

2026-10-27 14:46 UTC启动审计：Memory八rank共同完成192次正式更新、合法恢复点100；
Time共同完成596次更新、合法恢复点500。两组第一inner ratio误差均为0，已记录更新梯度有限。
这些是启动快照，不代表后续实时状态，也不能据此判断PDMS收益。

启动入口：

```bash
python -m tools.qwen_world_postcot.start_scope_rl --registration /absolute/scope/REGISTRATION.json
```

登记根：`/mnt/project/qwen-world-postcot-20261025/scope_rl_IL40k_20261027`，
各组独立owner/资格/正式run/原子checkpoint与失败证据，不停止原attention-only对照。
原zt4与本地的下一阶段曾被新出现的其它任务占用拦截，原生门禁明确记录
`REFUSED_RESOURCE/child_not_started=true`；未抢占、未丢弃合法评分、未把它解释为算法失败。
两组随后在用户新授权的zt6八卡共享启动，任务启动成功后用户又临时授权tsinghua-zt2与rl-zt2，
因此安排Memory保存到合法inner边界再迁至空闲tsinghua-zt2，Time继续zt6，消除两组之间的GPU争用。
rl-zt2留作后续已登记评价容量，不为填满资源新增方法或seed。

实际训练源码始终为不可变`367bffd27aff5785fce6352481b89e2a433a9ee1`。
CPU调度器后续`057452d`/`7c4d29f`只允许凭已封存收据继续资源拒绝后的资格/正式启动；
actor、采样、概率、loss、optimizer与数据未改变，运行中的源码快照没有热修改。
资源授权为任务级可撤销临时授权，未扩大其它项目的权限，也未重开旧九小时租约或暂停的实验。

14:56 UTC资源重新分配已完成：Memory在391/inner3保存后于tsinghua-zt2完成同一旧链的392/inner4，
随后393/inner1进入新behavior，八rank日志无缺口/重复，新的第一inner ratio误差仍为0。
此时Memory共同完成419更新/105份fresh rollout/6720新候选，Time完成1287更新/
322份fresh rollout/20608新候选；两组各用独立八卡。
详细恢复边界、实际命令和owner路径见 [resource_reassignment.md](resource_reassignment.md)，
机器可读快照见 [resource_assignment.json](resource_assignment.json)。

固定保存0/5k/10k/20k/35k/50k，恢复点每100更新；主评价单候选10步shift5 ODE、seed2101、
canonical v1.1全Navtest，报告相对IL和原LoRA对照的PDMS、NC/DAC/TTC、G+/G−与零分转移。
训练reward、资格batch、DEV或候选最优值都不替代Navtest有效性结论。
