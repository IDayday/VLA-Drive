# DDP SimWAM：末步 ODE 单因素对照

本轮沿用正在执行的强化学习改进目标，登记一个有限对照
`R_CANONICAL_LAST_ODE`。对照是已完成的 `R_CANONICAL`；从同一 F-SFT
重新初始化，同一 LoRA 初始化、困难 TRAIN 清单与数据顺序、seed42。
不叠加 X1 的 anchor 修改。X1 的既定评价独立继续。

唯一科学配置差异：`grpo.sample.sde_exclude_last_step: false → true`。
复用锁定 SimWAM 已有实现，三个随机步骤从 `[7,8,9]` 前移到 `[6,7,8]`，
第 9 步走源 scheduler 的 Euler ODE。它不是保存链反事实中的
score-corrected SDE mean，也不是只删除同一条链中的一个 epsilon。
该变化还改变了一个更早的随机状态与被优化转移的位置。

动机来自固定 TRAIN64 的已保存链诊断：末次创新显著损害该有限候选集的
均值、安全比例及请求轨迹平滑性。诊断没有训练学生，也没有证明原 ODE
部署会提升；具体数据与限制见[多来源分析](../ddp_simwam_x_anchor_20261015/flow_design_review/multi_source_analysis_20261016.md)。

## 固定合同

- 20,000 AdamW updates；每份 fresh rollout 四次 inner；5,000 份全局 rollout、
  40,000 次场景曝光、320,000 条新候选；训练八卡，每卡一 scene、G8。
- K10、shift5、源 rigorous SDE、std 下限、创新及 relative logprob clamp
  全部不改。只对真实随机步骤 `[6,7,8]` 计算源 dimension-mean 概率代理。
- canonical 5:5:2 奖励；全局 raw reward std；原源 rank 归约；anchor beta50、
  `[.2,1,.5,.5]`；无 BC、GT replay、dual 或 learned-noise。
- 原 attention LoRA r16/alpha32，192 张量；上游及 qwen_proj 冻结。
  LR5e-5、warmup100、constant、源有效 gradient clipping=0 不改。
- 永久 SFT reference、保存链四次复用、恢复事务、固定 DDP bucket 均沿用。
  共享 LoRA 也影响确定性前后缀，但 loss 仍只优化保存的随机转移；
  不宣称这是整条混合链的无偏梯度。
- 5k/10k/15k/20k：完整 DEV1696 与固定 TRAIN512，均分别评价
  E_NATIVE/E_SHIFT5；最终固定20k、Navtest12146、seed2101，不选 best-of-two。
  已封存的 SFT 和 canonical 控制结果复用。历史 Navtest 非完全盲测。

完整参数、预算、终点规则及授权边界见[registration.json](registration.json)。
目标仍是原单候选 E_NATIVE canonical PDMS 达到92以上，并如实评估
NC/DAC/TTC 新违规、恢复、SFT-safe 退化与 G+/G−；不是保证训练结果。

## 实现与验收

源 sampler、概率、loss、actor 和 optimizer 实现均未修改。新增完整配置
校验，只允许上述一个布尔值和变体名变化；原 SOURCE/CANONICAL/X1
校验不放宽。collector 与恢复边界按同一变体表验证窗口：历史变体仍
严格要求 `[7,8,9]`，新变体严格要求 `[6,7,8]`。现有训练和评价 controller 共用有限 follow-up 注册入口，
保留旧 CLI 别名，不修改运行中的不可变快照。

[CPU 验收](cpu_contract.json)直接抽取锁定上游类方法进行52项对照，覆盖
3/4维、B8/16、完整11状态、随机窗口、RNG、实际采样 mean/std、last-step
Euler、概率重算、loss 与 LoRA 梯度。旧/current ratio=1；末端确定性
状态没有 Gaussian 概率；10种未登记配置突变被拒绝。

最初测试把保存链的 strided slice 当作采样期间的 contiguous x，要求
跨布局逐位相同，观察到1.19e-7的 FP32 误差。测试改为按真实 live layout
验证末步 Euler；采样与保存状态重算保留已有1e-6容差。未修改 kernel，
没有正式训练或真实评分受到影响。失败记录保留于
[cpu_harness_failure.json](cpu_harness_failure.json)。

首次八卡验收发现 collector 中仍有历史窗口的硬编码断言，在所有 rank
评分和更新前拒绝新窗口。原生 `latest` 保持在合法 READY/update0，失败
仅写 evidence，没有把未完成链发布为恢复点。已补齐 collector 和恢复
校验的变体映射；新增四个变体的正确/错误窗口边界测试均通过。
[失败与实际成本记录](eight_gpu_v1_failure.json)保留；后续使用新 attempt
完成真实八卡验收，没有将第一次失败目录改写成成功。

真实八卡第二次验收已通过：连续路径与 inner1 后恢复路径均达到8更新，
所有 rank 的 LoRA、Adam、scheduler、RNG、后续 fresh chain、old probability、
优势、loss、梯度范数及 LR 完全一致。192个 LoRA 张量均更新，全部192个
Adam 一阶/二阶梯度矩均非零；冻结参数哈希不变。只读核对 gradient moments
避免把 AdamW weight decay 引起的参数变化误判为任务梯度。

零/非零 LoRA 的真实 DDP 采样记录核验通过：实际 `[6,7,8]` 三步的
mean 最大误差2.3842e-7，std 最大误差1.5293e-9；记录没有改变 RNG 或链。
这些误差与源 FP32 布局合同一致，不通过放宽 PPO clip 解决。

- [八卡恢复](eight_gpu_recovery.json)
- [真实学习与验收成本](real_learning_and_cost.json)
- [192个梯度状态](optimizer_gradient_evidence.json)
- [实际采样与重算](sampling_recompute.json)

两条资格验收路径实际共16次 optimizer updates、256次候选评分，约178.52秒
job 跨度、1317.43秒累计 worker elapsed（含加载）。失败 v1 的约280.53秒
累计 worker elapsed 另列。共享 worker 时间不是独占 GPU 计算时间。
验收权重不作为正式初始化；这仍不是驾驶效果验收。

## 正式启动与早期证据

正式源码锁定 `8b4dda71ea299e3eef53a8c0b613385d752b2907`，独立 run 为
`R_CANONICAL_LAST_ODE_seed42_v1`，训练位于当前授权的 tsinghua-zt2 八卡，
共享现有资源且无抢占。阶段评价和最终固定20k评价 controller 已启动，
分别有自己的单 owner 锁。运行记录见[formal_launch.json](formal_launch.json)。
后续文档/只读分析提交不改变这份训练快照。

2026-10-16 12:12 UTC 实际核查：launcher的PID、start ticks与完整命令
匹配；两名行政controller也匹配单owner记录。合法checkpoint为800更新，
八rank共同完整日志边界为812更新。下列440更新profile保留为更早的计时证据。

最新发布的[训练快照](initial_training_profile_v3.json)已检查八个 rank 的
共同完成边界：440 updates、110 fresh rollouts、880次场景曝光、7040条
候选；98.18%的 scene group 有 PG 信号。按最近已封存 checkpoint 的同一
invocation 计时线性外推，剩余训练约7.78小时，不含评价，也不是共享负载下
的完成时间保证。这是时间点快照，不代表读者阅读时的即时进度。

固定最初100次更新的比较使用相同200场景、各1600条候选，原始输入顺序、
初始 LoRA、第一批初始噪声及共同六步 ODE 前缀均核对一致。没有增加评分：

| TRAIN 随机策略读出 | R_CANONICAL | LAST_ODE |
|---|---:|---:|
| canonical PDMS | 78.9342 | 82.5614 |
| NC | 99.7500 | 100.0000 |
| DAC | 92.6250 | 96.2500 |
| TTC | 97.8125 | 99.1250 |
| EP | 62.3922 | 64.4723 |
| NC/DAC/TTC 联合通过率 | 90.5000 | 95.3750 |
| 有 PG 的 group 比例 | 98.5000 | 97.0000 |

这是早期候选质量的正信号，不能当作单候选 ODE、DEV 或 Navtest 提升。
PG 梯度量级也降低，尚需验证更干净的候选能否转化为充分更新和部署收益；
不据此调大 LR、减小 anchor 或更换预算。[完整匹配记录](matched_first100.json)

同时，[X1 全部阶段评价](parent_X1_stages.json)已完成：20k DEV 原 ODE
94.5538，对照94.6719；TTC99.9410，对照99.1156；EP87.2830，对照88.3917。
TRAIN诊断原ODE为86.4722，对照85.6523。它表现为不同的安全/进度权衡，
不能仅根据容易的 DEV TTC 宣布解决 Navtest 安全问题。X1 固定20k Navtest
已在另外两台授权主机启动，完整结果尚未产生。

后续12:31 UTC更新：X1完整Navtest已经完成，原ODE89.3817，相对SFT
+0.4701分、相对Canonical+0.4300分；相对SFT仍有NC/TTC净退化。
见[最终报告](../ddp_simwam_x_anchor_20261015/final_navtest_20261016/README.md)。
四阶段[保存batch梯度诊断](X1_saved_gradient/README.md)进一步确认方向保持
改变了行为，但不把单batch冲突当全程结论。LAST_ODE配置与预算不变。

## 成本汇总更正

本次只读 profile 暴露了旧成本汇总将 `RUNNING` 和 `ABORTED` 合并的问题。
训练没有停止或失败。schema3 将未关闭记录单列，不能仅凭状态文件推断
进程存活；失败时间只计明确 `ABORTED`，累计时间补入已有 preparation
字段。运行中的吞吐仅使用同一 invocation 的已封存 checkpoint 窗口，
以 `checkpoint_update − invocation_start_update` 为分子；不足100更新不外推。

[7项回归验收](cost_window_tests.json)覆盖恢复后计数、旧记录、短窗口、
失败和完成状态。原错误派生账本保留在[更正前快照](previous_cost_classification.json)，
原私有文件也保留。新旧快照时间不同，不把其总成本之差全部归因于补丁。
此修改仅用于分析，不热改正式8b4dda7训练或正在执行的X1评价代码。

## 实际入口

以下命令使用已激活的项目 Python，在本任务源码目录执行；GPU 命令
必须位于当前授权且容量合格的 Tsinghua 主机，不能直接在管理机启动。
输出目录保持独立，已存在的验收目录不能覆盖。

```bash
python -m scripts.analysis.simwam_recipe.qualify_terminal_ode \
  --output /mnt/project/simwam-terminal-ode-20261016/qualification/cpu_contract.json
python -m scripts.analysis.simwam_recipe.qualify_recovery_run \
  --name eight_gpu_window_recovery_v2 \
  --root /mnt/project/simwam-terminal-ode-20261016 \
  --config configs/flow_rl/simwam_canonical_last_ode_ddp.yaml
```

正式派发入口仍是 `scripts.analysis.simwam_recipe.dispatch_formal`，变体名
`R_CANONICAL_LAST_ODE`。评价入口仍是 `watch_evaluations` 和
`final_navtest serve`，使用 `--followup-last-ode` 及明确主机参数。
恢复命令必须指向同一不可变源码和通过阶段合法性校验的原生 checkpoint；
合并权重只用于部署，不可作为 exact resume。

新增只读分析入口已实际执行；再次运行须指定新输出文件，保留旧摘要：

```bash
python -m scripts.analysis.simwam_recipe.compare_training_prefix \
  --control /mnt/project/simwam-recipe-20261015/runs/R_CANONICAL_seed42_v1 \
  --candidate /mnt/project/simwam-terminal-ode-20261016/runs/R_CANONICAL_LAST_ODE_seed42_v1 \
  --updates 100 --output "$NEW_PREFIX_SUMMARY"
python -m scripts.analysis.simwam_recipe.summarize_training \
  --run /mnt/project/simwam-terminal-ode-20261016/runs/R_CANONICAL_LAST_ODE_seed42_v1 \
  --window-updates 100 --output "$NEW_TRAINING_SUMMARY"
python -m scripts.analysis.simwam_recipe.qualify_cost_window \
  --output "$NEW_COST_CHECK"
```

这些变量表示尚未存在的新文件；不是任意终端已预设的路径。分析不触发
新的模型采样或评分，也不改变运行中的immutable源码。
