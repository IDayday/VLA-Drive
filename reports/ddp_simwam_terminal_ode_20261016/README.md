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
校验不放宽。现有训练和评价 controller 共用有限 follow-up 注册入口，
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

正式派发仍必须通过真实八卡学习/中断恢复及零/非零 LoRA 的实际采样
记录验收；CPU PASS 不代替驾驶效果。验收权重不作为正式初始化。
资格验收、失败重算及评价成本与20k科学预算分别计数。

## 实际入口

以下命令使用已激活的项目 Python，在本任务源码目录执行；GPU 命令
必须位于当前授权且容量合格的 Tsinghua 主机，不能直接在管理机启动。
输出目录保持独立，已存在的验收目录不能覆盖。

```bash
python -m scripts.analysis.simwam_recipe.qualify_terminal_ode \
  --output /mnt/project/simwam-terminal-ode-20261016/qualification/cpu_contract.json
python -m scripts.analysis.simwam_recipe.qualify_recovery_run \
  --name eight_gpu_window_recovery_v1 \
  --root /mnt/project/simwam-terminal-ode-20261016 \
  --config configs/flow_rl/simwam_canonical_last_ode_ddp.yaml
```

正式派发入口仍是 `scripts.analysis.simwam_recipe.dispatch_formal`，变体名
`R_CANONICAL_LAST_ODE`。评价入口仍是 `watch_evaluations` 和
`final_navtest serve`，使用 `--followup-last-ode` 及明确主机参数。
恢复命令必须指向同一不可变源码和通过阶段合法性校验的原生 checkpoint；
合并权重只用于部署，不可作为 exact resume。
