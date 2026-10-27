# Qwen-World IL40k → canonical RL

2026-10-27：用户明确指定 IL40k 为新 RL 父模型。其已完成的 canonical NAVSIM v1.1
Navtest 为 **87.972382 / 12146 scenes**，seed2101，单候选 clean-x shift5 ODE10。
这是已观察测试结果后的用户选择，不能称盲选，也不冒充100k IL完成后的DEV选择。

## 配方与模型

直接复用现有 QwenWorldPostCoT clean-x runtime：当前三视图Qwen → 144空间+16全局+
32预测memory → 20层/1024宽/16头 action expert → 8×3轨迹。
永久冻结 IL40k 的 Qwen、memory/provider、base actor 和 reference；只训练动作注意力LoRA。
训练不运行教师、DINO、未来标签、PostCoT readout或GT replay。

- R_CANONICAL；真实v1.1 PDMS 5:5:2；逐候选独立参考评分。
- G8/K10/shift5；随机步骤[7,8,9]；源rigorous SDE、创新±5、逐维relative logprob clamp7。
- 组内中心化、fresh rollout跨rank原始reward共同std；四次inner更新固定旧链与优势。
- PPO clip0.005；转移均值anchor beta50、三维权重[0.2,1,1]。
- AdamW5e-5，betas[0.9,0.95]，warmup100，之后constant。
- **沿用World已登记有效gradient clip1**。旧DDP源clip0不属于本World配方，不在此更改。
- train42/eval2101；50k optimizer updates = 12500 fresh全局rollout、100000 scene曝光、800000候选。
- TRAIN采用World物理log隔离清单上的作者公开hard-list交集，15171场景；不混用旧模型的TRAIN。
- 部署单候选clean-x shift5 ODE10，不使用训练SDE结果替代部署评价，不自动增加seed。

## 本次修改边界

新增独立、sealed 的 `world_postcot_explicit_IL40000_parent_v1` 用户父模型选择合同。
绑定完整IL40k模型/清单/身份、现有训练YAML、TRAIN hard manifest和种子。
原 `world_postcot_DEV_selection_v1` 的100k/完整DEV窗口检查保持原样。
actor、SDE、概率、优势、loss、optimizer和训练YAML均不修改。

新增有限owner依次运行真实8卡16-update验收和正式50k；验收必须证明冻结hash、LoRA更新、
actual sampling/recompute mean/std与独立评分通过；正式训练重新加载原IL40k并重新零初始化LoRA。
不恢复旧IL controller、不停止任何其它实验、不使用已到期临时主机。

## 资产和运行入口

- 父模型、hash和选择事实：[parent_selection.json](parent_selection.json)。
- 运行根：`/mnt/project/qwen-world-postcot-20261025/parent40k_user20261027`。
- 正式run：该根 `runs/RL_CANONICAL_IL040000_seed42`。
- 验收run：该根 `qualification/RL16`；不能作为正式初始化。
- owner调用：

```bash
python -m tools.qwen_world_postcot.start_parent40_rl \
  --registration /mnt/project/qwen-world-postcot-20261025/parent40k_user20261027/REGISTRATION.json
```

只启动一次；`owner.lock`、阶段INTENT、原生worker的GPU UUID锁与结果防重复。
源使用git归档的独立不可变快照；失败保留最后合法恢复点并停止，不盲信failure目录、不自动换配方。
正常checkpoint为原生 `updateNNNNNN_innerI`，恢复时在相同source/原run上给原命令追加：

```bash
--resume /absolute/original/run/checkpoints/updateNNNNNN_innerI
```

38项定向测试已通过（新父模型合同、原100k选择合同、clean-x真实源算术/LoRA/采样与重算/上下文生命周期）。
这不代表规划成绩提高；真实16-update验收、正式训练及后续Navtest状态以运行收据为准。

首轮8卡验收在第一次Qwen前向时因启动环境缺少 `CUBLAS_WORKSPACE_CONFIG` 被确定性检查拒绝，
未完成采样、评分或任何optimizer更新。原失败目录完整保留，不能算成功验收。
owner现显式在导入torch之前设置已有数值合同要求的 `:4096:8`；不关闭deterministic检查，
不改actor数学。第二次验收使用新的attempt/qualification目录，禁止覆盖首轮失败证据。

训练milestone保持0/5k/10k/20k/35k/50k；Navtest只有固定seed2101、全12146、同源单候选ODE。
后续与自身IL40k作逐场景配对，报告NC/DAC/TTC、G+/G−和零分转移。
不因训练reward或DEV上升宣称成功；未完成阶段不填造分数。
