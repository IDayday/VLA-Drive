# 已保存batch的anchor通道复核

本补充只读取两组5k/10k/15k/20k权重和对应已保存的真实链。**没有新候选、评分、optimizer更新或新训练配置**；正式训练与Navtest结果不变。

此前建议的“纵向权重0.2改为1”必须加一个明确限定：源实现会重新除以权重和。它是一个配置字段的变化，**不是只增加纵向约束、其余约束强度不变**。

## 归一化的真实影响

三个逻辑误差为`e_x`、`e_y`和`e_heading = (e_sin + e_cos)/2`，均为保存状态上current/reference随机转移均值的平方差。

```text
原锚定：50 * (0.2*e_x + e_y + e_heading) / 2.2
拟议锚定：50 * (e_x + e_y + e_heading) / 3
```

因此纵向系数变为原来的 **3.6667倍**，横向和heading-pair系数均变为 **0.7333倍**。没有偷偷保留2.2作分母，也没有修改原模型。heading-pair仍是原归一化sin/cos平方误差，不是角度MSE。

## 真实梯度证据

每个milestone按原八rank等权聚合，再对随机步骤7/8/9取平均。下表范数采用FP64归约，anchor已包含beta50；总梯度为PG加anchor，**不是Adam预条件后的实际更新**。

| 保存batch | PG与原纵向anchor cosine | 原anchor norm | 拟议anchor norm | 原/拟议总梯度 cosine |
|---|---:|---:|---:|---:|
| R_SOURCE / 5k | -0.3725 | 0.012158 | 0.022325 | 0.8794 |
| R_SOURCE / 10k | -0.3600 | 0.007328 | 0.020743 | 0.7435 |
| R_SOURCE / 15k | -0.4906 | 0.008817 | 0.023363 | 0.7653 |
| R_SOURCE / 20k | +0.3338 | 0.013256 | 0.041966 | 0.9249 |
| R_CANONICAL / 5k | -0.3669 | 0.014256 | 0.019507 | 0.8605 |
| R_CANONICAL / 10k | +0.0193 | 0.005790 | 0.013151 | 0.8844 |
| R_CANONICAL / 15k | -0.4105 | 0.008001 | 0.017039 | 0.8547 |
| R_CANONICAL / 20k | +0.1853 | 0.010310 | 0.028381 | 0.9006 |

纵向anchor与PG的夹角随阶段和batch改变；例如CANONICAL在5k/15k为负，10k/20k为正。不能断言纵向保持始终阻碍进度，也不能由重新加权后的梯度推出TTC一定改善。

四个milestone两组使用相同场景顺序；共32个独立场景、64次场景读取、8份全局batch。它们是已登记milestone的最后一份behavior，不能当作训练人口无偏样本。重算使用完成该collection四次inner后的检查点与原old链，这不是第一inner的ratio检验。

这项分析表明该建议会实质改变更新方向，值得作为独立单因素假设核实；同时，它削弱横向/朝向相对保持的副作用必须在结果中单列。**尚无“换这个权重即可到92”的证据。**困难清单人口偏好和DEV覆盖差异仍未解决；本轮不改变它们。

## 验收与精度

- CPU FP64/FP32对照字面源H/D reduction和逻辑通道分解，包括G1/G8、heading-pair和零误差。
- 8份真实batch的通道梯度和直接source anchor梯度最大绝对差为`3.493e-10`；额外直接调用原`_policy_loss`核对总loss和总梯度。
- 24个随机步骤的PG范数，用旧FP32归约与既有诊断**完全一致**。新FP64归约比旧FP32大约高0.097%—0.141%；这是数百万维向量的统计精度差异，不是actor数值变化。旧摘要保持不变，两种读数均保留。
- 所有LoRA权重与加载值逐项相同，所有参数没有残留`.grad`；没有创建或调用optimizer。

最初真实验收将FP64范数与旧FP32读数直接比较而拒绝。已修正比较口径，保留失败日志与两种统计，不放宽训练clip、改变梯度或覆盖旧证据。另一次启动在源码archive完成前失败，未加载模型；后续均在完整不可变快照执行。

[validation.json](validation.json)保存实际检查，[results.json](results.json)包含逐步骤/通道、Gram矩阵和输入hash。

## 成本和源码

8份成功分析合计376.094秒共享GPU worker wall，其中加载146.393秒、分析229.701秒。先单卡资格，再7卡并行；成功任务起止跨度117.405秒，含两阶段调度间隔。两次失败的精确worker时间未记录，保持UNKNOWN并保留日志hash，不能将376秒声称为包含失败的总成本。

实际使用`training-vlawm-tsinghua-zt`空闲GPU，结束后八卡均释放；不修改G任务，不恢复C/N。源码`2d5b402f93f183d456184b09b41fe37b72aa98e4`相对已交付`f10647f`只新增一个分析脚本，未改任何训练模块。[identity.json](identity.json)、[costs.json](costs.json)保留身份和成本。

## 可执行命令

CPU资格已实际执行，GPU命令在上述不可变源码目录执行；输出已存在，重复运行应换新输出名以保留证据。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.simwam_recipe.diagnose_anchor_channels --qualify \
  --output /mnt/project/simwam-audit-recovery-20261015/anchor_channels/qualification_v1.json

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 NO_ALBUMENTATIONS_UPDATE=1 \
  /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.simwam_recipe.diagnose_anchor_channels \
  --run /mnt/project/simwam-recipe-20261015/runs/R_CANONICAL_seed42_v1 \
  --checkpoint /mnt/project/simwam-recipe-20261015/runs/R_CANONICAL_seed42_v1/checkpoints/update020000_inner0 \
  --collection 5000 \
  --previous-diagnostic /mnt/project/simwam-audit-recovery-20261015/gradient_diagnostics/R_CANONICAL_update020000.json \
  --output /mnt/project/simwam-audit-recovery-20261015/anchor_channels/R_CANONICAL_update020000.json
```

## 唯一下一轮提议，尚未启动

[proposal.json](proposal.json)明确一条候选训练线`R_CANONICAL_X1`：从同一SFT/LoRA初始化开始，仅改`kl_dim_weights`为`[1,1,0.5,0.5]`，其余正式配方全部保持，以已完成R_CANONICAL作控制。旧两组加载器仍拒绝第三配置，未绕过合同。

预算为20k updates、5k fresh rollouts、40k场景曝光、320k新候选；单机八卡、seed42，评价seed2101。固定5k/10k/15k/20k的DEV及TRAIN诊断、锁定20k后的完整Navtest，原生ODE和shift5分列。新增阶段预测17,664条，终点预测24,292条，评价和资格成本另计；历史训练wall约5—7.4小时仅供资源估计，受共享负载影响。

仍以原生ODE真实净收益和NC/DAC/TTC新增违规/恢复共同判断，不以最佳中间点、网格或种子挑92。最近审阅任务明确要求“下一轮建议，但不自动启动新算法”，因此这里只提供具体可审阅的单因素提议，不启动训练。
