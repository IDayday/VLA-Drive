# 固定安全难例曝光对照：R_CANONICAL_X1_SAFETY25

2026-10-16。基于已完成X1和完整TRAIN信号分析，登记一个有限、独立命名的数据曝光对照。目标仍是canonical v1.1、原十步均匀ODE、seed2101、单候选Navtest的真实提升及NC/DAC/TTC保持。92尚未达到，也不是预设结果。

本提交状态：代码、登记和CPU数据合同验收完成；真实八卡恢复、合并部署和正式训练尚待执行。后续执行证据单独补充，资格权重不会作为正式初始化。

## 为什么比较这一项

[X1固定20k Navtest](../ddp_simwam_x_anchor_20261015/final_navtest_20261016/README.md)原ODE为89.3817，比SFT高0.4701分、比Canonical高0.4300分，但相对SFT仍有NC/TTC净退化。[完整TRAIN分析](../ddp_simwam_terminal_ode_20261016/TRAIN_safety_signal/README.md)显示，固定SFT的TTC失败场景占实际曝光1.84%，NC失败占0.12%；这些失败场景能够产生安全候选，并非完全没有排序信号。当前有效PG组超过99.6%，不以制造更多非零优势为目标。

因此只改变固定TRAIN安全分层的曝光比例，使用已完成X1作匹配控制。它不依赖正在运行的LAST_ODE结果，可以独立使用授权空余资源；LAST_ODE保持其原配方与预算，不叠加本改动。

## 唯一实验因素及边界

相同22,017个困难场景，按原固定SFT、seed2101、canonical v1.1的单候选NC/DAC/TTC标签分成20,971个联合通过和1,046个未完全通过场景。NC=0.5归入未完全通过。两类均来自RL_TRAIN，成员固定，不利用DEV/Navtest或当前reward改变成员。

每份八卡fresh rollout恰好取6个安全困难scene groups和2个安全失败scene groups，每组保留全部G8真实当前策略候选。rank位置逐collection轮换；每个rank完整预算均为3,750次安全、1,250次失败曝光。两类各用原ResumableEpochSampler的seed42+epoch排列，越过自身epoch后继续取样，无padding、无reward拒绝采样。

| 固定20k预算 | 已完成X1 | 新曝光对照计划 |
|---|---:|---:|
| 安全困难scene曝光 | 38,098 | 30,000 |
| 安全失败scene曝光 | 1,902 | 10,000 |
| 总scene曝光 | 40,000 | 40,000 |
| fresh全局rollout | 5,000 | 5,000 |
| 新候选/逻辑评分 | 320,000 | 320,000 |
| optimizer updates | 20,000 | 20,000 |

25%是一次工程对照，不宣称最优配额，不进行配额搜索。新的人口会改变global raw reward std的数值，但其计算与loss系数不变；没有importance correction，不声称仍是uniform人口的无偏估计。scene顺序有意不同，不能把两组训练链称为逐候选随机配对。安全标签只基于一次固定SFT采样，不是所有噪声下安全。

所有其他配置从完整X1配置严格派生：原F-SFT和同一零B LoRA初始化、96个attention投影/192张量、冻结上游和qwen_proj、X1 anchor=[1,1,0.5,0.5]/beta50、G8/K10/shift5、随机[7,8,9]、源两种clamp、canonical5:5:2、四次inner、AdamW5e-5、warmup100、effective gradient clipping=0、train42/eval2101。没有GT replay、BC、dual、learned-noise、外部候选或动态课程；不从已训练X1继续训练。

新增代码只接入数据流选择、数据科学身份和固定X1评价对照。actor、sampler kernel、优势、reward、collection和正式评价代码与`8bc2d08`完全相同。旧SOURCE/CANONICAL/X1/LAST_ODE合同保留；恢复仍保存old链、概率、advantage、optimizer/RNG和inner进度。

## 已完成的CPU验收

[cpu_contract.json](cpu_contract.json)包含28项实际检查：完整40,000曝光的6+2预算、所有困难场景覆盖、每rank人口一致、跨两个cohort边界的cursor恢复及不同rank调用顺序、actor RNG不变、TRAIN/log隔离、重复/未知标签拒绝、NC=0.5处理、固定world8、严格配置拒绝和旧身份兼容。

原X1的40,000次scene位置通过旧类与新factory逐项相同。新增配置任何LR、预算、种子、clip、配额、窗口或anchor变更均被拒绝。两类排列不是通过修改旧SceneStream实现；只有新注册变体选择新类。

预先计算的新曝光中，TTC失败为3,865次（9.6625%），NC失败为248次（0.62%）。这是未来数据流计划，不是已经观察到的正式训练样本或驾驶收益。

## 真实资格验收和评价规则

资格数据在运行前按固定hash从TRAIN选24个安全、8个失败场景。八卡连续8更新，对照inner2后中断并恢复到8更新；总16次optimizer更新、256条新评分候选。比较全部rank的adapter/Adam/scheduler/RNG、相同后续链、old probability、advantage、scene与分层位置、loss/梯度/ratio。之后用合并权重从两条原始TRAIN观测执行原部署入口；资格权重丢弃，不进入正式初始化。

正式阶段仍为5k/10k/15k/20k，DEV1696与TRAIN诊断512；两个ODE网格分别报告。固定20k后锁定权重和评价源码，再执行Navtest12146/136 logs、seed2101，不取best-of-two、不增加seed。匹配主对照是同阶段X1，另保留同网格SFT；不能误把Canonical标成这项唯一因素的控制。

需要同时检验总分、NC/DAC/TTC新增/恢复、固定SFT-safe人口退化、零分转移、G+/G−与EP。更多失败曝光可能过拟合1,046个场景或损失进度；非零优势更多和随机策略均分提高都不是最终成功标准。

## 入口与资源

CPU检查已实际执行；下列GPU入口在首次真实执行后记录具体源码、host、PID/start-time、结果和成本。所有新attempt使用独立输出目录，保留失败证据；不覆盖正式run或原始资格目录。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m scripts.analysis.simwam_recipe.qualify_safety_exposure \
  --output "$NEW_CPU_QUALIFICATION_OUTPUT"

python -m scripts.analysis.simwam_recipe.qualify_recovery_run \
  --name eight_gpu_safety25_v1 \
  --root /mnt/project/simwam-safety-exposure-20261016 \
  --config configs/flow_rl/simwam_canonical_x1_safety25_ddp.yaml \
  --manifest /mnt/project/simwam-safety-exposure-20261016/qualification/real_TRAIN_tokens.json \
  --interrupted-after 2
```

正式调度复用`dispatch_formal --variant R_CANONICAL_X1_SAFETY25`和登记root；阶段与终点评价复用`watch_evaluations --followup-safety25`、`final_navtest serve --followup-safety25`。调度器要求CPU和真实八卡资格PASS，且正式source_files与已验收源码一致。

只使用当前Flow授权的Tsinghua zt/zt2/zt3/zt4，实时检查显存并允许共享。保留未知GPU占用，不发信号或reset；不使用已撤销主机，不恢复旧C/N。研究登记见[registration.json](registration.json)。本分支不合并、不上传权重/cache/原始数据/私有scene清单。
