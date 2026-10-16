# 固定安全难例曝光对照：R_CANONICAL_X1_SAFETY25

2026-10-16。基于已完成X1和完整TRAIN信号分析，登记一个有限、独立命名的数据曝光对照。目标仍是canonical v1.1、原十步均匀ODE、seed2101、单候选Navtest的真实提升及NC/DAC/TTC保持。92尚未达到，也不是预设结果。

状态更新：真实八卡恢复、合并部署已通过，2026-10-16 13:44:23 UTC在授权的Tsinghua zt4启动正式训练，首100次实际更新已验证。正式训练与控制器使用不可变源码`12c46a5bafd1e37a14435cd1e34aeb175731ca2f`；本报告及后续只读审计补充不热修改运行源码。资格权重未作为正式初始化，尚无本变体的单候选ODE部署收益结论。

2026-10-16 14:50 UTC补充：[首2500更新同候选预算分析](early_training2500/README.md)已完成。相对X1的安全差异组为21.84%→35.86%，progress-only为72.86%→59.64%，有效PG组99.48%→98.54%。新人口已经覆盖全部1,046个固定安全失败场景并产生204次重复曝光。安全排序信号更多，但两组训练人口、global std和重复情况不同，不能用随机训练均分证明ODE收益；仍等待固定阶段和终点评价。下文首100吞吐/ETA保留为早期历史快照，不代表当前剩余时间。

后续已完成[六个固定保存批次的分层梯度复算](gradient_prefix2500/README.md)：失败场景的PG方向未被更大global std消除，原X1在出现同类场景时也有明显贡献；PG/anchor关系随批次和步骤变化。原loss梯度对照及只读性通过，三次统计归约误报与修正完整保留。没有新增策略轨迹或参数更新，也不以此代替ODE评价。

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

CPU和下列GPU资格入口均已实际执行，结果见[qualification.json](qualification.json)。所有新attempt使用独立输出目录，保留失败证据；不覆盖正式run或原始资格目录。

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

## 真实学习、恢复和部署验收

八卡连续8更新与inner2中断后恢复到8更新完全一致：192个LoRA张量、Adam及scheduler、全部rank RNG、下一份fresh链、old probability、advantage、scene/分层位置、loss、gradient norm和ratio均通过比较。192个张量均实际改变，192份Adam一/二阶状态均为非零FP32，冻结SFT/reference hash前后相同，DDP没有重建bucket。正式起始adapter hash与原X1完全相同。

真实采样仍是[7,8,9]，每次collection实际6安全+2失败。首inner未更新时ratio误差和PPO clip fraction均为0，后续inner的真实clipping不被清零。新增实际曝光审计在连续与恢复日志上均通过，并通过错scene、错曝光、错behavior版本、错old-prob摘要、错collection五类内存故障注入；没有改动原始记录。

非零LoRA资格权重合并后，从2个原始TRAIN观测完成两个网格的4次独立canonical评分，另有1次原部署前向一致性检查；四个评分status均为OK。推理不依赖teacher、训练condition cache、LoRA wrapper或搜索。该结果明确为两场景资格验收，不能写成完整TRAIN512评价或新模型效果。

八卡资格任务端到端181.69秒；24份worker invocation包含连续8rank和两段恢复各8rank，累计worker wall含加载1,285.85秒，其中准备563.04秒、保存36.85秒。训练资格共256条新评分、16次逻辑optimizer更新，未因恢复把old collection重算计入新候选。合并导出43.33秒，原观测评价worker总35.97秒，二者单列。共享worker时间不是独占GPU计算时间；每rank测得6.63GiB仅是torch tensor峰值，不是NVML总显存占用。

一次管理用检查最初把tensor类型的`stoch_idx`直接与Python列表比较而失败；实际保存值为tensor([7,8,9])，转换成值列表后通过。训练、采样和已保存状态没有因此改动，完整披露在资格记录中。

## 首100更新：真实执行证据

[first100_profile.json](first100_profile.json)核对八个rank的前100次完整更新、25次fresh rollout、200次scene曝光、1,600条候选。实际人口为150次安全困难/50次安全失败，与登记吻合；四次inner中的behavior、chain/old-prob/advantage身份和探索统计固定。192个正式LoRA张量均变化，全部192份Adam状态计数为100。

此窗口PG组比例100%，随机策略PDMS75.3873、联合安全候选比例82.0%。这是新75:25人口上的随机训练读出，不能与原X1的95.245:4.755人口均分直接比较，也不能用来判断部署提升。两组真正匹配的比较仍是固定DEV/20k Navtest。

首个封存100更新窗口实测121.19秒，即0.8252 updates/s，按当前共享负载线性外推剩余训练约6.70小时，另计正式评价；它不是完成时间承诺。更新、rollout、评分、保存和加载分别保留，后续用完整新窗口刷新估计，不把累计update除以一次恢复的wall。

正式训练命令已执行：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m scripts.analysis.simwam_recipe.dispatch_formal \
  --variant R_CANONICAL_X1_SAFETY25 \
  --root /mnt/project/simwam-safety-exposure-20261016 \
  --host training-vlawm-tsinghua-zt4 \
  --snapshot /mnt/project/simwam-safety-exposure-20261016/source/12c46a5bafd1e37a14435cd1e34aeb175731ca2f \
  --qualification eight_gpu_safety25_v1
```

`dispatch_formal`在已有正式identity时拒绝无`--resume`重启。需要恢复时，先检查最后合法checkpoint与当前PID/start-time，确认旧进程已停止，再向同一入口显式传入合法未合并checkpoint；不得把上述命令作为重复启动操作。阶段/终点controller已分别持有独立队列的单owner锁，完成兼容性与SSH路由预检，等待登记的观察点；不重复启动controller。

只读曝光审计已执行于首100窗口，可对后续已提交窗口使用新输出路径：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m scripts.analysis.simwam_recipe.audit_safety_exposure \
  --run /mnt/project/simwam-safety-exposure-20261016/runs/R_CANONICAL_X1_SAFETY25_seed42_v1 \
  --limit-updates 100 --output "$NEW_EXPOSURE_AUDIT_OUTPUT"
```

正式运行与controller身份见[formal_launch.json](formal_launch.json)。LAST_ODE仍独立在zt2运行，其配方、权重和队列均未修改。zt留给已登记的阶段评价；zt3本次实测每卡约9.8—10.5GiB空余，未达到现有worker准入阈值，因此没有强行加入或停止其真实任务。
