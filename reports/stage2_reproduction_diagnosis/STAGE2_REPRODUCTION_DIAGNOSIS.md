# DriveVLA-M0 Stage-2 复现差异诊断

审计日期：2026-08-30  
基准分支：`fix/stage2-official-reproduction`  
基准提交：`6e96cf7321b134c42c2cf0fbbc315cd61c925b11`

## 当前结论

本地已完成模型的 Navtest PDMS 为 `0.8998889219`，开源 Base 权重在完全相同的
12,146 个场景和本地评测链路上为 `0.9095938788`，差值为 `-0.0097049569`。

这里的正确复现目标是**无记忆的裸 Base Model，约 `0.910`**，不是论文 Table 1
中的 `0.923/0.941`。论文 Table 3 明确把 `Base Model†`（without memory）列为
`91.0` PDMS；`92.3` 来自 4K memory 下的 Map+Agent retrieval 与 TTT，`94.1`
又使用 10K synthetic memory 的 Scale 配置。当前 ModelScope checkpoint 的
`90.9594` 与论文裸 Base 只差 `0.0406` 个百分制点。因此本次 Stage-2 训练若达到
约 `0.910` 即完成 base-action-decoder 复现，不能要求一个不运行 retrieval/TTT 的
checkpoint 单独达到 `0.923` 或 `0.941`。完整取证见
`official_benchmark_disambiguation.json`。

发布侧另有一个尚不能做 tensor-level 对照的包装歧义：ModelScope 文件名为
`best-epoch_26-step_174312.server_merged.ckpt`（4,271,779,662 bytes），Hugging Face
列出的 Base 文件名则是 `VLM_forzen_actionhead_10epoch_merged.pth`
（4,264,911,545 bytes）。HF 仓库需要登录接受 gated access，服务器当前没有 token；
文件名和序列化大小不同只证明它们不是逐字节相同的包，不能据此断言模型 tensor
不同。发布 agent config 指向 ModelScope 风格的 27-epoch 文件，而该文件的本地
Navtest 已与论文无记忆 Base 基准吻合，所以当前训练继续以它作为可验证目标。

目前已经找到一个量级足以解释 proposal-bank 差距、并经单变量训练确认的首要根因：
旧训练 cache 只有普通 4 秒 GT target，官方权重的原始目录和行为指纹则指向发布源码
中未写入部署 YAML 的 `long_trajectory_additional_poses=2` 分支。该分支额外用 5 秒
logged future 重采样出的 8 点轨迹监督 proposal diversity。相同 128 场景、相同
初始化、相同 runtime 和前 1,000 step 下，仅启用这个 target 就把 best-of-64 PDMS
从 `0.943566` 提升到 `0.977166`，成对增益 `+0.033601`，按 128 条独立日志 bootstrap
的 95% CI 为 `[+0.006762,+0.065910]`。这是当前最大的因果效应；selected PDMS 在
这个早期时点尚未提升（`0.685810 -> 0.683447`），说明 scorer 尚未学会利用更宽的
候选库，不能拿 1,000-step selected 分数否定该目标。

完整 epoch-0、18,179-scene validation 现已进一步确认这个结论。在相同的 source
warmup-cosine 曲线下，long-2 相对 no-long 把 selected PDMS 从 `0.718126` 提高到
`0.767152`，best-of-64 从 `0.966110` 提高到 `0.987236`，regret 从 `0.247985`
降到 `0.220083`，L2 从 `1.444121` 降到 `1.059989`。仅 proposal ceiling 一项就
收回了 no-long 到公开最终权重差距的 `94.23%`，当前只剩 `0.001295`。这是全数据
validation 尺度的结果，不是 sample 顺序或少量场景波动；最终 selected PDMS 和
Navtest 是否完全闭合仍由正在运行的 27-epoch 曲线裁决。

按重要性排序的当前判定是：

1. **缺失 long-trajectory target：首要根因，已由 artifact、行为指纹、严格 A/B
   和完整 epoch-0 validation 四重确认；最终 Navtest 贡献等待完整曲线确认。**
   这里的 long-2 不是把部署输出改为 10 点/5 秒：target builder 从 10 个 logged
   poses 构造仍为 8 点的渐进前移目标，其源时间为
   `[0.528, 1.083, 1.667, 2.278, 2.917, 3.583, 4.278, 5.000]s`。loss 对普通
   4 秒 GT 与 long target **分别**执行 `min-over-64` 后相加，所以候选 `i` 和
   `j` 可以不同；这等价于给 proposal generator 两个正样本模式，而不是让同一条
   轨迹同时拟合二者。模型推理仍输出 8 点/4 秒，PDM score target 也不变。该机制
   直接解释了为什么主要改善 best-of-64 proposal coverage，精确映射见
   `long_target_mechanism.json`。

   同 token、同 runtime 的 step-1000 proposal A/B 进一步验证了这一机制确实出现在
   模型输出中，而不只存在于源码解释里。128 个场景分别来自 128 条日志；4 秒 target
   与 long target 的平均末点间隔为 `5.497 m`。相对 no-long，long-2 将最接近 long
   target 的候选 L1 降低 `0.249`（日志级 bootstrap 95% CI
   `[-0.449,-0.056]`），同时将“4 秒最近候选”和“long 最近候选”的末点距离增加
   `0.718 m`（CI `[+0.333,+1.092]`），long 模式的末端行程增加 `1.151 m`
   （CI `[+0.779,+1.512]`），而 4 秒 target 最优 L1 没有显著变化
   （`+0.013`，CI `[-0.123,+0.139]`）。两组模型中两个 argmin 本来就常常不同
   （`88.28%` 与 `87.50%`），因此关键证据不是“不同 argmin 比例”，而是两模式的
   几何分离与 long-target 拟合同步增强。可复现明细见
   `long_target_candidate_specialization.json` 和
   `local_stage2/audit_long_target_candidate_specialization.py`。
2. **Flash Attention：确定的有害训练语义改动，已修正为 eager。**
3. **逐 step LR scheduler：已由官方 checkpoint loop state 直接确认存在。** 四个
   历史 shard 的 scheduler progress 均为 `174312/174312`；精确类型虽被剥离，
   但发布源码唯一实现和作者训练谱系都指向 10% warmup-cosine，当前主跑已修正。
4. **Lightning 2.2.1、seed 2、16×1、Transformers 4.48.3：必须锁定的复现条件，
   但各自短程效应显著小于 long target，不再作为首要原因。**

差值不是由样本顺序或本地 evaluator 引起。下面 1--13 项保留了发现过程；
`官方 checkpoint 原始元数据修正`一节中的直接 artifact 证据覆盖其中早期的
runtime/scheduler 优先级判断。

1. **主差距来自 proposal bank，不是 scorer 排序。** 在完整 navtrain
   validation 上，公开权重与本地 best 的 selected PDMS 差为 `0.013444`，
   但 best-of-64 上界差为 `0.013820`，可解释约 103% 的 selected 差距。
   两者 scorer regret 分别是 `0.037056` 和 `0.036681`，本地 scorer 甚至略低。
   因此真正需要修复的是轨迹生成分支的候选质量/覆盖，而不是 PDM
   评分器或 scorer 选择逻辑。进一步检查 27-epoch 曲线发现，本地 best-of-64
   在 epoch 2 已达到 `0.983240`，到 selected 最优的 epoch 25 降至
   `0.974710`，epoch 26 又降至 `0.966451`；与此同时 L2 从 epoch 2 的
   `0.669527` 继续下降到 epoch 26 的 `0.375107`。这是训练后期 proposal
   planning coverage 塌缩的直接证据。
2. **Flash Attention 是已由因果 A/B 确认的有害训练改动。** 旧本地全量训练启用了
   FlashAttention 2，而发布配置为 eager。相同 seed、数据、优化器和 1,000 step
   下，Flash 的验证 PDMS 为 `0.8058354`，eager 重复实验为 `0.8382344`，下降
   `0.0323991`；64 候选的 oracle 上界基本不变，损失主要来自 scorer 选择。单步
   审计中 Flash 相对 eager 的 hidden-state 相对 L2 差异为 `16.52%`，梯度为
   `9.89%`；经过 AdamW 后，参数更新向量相对 L2 差异达到 `60.28%`、cosine 仅
   `0.8258`。它不是可以忽略的最后几位浮点误差。
3. **旧本地 8×2 不是公开权重的 16×1 训练布局。** 公开 checkpoint 的
   `174312 / 27 = 6456` step/epoch，而 `ceil(103288 / 16) = 6456`，从 checkpoint
   本身可以反推全局 batch 为 16。结合论文的 16 张 H20，实际布局应为
   16 rank × 每 rank 1 样本，而不是发布 YAML 中的占位 batch 值。相同
   global batch/seed/LR 的 1,000-step 实测中，4×4、8×2、16×1 分别为
   `0.839999`、`0.794061`、`0.807793`。差异不呈单调，但已证明 rank×local-batch
   是大变量。ActionDecoder 硬编码启用 0.1 dropout 和 0.2 stochastic depth，
   各 rank 又使用同一 global seed，因此不同 rank 布局会改变随机 mask 与
   样本的绑定；这不是样本顺序解释。
4. **公开权重使用 seed 2 初始化，本地旧训练使用 seed 0。** 这是逐位精确的
   checkpoint 指纹，不是猜测。不过 seed 2 在 1,000 step 的验证 PDMS 为
   `0.8265270`，低于 seed 0 两次重复的 `0.8382344--0.8444353`，所以 seed 不足以
   单独解释公开权重更高的最终成绩。
5. **发布仓库没有包含生成公开权重的完整 launcher/config。** 发布 agent YAML
   指向最终公开 checkpoint，并保留 `batch_size=2`、`num_gpus=1`、`base_lr=5e-4`；
   generic trainer 又是 `batch_size=64`、`devices=1`、`max_epochs=20`。这些值与公开
   checkpoint 可反推的 27 epoch/global batch 16 不可能同时成立。因此“直接运行
   开源 YAML”不是官方 Stage-2 复现；私有 launcher 中的 LR 缩放、scheduler 和
   runtime 仍未公开。
6. **实际 peak LR 细节仍有歧义，但 scheduler 是否存在已经锁定。** 论文附录明确报告 base model
   使用 AdamW、学习率 `1e-4` 和 16 张 H20；但发布代码会把 `base_lr` 再乘
   `sqrt(global_batch/base_batch)`。论文没有说明 `1e-4` 是缩放前的配置字段还是
   缩放后的 optimizer-group LR。global batch 为 16、`base_batch_size=64` 时，两种
   解读分别得到实际 LR `5e-5` 和 `1e-4`。发布 YAML 自身的 `base_lr=5e-4` 又会得到
   `2.5e-4`；其未随 launcher 改动的 `agent.batch_size=2,num_gpus=1` 则会得到
   `8.84e-5`。本机 PL 2.6 的 `2.5e-4` 在 1,000 step 得到 `0.8258788`，没有显示
   实质早期优势；`5e-5` 在 1,000 step 欠拟合，但这不能替代 27 epoch 的完整曲线。
   发布 YAML 虽写着 `scheduler_args: null`，四个官方历史 shard 的 Lightning loop
   state 却逐一记录 scheduler `ready=completed=174312`，与 optimizer step 完全相等；
   本地无 scheduler 控制的同一字段严格为 0，有 source-cosine 时则等于全部 step。
   因而私有训练启用了逐 step scheduler 已是 artifact 事实。scheduler state 被剥离，
   无法仅从 checkpoint 恢复精确类型；发布源码唯一实现是完整的 10% warmup + cosine。
   开源权重相对
   seed-2 初始化的有效 action-head RMS 位移为 `0.0219354`，本地旧权重相对 seed-0
   初始化为 `0.0416902`，比例为 `0.5262`；逐模块比例也集中在 `0.48--0.66`。
   常数 `5e-5` 与 peak `1e-4` 的 cosine 都约有常数 `1e-4` 一半的累计 LR，单凭
   checkpoint 位移无法区分两者。严格 16×1 的恒定 `1e-4` 已完成一整轮，其位移
   外推明显超过公开权重；结合 DriveVLA-M0/DrivoR/ReCogDrive 的源码沿袭关系，当前
   优先完整验证 peak `1e-4` 的源码 warmup-cosine，恒定 `5e-5` 保留为失败后的首个
   完整对照。当前完整主跑采用源码 schedule；最终曲线检验其类型和 peak LR 是否也匹配，
   而不再检验“官方是否用了 scheduler”。
7. **冻结 VLM 的 train/eval mode 是次要因素。** 1,000-step A/B 中，eval-mode
   反而比 train-mode 高 `0.00141` PDMS；差异很小且方向不能解释旧本地结果偏低。
8. **Lightning 版本必须锁定，但不是单独主因。** 官方历史 checkpoint 的四个
   原始 shard 都直接记录 `pytorch-lightning_version=2.2.1`。严格 16×1、
   seed-2/eager/实际 LR `1e-4` 的 1,000-step 单变量对照中，2.2.1 与 2.5.1 的
   selected PDMS 为 `0.801635/0.798398`，best-of-64 为
   `0.985965/0.991843`。selected 只差 `+0.003236` 且 proposal ceiling 反向变化，
   量级远小于 long target 的 `+0.033601` best-of-64 增益。完整修正 run 因此锁定
   直接有 artifact 依据的 2.2.1，而不把框架版本包装为主要解释。

9. **Transformers 运行时会改变训练路径，但 4.37.2 已降为次级对照。** 4.48.3
   与 4.57.6 的单样本 eager 前向逐位一致，但反向梯度和 1,000-step 参数路径并不
   一致；4.48.3 的 proposal ceiling 高 `0.006041`。更重要的是，实际 Stage-1
   ReCogDrive 产物的 `generation_config.json` 明确记录 Transformers `4.37.2`，
   同目录源码又锁定 tokenizers `0.15.1`。恢复兼容的 PEFT `0.10.0` 后，同权重、
   同像素、同随机数下，4.37.2 相对 4.48.3 的 VLM hidden-state RMSE 为
   `0.130629`，单步 action-head 梯度 RMSE 为 `0.005095`，已经是不同的训练语义。
   PEFT `0.10.0` 与当前 PEFT 的前向则逐位一致、梯度 RMSE 仅 `0.000284`，说明
   主要分叉来自 Qwen2 Transformers 实现而不是 PEFT 包装。随后做的 checkpoint
   血缘审计纠正了版本证据的含义：公共 DriveVLA checkpoint 中 393 个非 LoRA
   backbone tensor、132 个 base-layer bias，以及 160 个 LoRA target 的
   `base_layer.weight` 都与 ReCogDrive 基座逐位一致；公共权重另外保存了非零
   LoRA A/B。也就是说，`4.37.2` 元数据只锁定了冻结基座的序列化来源，不能证明
   新增 LoRA 的 DriveVLA Stage-1/2 训练运行时。DriveVLA 所用 InternVL3 配置中的
   `4.48.3` 是更直接的本项目证据，因此保持 4.48.3 全曲线，4.37.2 只保留为
   scheduler 和梯度裁剪之后的匹配训练对照。
10. **梯度裁剪是新识别出的高影响、但公开证据较弱的候选。** DriveVLA 发布配置明确
    设置 `gradient_clip_val=0.0`，而其上游 ReCogDrive 的 Stage-2 默认值和真实
    Stage-1 `training_args.bin` 都是 `1.0`。固定真实批次上，action-head 未裁剪
    全局梯度范数约为 `703.66`，所以 clip=1 并非空操作。虽然 AdamW 对统一梯度缩放
    近似尺度不变，`eps` 和逐参数梯度分布仍会破坏这种等价性；模拟首步 Adam 梯度项
    后，clip=1 与不裁剪的更新相对 RMS 差为 `0.2763`、cosine 为 `0.96197`。
    DriveVLA 自身的显式配置仍是反对证据，因此不会直接覆盖主跑；但它现在排在
    scheduler 之后、4.37.2 版本对照之前做严格 A/B。

因此，现阶段最严格的表述是：**旧本地训练不是官方 Stage-2 的等价复现，首要代码/
配置错误是没有构造和启用 5 秒 long-trajectory 辅助监督，导致训练后期 proposal
bank 收缩；Flash Attention、8×2、seed 0、错误 Lightning 版本又进一步改变了训练
路径。** 完整 27 epoch 仍需判断修正后能否闭合最终 Navtest，以及 warmup-cosine
是否与私有 launcher 一致。sample 顺序已按用户要求移出主因验证队列。

工程修复不再只存在于一次性的启动命令中：`train_stage2_reproduction.sh` 和
`launch_stage2_multinode_reproduction.sh` 现在默认选择 long-2 cache、
`long_trajectory_additional_poses=2`、Lightning 2.2.1、Transformers 4.48.3、eager
attention 与当前优先验证的 source warmup-cosine；所有歧义仍可显式 override。
通用的 `train_stage2_full.sh` 保持旧实验兼容默认值。短程 no-long 因果对照则在各自
launcher 中显式固定普通 cache 和 `long=-1`，避免新的正确默认污染单变量实验。

正在运行的 rank-0 进程还做了 `/proc` 级语义锁，而不是只核对预期 YAML：seed、
16×1 布局、BF16、eager、long-2 cache、VLM 冻结、AdamW、实际 LR、scheduler 和
27 epoch 共 25 项关键 override 全部匹配。更直接地，候选生成
`action_decoder.py`、轨迹/评分 loss `episode_drive_loss.py` 和 target builder
`drivevla_features.py` 三个文件与发布 commit `b9a4f27` **逐字节相同**。现存加速项
只位于图像预处理、tokenization、离线 PDM 标签计算、validation 合并评分、训练日志
collective 和 checkpoint I/O；已有固定批次审计分别证明其数值等价。因此此前的加速
改动中真正改变训练语义的是已关闭的 Flash Attention，而不是仍启用的吞吐优化。
机器可读证据见 `active_run_semantic_lock.json`。

## 官方 checkpoint 原始元数据修正（当前最高优先级）

ModelScope 仓库历史仍保留四个已删除 checkpoint shard 的 LFS 对象。使用 HTTP
Range 只读取每个约 1 GB shard 的前 1 MiB ZIP pickle 元数据后，四份独立记录都给出：

```text
epoch = 26
global_step = 174312
pytorch-lightning_version = 2.2.1
```

四个 shard 各自包含 `319/320/339/345` 个不重叠 state key，总和 `1323`，正好等于
当前 merged checkpoint 的 tensor 数。进一步从 shard 4 的 ZIP central directory
只读取一个 4-byte action scorer tensor，并与 merged checkpoint 比较：

```text
key: agent.action_head.scorer.pred_score.no_at_fault_collisions.2.bias
historical shard storage: 339
merged storage: 1317
historical bytes: d59b623d
merged bytes:     d59b623d
float32 value: 0.05532439425587654
```

因此 Lightning `2.2.1` 是生成当前公开权重的直接事实，不再是 requirements 的版本
猜测。此前用 Lightning `2.5.1` 启动的 source-cosine 长跑已在保留 epoch-0
checkpoint/日志后停止。`2.2.1 vs 2.5.1` 的严格 16×1、1,000-step 单变量对照已经
完成：selected PDMS 分别为 `0.801635/0.798398`，best-of-64 为
`0.985965/0.991843`，regret 为 `0.184330/0.193445`。框架版本确实改变路径，但
selected 只差 `+0.003236` 且 proposal ceiling 反向变化，不能解释完整复现差距，
因此从主因降为必须锁定的运行时条件。

同一 pickle 元数据还证明四个历史 shard 中的 `optimizer_states` 与 `lr_schedulers`
都被保存为真正的空列表，而不是审计脚本加载失败；因此无法直接恢复 scheduler 类名
和 LR 数值。关键的新证据保留在未剥离的 loop state：四个 shard 的
`epoch_loop.scheduler_progress.total.completed` 都是 `174312`，与各自 optimizer
completed step 完全相同。本地 Lightning 对照证明无 scheduler 时该计数为 0，
source-cosine 时才逐 step 增长。因此官方逐 step scheduler 的存在已直接确认；
结合发布源码唯一 scheduler 分支和作者 Stage-1 的 10% warmup-cosine 习惯，当前
主跑选择该曲线不再只是按参数位移猜测，但精确类型仍需最终性能交叉验证。

四个 shard 的 callback tensor 还分别保存了完全相同的
`best_model_score=0.951407671`，monitor 名称为 `val/score_epoch`。当前本地 evaluator
对合并后的同一公开权重得到 `0.95147377`，绝对差仅 `0.00006610`。这把“本地验证
PDMS 与官方训练 callback 不是同一口径”从推测性排除提升为 checkpoint 直接验证，
并将修正训练的完整 validation 目标锁定在约 `0.9514`。

同一原始 metadata 还保存了真实 checkpoint 路径：

```text
training_episode_Nav1_traj_long_25epochs_visionlora/
.../best-epoch=26-step=174312.ckpt
```

`traj_long` 与发布源码中唯一同名语义的分支高度吻合：模型输出 8 个 0.5 秒轨迹点，
训练 Scene 却加载 10 个未来帧；`long_trajectory_additional_poses=2` 会将这 10 个
logged-future pose 用发布代码的 cubic spline 精确重采样为 8 点、把末点从 4 秒移到
5 秒，并把第二个 min-over-64 L1 无权重地加到 trajectory loss。发布 YAML 的 `-1`
属于指向最终 checkpoint 的推理/部署配置，不能再当作私有训练未启用该目标的证据。

已在完全不修改原 cache 的前提下，为 `103288/103288` 个训练样本生成独立 long-2
target cache；普通 4 秒 target 与 raw log 重建的最大误差为 `0`，long target 末点
相对普通 target 平均前移 `4.558734 m`、最大 `19.968922 m`。公开权重的 128-scene、
128-log proposal 指纹进一步显示：普通与 long target 的末点平均相距 `5.497433 m`，
但公开 64-candidate bank 对 long target 的最小 endpoint error 仅 `0.357977 m`，
`96.09%` 的场景存在 official-L1 小于 1 的 long-target proposal；普通/long 最近
proposal 只有 `3.91%` 是同一条。这与“双轨迹 min loss 保留两个 proposal mode”高度
一致，也精确解释了旧本地训练中“4 秒 L2 继续改善、best-of-64 planning ceiling
反而塌缩”的现象。

数值完整性审计进一步覆盖全部 `1192` 条训练日志，每条日志按 cache 原生枚举顺序
检查一个 target：`1192/1192` 个 long target 全部有限，没有航向原始单步跳变超过
π，也没有 wrap 后单步变化超过 `0.5 rad`；最坏 long-target 单步为
`0.465734 rad`。因此没有证据表明 cubic-spline 对 heading wrap 的处理在当前数据上
制造异常监督。该检查记录在 `long_target_cache_integrity.json`，可提高
`--samples-per-log` 做更密集审计。

随后已对旧本地 no-long best checkpoint 在完全相同的 128 个 token 上导出全部候选。
旧模型对普通 4 秒 target 的最小 L1 为 `0.255587`，略好于公开模型的 `0.262695`；
但对 5 秒 long target 则为 `0.781929`，是公开模型 `0.363170` 的 `2.15x`，末点误差
为 `1.286898 m`，是公开模型 `0.357977 m` 的 `3.59x`。候选几何也发生明显收缩：
公开/旧模型每场景最远候选半径为 `36.483/28.856 m`，候选终点两两平均距离为
`4.956/2.666 m`。4.5 秒目标上的公开/旧 L1 仅为 `0.299/0.399`，而 5 秒差扩大到
`0.363/0.782`，因此行为指纹具体指向额外 2 帧，而不是任意轻微延长。

严格 1,000-step 单变量训练现已完成，并把这条证据从强相关提升为训练因果。为避免
validation 子集不同造成混淆，两个 checkpoint 又在完全相同的 128 token/128 log
上重新导出全部 proposal：普通 target 与 long-2 target 的 best-of-64 分别为
`0.943566/0.977166`，成对差 `+0.033601`，日志 bootstrap 95% CI 不跨 0。long-2
还把候选终点两两平均距离从 `4.444 m` 提至 `6.009 m`，终点纵向跨度从
`15.427 m` 提至 `23.401 m`；对 5 秒 target 的最近候选末点误差从 `3.878 m`
降到 `3.032 m`。这正是旧训练后期候选覆盖收缩的反向修复。早期 selected PDMS
没有同步提升，符合 trajectory head 先扩展 proposal、score head 后续再学习排序的
两阶段收敛过程。

`visionlora` 只作为运行目录标签记录，不据此解冻 VLM；[论文附录](https://arxiv.org/html/2608.10413v1#A1)
明确说明 Action Decoder 阶段冻结 VLM。`25epochs` 与实际 27 个 zero-based epoch 的
step 记录也不一致，说明目录标签可以滞后，不能把所有片段都当成精确配置字段。

可复现证据：

```bash
python local_stage2/audit_stage2_checkpoint_history.py \
  --output reports/stage2_reproduction_diagnosis/public_checkpoint_history_audit.json

python local_stage2/audit_stage2_long_target_signature.py \
  --alternate-additional-poses 1 \
  --output reports/stage2_reproduction_diagnosis/public_long_target_signature.json

python local_stage2/compare_stage2_proposal_artifacts.py \
  --left /path/to/standard_subset128.pt \
  --right /path/to/long2_subset128.pt \
  --left-name standard_4s_target \
  --right-name long_5s_auxiliary_target \
  --output reports/stage2_reproduction_diagnosis/pl221_standard_vs_long2_paired_subset128.json
```

## 2026-08-30 高优先级复现更新

在用户明确要求不再把 sample 顺序作为主因后，审计继续沿“候选生成分支为何在
后期塌缩”推进，并新增了四组更强证据。

1. **完整 epoch 的参数位移现在直接支持 warmup-cosine。** 严格
   16×1、seed 2、eager、BF16、Lightning 2.5.1、恒定实际 LR `1e-4` 的完整
   epoch 0 已跑完；validation selected/best-of-64 分别为 `0.842049` 和
   `0.966837`。该 checkpoint 相对 seed-2 初始化的有效 action-head RMS 位移为
   `0.00736329`，而公开 27-epoch 权重只有 `0.02193538`。把更新近似成噪声随机游走，
   位移按累计平方 LR 的平方根缩放，三种候选配置的终值预测为：恒定 `1e-4`
   `0.0382608`（比公开值高 74.4%）、恒定 `5e-5` `0.0191304`（低 12.8%）、
   源码 10% warmup + cosine `0.0232993`（高 6.2%）。梯度统计会随训练变化，
   因而这不是对私有配置的数学证明，但它与旧失败权重的实测位移 `0.0416902`
   共同构成当前最强的 scheduler 根因证据。
2. **Transformers 版本会改变训练路径，不能再按“前向一致”降级。** InternVL3
   配置记录 `transformers_version=4.48.3`；旧本地和刚完成的恒定-LR 对照实际使用
   `4.57.6`。受控样本的 hidden state、proposal、loss 虽逐位一致，但 Q-Former
   反向梯度已经不同；严格 16×1 跑到 step 1000 后，两版本 action-head 更新向量
   cosine 仅 `0.5333`。4.48.3 的 best-of-64 为 `0.991843`，比 4.57.6 的
   `0.985802` 高 `0.006041`，方向上与当前需要修复的 proposal ceiling 一致；
   selected 分数则更低，说明不能用短跑 selected PDMS 单独选 runtime。需要注意，
   模型文件中的版本字段可能来自序列化 VLM，而不一定就是私有 Stage-2 环境，因此
   这里的判定是“最接近公开证据的 runtime lock”，不是已证实的官方环境事实。
3. **缓存、tokenizer 与 Stage-1 VLM 语义已进一步排除。** 从 64 条日志随机抽取
   128 个场景，feature cache 与 raw Scene 重建得到的轨迹、历史、command、ego
   status 全部逐值一致且图像路径均存在。InternVL 原始目录与 ReCogDrive VQA 目录
   在实际 2,800-token prompt 上产生相同 token/mask；严格加载公开 checkpoint 后的
   hidden-state SHA256 也一致。因此缓存加速和 VLM 目录选择不是当前约 1 PDMS
   差距的主因。
4. **完整 long-2 修正 run 已启动。** 当前实验
   `stage2_official_pl221_tf448_eager_seed2_long2_source_cosine_16x1` 使用本机与
   `training-vla-zt2` 共 16 张 A800，锁定 PyTorch `2.5.1+cu124`、直接取证得到的
   Lightning `2.2.1`、Transformers `4.48.3`、eager attention、BF16、seed 2、
   16 rank × batch 1、`long_trajectory_additional_poses=2`，并启用发布源码中逐 step
   等价的 10% linear warmup + cosine decay。完整 27 epoch 后自动运行 Navtest；
   最终结果才决定 scheduler 假设是否成立以及总体性能是否真正复现。
5. **未发现私有 loss 大幅重加权的 checkpoint 证据。** [论文附录](https://arxiv.org/html/2608.10413v1#A1)
   将 score loss 简写为单一 PDM quality BCE，而发布实现对六个独立 factor head
   分别计算 BCE。公开 checkpoint 的六个独立 head 全部离开初始化；尤其发布推理
   聚合权重为 0 的 DDC head 相对初始化 RMS 位移仍有 `0.0272502`，排除了“只通过
   发布聚合公式训练一个最终分数”。进一步把公开权重与旧本地发布-loss 全量训练
   权重的六个 head 位移分别按均值归一化，模式 Pearson 相关为 `0.9516`，归一化
   RMSE 为 `0.0551`。这不能从权重反推出逐样本 target，也不能严格证明私有 loss
   完全相同，但没有看到足以优先于 scheduler/runtime 的 head-weighting 异常。
6. **Stage-1 环境元数据支持 warmup-cosine，但不能锁定 DriveVLA 的 4.37.2。**
   `/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B/training_args.bin`
   记录 BF16、10% warmup、cosine、3 epoch、24 rank、batch 1 × accumulation 16；
   `trainer_state.json` 的学习率首尾也与该 schedule 一致。它不能直接证明私有
   Stage-2 launcher 完全复用了 Stage-1 环境，但强化了 warmup-cosine 的作者
   训练习惯。checkpoint 血缘审计进一步证明，公共权重中的冻结 base tensors 与
   该 ReCogDrive 基座逐位一致，而额外 LoRA adapter 是后来加入的；因此基座目录的
   `4.37.2` 字段不能外推为 DriveVLA LoRA 的训练运行时。对公共 Stage-2 checkpoint 做了
   128-scene/128-log 成对推理：4.37.2 与 4.48.3 的 selected PDMS 分别为
   `0.962780`/`0.962948`，平均差 `+0.000168` 的 bootstrap 95% CI 为
   `[-0.000726, +0.001119]`；虽然有 `36/128` 个场景改变了选中 candidate、选中
   轨迹 RMSE 达 `0.387336`，平均 PDMS 没有显著优劣。因此不能拿公共权重推理结果
   代替训练对照。4.37.2 因而保留为次级训练对照，不再优先于梯度裁剪。
7. **rl-zt3 对照已改为严格匹配的顺序队列，但服务当前不可达。** 4×1×acc4 虽重放相同全局样本，
   但每 rank 的 dropout/RNG 流与 16×1 不同，不能把单个 4.37 结果直接和主跑归因。
   队列因此依次执行 `TF4.48/PEFT0.10/clip0`、
   `TF4.48/PEFT0.10/clip1`、`TF4.37/PEFT0.10/clip0`，三次使用相同布局、seed、
   cosine schedule 和验证子集。前两次只归因梯度裁剪，第一和第三次只归因
   Transformers；服务恢复后会自动在用户授权的 GPU 3/5/6/7 上顺序运行。
8. **此前 no-long source-cosine 的完整 epoch-0 验证符合 warmup 预期。** 16×1 对照在完整
   18,179-scene navtrain validation 上得到 selected PDMS `0.718126`、
   best-of-64 `0.966110`、regret `0.247985` 和 L2 `1.444121`。同期实际 LR 仅
   `3.6997e-5`，所以 scorer 与 GT 拟合明显落后于恒定 `1e-4` 对照是预期现象；
   更关键的是 proposal ceiling 与恒定-LR epoch-0 的 `0.966837` 只差
   `0.000727`。该结果说明 warmup 第一轮没有提前损坏候选覆盖，但不能用来宣称
   已复现最终性能。主判据仍是 warmup 结束附近的 epoch 2--3，以及后续 ceiling
   是否避免旧实验从 epoch 2 到 epoch 26 的持续塌缩。
9. **多节点 world-size 字段已显式修正，但它不是性能根因。** 旧命令中的
   `agent.num_gpus=8` 曾是误导字段；显式 `effective_global_batch_size=16` 已经优先
   决定实际 LR 和 `T_max=174312`，所以它没有改变旧主跑优化语义。新 launcher 又把
   `agent.num_gpus` 显式设为 `STAGE2_WORLD_SIZE=16`，当前 long-2 主跑的命令和运行时
   都已验证为 16。该修正消除了审计歧义，不计作性能修复。
10. **epoch-0 到公开最终权重的更新方向不能用于选择 scheduler。** 从共同的 seed-2
    初始化出发，source-cosine epoch-0 与公开最终权重的有效 action-head 更新 cosine
    为 `0.105237`；恒定 `1e-4` epoch-0 的对应值为 `0.114977`。两者都很低，而且
    早期更新与 27-epoch 最终更新本来就不是同一时间尺度。因此当前对 scheduler 的
    排序只依据发布源码分支、作者训练谱系、累计更新幅度与完整 validation 曲线，
    不再把 early-to-final update cosine 当作支持证据。
11. **long-2 的完整 epoch-0 验证已经闭合 proposal ceiling。** 当前 Lightning
    2.2.1 修正主跑在同一 18,179-scene validation 上得到 selected `0.767152`、
    best-of-64 `0.987236`、regret `0.220083`、L2 `1.059989`。相对第 8 项 no-long
    对照，selected 提高 `0.049026`，best-of-64 提高 `0.021125`，regret 相对下降
    `11.25%`，L2 相对下降 `26.60%`。公开最终 best-of-64 为 `0.988530`，所以
    long-2 已收回 no-long proposal-ceiling 缺口的 `94.23%`，只剩 `0.001295`。
    no-long 对照使用 Lightning 2.5.1，但 2.2.1/2.5.1 的严格短对照中框架变化反而
    让 best-of-64 下降，因此不能解释这里 `+0.021125` 的改善；128-log、同 runtime
    的 long/no-long 配对 A/B 又独立得到 `+0.033601`。两条证据共同把 long target
    从“高概率原因”提升为已确认的 proposal-bank 首要根因。

    改善也不是由单一 PDM factor 偶然抬高：相对 no-long epoch 0，selected
    collision、DAC、progress、TTC 分别提高 `0.014732`、`0.022318`、`0.063776`、
    `0.021438`，comfort 只变化 `-0.000055`。共同 seed-2 初始化下的参数位移比较
    显示 long-2/no-long 总体 update cosine 为 `0.7574`，trajectory head 为
    `0.8779`、trajectory decoder 为 `0.6896`，而 scorer 更新 RMS 基本不变
    (`0.003334/0.003363`)。因此 long target 改变的是 proposal 生成路径和场景编码，
    不是通过放大 scorer 更新伪造提升。
12. **当前 scheduler 实现与发布源码数值等价。** 在锁定的 PyTorch 2.5.1 下，将
    当前 LambdaLR 与原始 `LinearLR(start_factor=1e-6, 10%)` 加
    `CosineAnnealingLR` 在全部 174,312 step 逐点比较，最大 LR 绝对差仅
    `7.74e-18`。epoch 1 checkpoint 又直接记录 scheduler/optimizer/loop progress
    均为 `12,912`，318 个 optimizer state 的 step 也全部为 `12,912`；scheduler
    `last_epoch=12,912`、LR=`7.407494991e-5` 与公式一致。因此还可排除漏 step、
    重复 step 和 off-by-one。后续完整曲线检验的是私有 scheduler 是否就是该源码
    曲线及 peak LR，而不是本地 schedule 实现是否错位。状态快照见
    `corrected_epoch1_optimizer_state.json`。
13. **官方 scheduler presence 已由 loop state 直接取证。** 四个历史 checkpoint
    shard 都没有 `hyper_parameters`，且 `lr_schedulers=[]`，所以不能恢复类名；但
    每份都保留 scheduler total ready/completed `174312/174312`，与 optimizer
    `174312/174312` 一致。相同 Lightning 语义下，本地 constant-LR/no-scheduler
    checkpoint 是 `0/0`，source-cosine 和当前修正 run 都是每 step 递增。因此
    “旧训练遗漏官方逐 step scheduler”已确认，剩余问题只是在未公开 launcher 中
    它是否就是发布源码的 10% warmup-cosine 以及 peak LR 的精确解释。
14. **修正版 epoch 1 仍维持高 proposal ceiling，scorer 正在 warmup 中收敛。**
    在完整 18,179-scene validation 上，selected 从 epoch 0 的 `0.767152`
    提高到 `0.855873`，regret 从 `0.220083` 降到 `0.129305`，相对下降
    `41.25%`；L2 从 `1.059989` 降到 `0.549403`。best-of-64 从 `0.987236`
    小幅变为 `0.985178`，仍比旧失败训练 epoch 1 的 `0.979800` 高
    `0.005378`，也高于旧训练全程峰值 `0.983240`。这说明 long-2 的改善没有在
    第二个 validation point 消失，但 selected 仍比公开最终值低 `0.095601`，不能
    提前宣称复现。epoch 1 结束时 scheduler 仍未走完 17,431-step warmup；跨过
    peak LR 的 epoch 2 才是第一个决定性优化曲线检查点。逐字段结果保存在
    `corrected_long2_early_curve.json`。
15. **当前训练实际应用的 LR 逐点符合公式，而不只是配置看起来正确。** 对正在增长的
    TensorBoard 标量 `lr-AdamW/action_head_decay` 做快照审计，首个记录 step 9、
    最新记录 step 17,499，共 1,750 个每 10 step 采样点，已经覆盖 warmup/cosine
    分段边界；逐点与 peak `1e-4`、
    17,431-step linear warmup、随后 cosine decay 的公式比较，最大绝对误差仅
    `3.63e-12`（TensorBoard float32 量化范围），所有采样间隔严格为 10。由此可排除
    当前已运行区间内 Lightning scheduler 调用顺序、漏 step、重复 step 和
    off-by-one。可复跑脚本为 `local_stage2/audit_active_stage2_lr_trace.py`，证据快照
    为 `active_run_lr_trace.json`。
16. **跨过 warmup 的 epoch 2 是不利但尚未终局的曲线证据。** 完整 validation 的
    selected 从 epoch 1 的 `0.855873` 提高到 `0.864581`，regret 从 `0.129305`
    降到 `0.114956`；但 best-of-64 从 `0.985178` 回退到 `0.979538`，L2 从
    `0.549403` 变为 `0.709160`，且旧失败 run 的 epoch-2 selected/best-of-64
    `0.899419/0.983240` 仍领先。因而 long target 已确认的因果收益不能再表述为“已足以
    复现最终权重”，完整曲线、梯度裁剪和私有 scheduler 形状仍需检验。另一方面，
    当前 19,368 step 只消耗 source schedule 全程 `11.98%` 的平方 LR 预算；随机游走
    近似预期参数更新范数为公开最终权重的 `34.62%`，实测为 `37.53%`。更新幅度与
    schedule 预算相符，因此该结果也不是 scheduler 漏步或 peak LR 明显缩小的证据。
    预先设定的 epoch-9 checkpoint 仍作为中程裁决点；明细见
    `corrected_long2_early_curve.json` 与
    `corrected_long2_epoch2_vs_public_update.json`。

## 已排除或降级的因素

| 因素 | 证据 | 判定 |
| --- | --- | --- |
| Navtest 评测实现 | 开源权重在同一本地 evaluator 得到 `0.9095938788` | 排除为主因 |
| 场景集合 | 两个 checkpoint 的 12,146 个评测 token 完全一致 | 排除 |
| Stage-1 权重 | 1,005 个 frozen backbone tensor 全量、严格恢复 | 排除 |
| 图像 worker 预处理 | hidden state、proposal、score、loss 逐位一致；单步参数更新 RMSE 约 `4e-7`，远小于 `1e-4` 量级更新，另有同路径重复作为数值噪声基线 | 排除为主因 |
| worker 预分词 | token 和 attention mask 精确一致 | 排除 |
| PDM process pool/partition | sequential、pool、partition 输出逐元素一致 | 排除 |
| fused validation scoring | 与逐候选评分一致 | 排除 |
| PDM 监督/cache 代码版本 | 103,288/103,288 cache 完整；`train_pdm_scorer.py`、MetricCache、PDMSimulator 核心源码与 `upstream/main` 相同，当前改动只做任务切分和只读实例复用 | 未发现本地语义漂移；作者私有 cache 本身不可直接比对 |
| loss/head 权重 | 六个公开 score head 均被训练；公开/本地 head 位移模式相关 `0.9516`，归一化 RMSE `0.0551`；轨迹项与论文均为 min-over-64 L1 | 未发现大幅私有重加权；保留为低优先级未知量 |
| Transformers 4.37.2/4.48.3/4.57.6 | 4.48.3/4.57.6 前向一致但 step-1000 更新 cosine `0.5333`；4.37.2/4.48.3 hidden RMSE `0.130629`、梯度 RMSE `0.005095`；但 4.37.2 元数据只锁定冻结基座血缘 | runtime 会改变路径；4.48.3 证据更直接，4.37.2 降为次级匹配控制 |
| 梯度裁剪 | DriveVLA 发布值为 0；ReCogDrive/Stage-1 值为 1；固定批次未裁剪梯度范数约 `703.66`；首步 Adam 更新相对 RMS 差 `0.2763` | 有实际影响但证据冲突；排在 scheduler 后、4.37.2 前做匹配短对照 |
| 多节点 `agent.num_gpus` 字段 | 旧命令虽显示 8，但显式 `effective_global_batch_size=16` 已决定优化语义；当前主跑两个字段均显式为 16 | 排除为根因；launcher 已消除歧义 |
| BF16/TF32 | 发布配置和 Stage-1 元数据均为 BF16、非 FP16；action-head FP32 参数在 BF16 autocast 下训练，当前 TF32 关闭 | BF16 已匹配；TF32 与 H20/A800 kernel 仅列为低优先级残余 |
| checkpoint 中 VLM dtype 分布 | 320 个 dtype 不同 tensor 全是冻结 LoRA；BF16 提升到 FP32 后 3,358,720 个值逐位相同，最大误差 0 | 仅存储格式，排除 |
| norm/bias weight decay | 只影响约 0.47% action-head 参数，复现路径已恢复发布行为 | 次要 |
| 样本顺序 | 按用户要求不作为关键根因继续投入实验 | 不作为主线 |

## Proposal 上界分解

公开权重和本地权重使用同一个 18,179-scene navtrain validation、同一个
PDM scorer 和同一套 metric cache。本地列为 epoch 25 的 best checkpoint，不是较差的
last checkpoint。

| checkpoint | selected PDMS | best-of-64 | scorer regret | selected L2 |
| --- | ---: | ---: | ---: | ---: |
| 公开 `epoch_26-step_174312` | 0.951474 | 0.988530 | 0.037056 | 0.565385 |
| 本地 `epoch=25-step=167856` | 0.938030 | 0.974710 | 0.036681 | 0.484313 |
| 公开 - 本地 | +0.013444 | +0.013820 | +0.000375 | +0.081073 |

best-of-64 差距比 selected 差距还大，而 scorer regret 几乎一致。因此不应该继续
把主要精力放在 scorer 或 PDM 标签管线。同时，本地 selected L2 更低、规划
分数反而更低，说明纯 GT 模仿误差不能保证 proposal bank 的规划覆盖。这与本地
有效 action-head 参数位移 RMS 约为公开权重 1.90 倍的观察一致：常数高 LR、
Flash 数值路径或错误的 rank 布局都可能让轨迹分支过度追逐 GT、降低候选覆盖。

完整训练曲线进一步排除了“只是最终 checkpoint 偶然较差”的解释。proposal
ceiling 在 epoch 2 达到本地峰值 `0.983240`，比公开最终 ceiling 仍低
`0.005290`；随后到本地 selected 最优 checkpoint 已回落 `0.008530`，到最后一轮
总计回落 `0.016789`。同一期间 selected PDMS 从 `0.899419` 上升到
`0.931586`，scorer regret 从 `0.083822` 降到 `0.034866`，而 L2 继续改善。
因此训练目标正在提升 GT 拟合和候选选择，却牺牲 64 条候选的规划覆盖；学习率/
scheduler 和会改变梯度路径的 Flash/rank/runtime 设置必须以 proposal ceiling
而不是只看 selected PDMS 或 L2 来筛选。

Navtest 的分项也指向轨迹质量，而不是单一 scorer 校准问题。公开权重相对本地
权重的主要改善是 progress `+0.014864` 和 TTC `+0.009386`；DAC 反而低
`0.001647`，comfort 基本一致。

| Navtest 分项 | 公开 | 本地 | 公开 - 本地 |
| --- | ---: | ---: | ---: |
| PDMS | 0.909594 | 0.899889 | +0.009705 |
| collision | 0.982216 | 0.980693 | +0.001523 |
| DAC | 0.972584 | 0.974230 | -0.001647 |
| DDC | 0.972872 | 0.971925 | +0.000947 |
| progress | 0.884715 | 0.869851 | +0.014864 |
| TTC | 0.942039 | 0.932653 | +0.009386 |
| comfort | 0.999835 | 0.999918 | -0.000082 |

## Checkpoint 初始化指纹

发布 loss 配置为 `prev_weight=0`、`inter_weight=0`。`proposal_list` 的循环会使
trajectory loss 最终只保留最后一层 proposal loss；scorer 也只消费最后一层。
因此 `traj_head.0` 到 `traj_head.3` 的 5,365,856 个参数在完整训练后仍等于初始化，
可用于恢复真实初始化 seed。

| checkpoint | 精确匹配 seed | 匹配 tensor | 最大绝对误差 |
| --- | ---: | ---: | ---: |
| 开源 `epoch_26-step_174312` | 2 | 40/40 | 0 |
| 本地 best epoch 25 | 0 | 40/40 | 0 |
| 本地 last epoch 26 | 0 | 40/40 | 0 |

复现命令：

```bash
python local_stage2/audit_stage2_initialization_fingerprint.py \
  --output /mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/numerics/initialization_fingerprint.json
```

## 1,000-step 受控结果

所有分数使用同一 2,048-scene 验证子集；`best64` 是候选集的离线 oracle 上界，
`regret = best64 - selected`。

| 配置 | 验证 PDMS | best64 | regret | collision | TTC | progress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eager, seed 0 | 0.844435 | 0.988603 | 0.144168 | 0.976807 | 0.900391 | 0.758811 |
| eager, seed 0 repeat | 0.838234 | 0.989899 | 0.151664 | 0.972168 | 0.903320 | 0.748189 |
| Flash, seed 0 | 0.805835 | 0.989746 | 0.183911 | 0.975586 | 0.892578 | 0.695852 |
| eager, seed 1 | 0.822581 | 0.980014 | 0.157433 | 0.974121 | 0.892090 | 0.702922 |
| eager, seed 2, LR `1e-4`，vla-zt2 旧 runtime | 0.826527 | 0.986674 | 0.160147 | 0.961670 | 0.904297 | 0.734752 |
| eager, seed 2, LR `5e-5` | 0.762258 | 0.969300 | 0.207042 | 0.929443 | 0.853516 | 0.674153 |
| eager, seed 2, LR `1e-4`, Lightning 2.5.1 | 0.787970 | 0.980317 | 0.192347 | 0.935059 | 0.878418 | 0.709936 |
| eager, seed 2, 发布 YAML 的实际 LR `2.5e-4`, Lightning 2.6.0 | 0.825879 | 0.983687 | 0.157808 | 0.957275 | 0.901855 | 0.717148 |
| eager, seed 2, LR `1e-4`, Lightning 2.6.0, 4×4 | 0.839999 | 0.989416 | 0.149417 | 0.975342 | 0.904297 | 0.743615 |
| eager, seed 2, LR `1e-4`, Lightning 2.6.0, 8×2 | 0.794061 | 0.980500 | 0.186438 | 0.947998 | 0.856934 | 0.719650 |
| eager, seed 2, LR `1e-4`, Lightning 2.6.0, 16×1 | 0.807793 | 0.990962 | 0.183169 | 0.962891 | 0.895508 | 0.704854 |
| eager, seed 2, LR `1e-4`, Lightning 2.5.1, 16×1 | 0.812480 | 0.985802 | 0.173321 | 0.957764 | 0.879395 | 0.707391 |
| eager, seed 2, LR `1e-4`, Lightning 2.2.1, 16×1, 4 秒 target | 0.801635 | 0.985965 | 0.184330 | 0.938477 | 0.883789 | 0.708672 |
| eager, seed 2, LR `1e-4`, Lightning 2.2.1, 16×1, long-2 target | 0.752680 | 0.988884 | 0.236204 | 0.903809 | 0.805664 | 0.690599 |

`5e-5` 在 1,000 step 显著欠拟合；`2.5e-4` 也没有获得有意义的早期增益。因此短
实验不支持只修改常数 LR。warmup/cosine 不是“更小常数 LR”：它先达到论文报告的
peak LR，再逐步减小累计更新，必须用完整 27-epoch 曲线验证。

上表最后两行使用短跑 validation 的整体均值；为得到逐场景可配对的置信区间，另在
严格相同的 128 个场景上成对重评。该成对审计得到
long-2 的 best-of-64 增益 `+0.033601`（95% CI `[+0.006762,+0.065910]`），是本文
对 long target 因果判断的主要短程依据，而不是最后一行尚未收敛的 selected PDMS。

## 严格 16×1 复现路径

公开 checkpoint 的 step 数已经把全局 batch 锁定为 16；论文又报告 16 张 H20。
因此使用本机和 vla-zt2 各 8 张 A800 运行 16 rank × batch 1，而不再把
8×2 当作官方等价设置。当前锁定配置为：

```text
GPUs: 16 x NVIDIA A800-SXM4-80GB (2 nodes)
global batch: 16 (16 x 1)
seed: 2
attention: eager
precision: bf16-mixed
frozen VLM mode: train
AdamW: betas=(0.9, 0.95), weight_decay=1e-4 on all action-head tensors
LR: peak 1e-4, 10% linear warmup (17431 steps), then cosine decay to zero
runtime: PyTorch 2.5.1+cu124, Lightning 2.2.1, Transformers 4.48.3（当前主跑）
trajectory targets: 4 秒 GT + 5 秒 logged-future 重采样 auxiliary target
epochs/steps: 27 / 174312
```

当前活跃命令同时显式记录 `agent.num_gpus=16` 与
`effective_global_batch_size=16`，并在两端启动前逐值校验 runtime，避免再出现
world-size 字段歧义。实际训练是 16 rank × 1。

Stage-1 基座产物记录 Transformers 4.37.2，但血缘审计证明该字段不能锁定后来
DriveVLA LoRA 的运行时；因此保持 4.48.3 主跑，同时把 4.37.2 降为次级对照。
Lightning 2.2.1 和 2.5.1 的 1,000-step 对照均已在相同两台主机上完成；2.2.1
是历史 checkpoint metadata 的直接事实，因此 27 epoch 全曲线锁定 2.2.1。

两次短跑从完全相同的 seed-2 初始化开始。训练 1,000 step 后，全部有效
action-head 参数的更新 RMS 分别为 `0.0031865` 和 `0.0032139`，范数比
`0.9915`，但 cosine 仅 `0.5681`、同号比例 `68.58%`。这说明版本差异主要改变
更新方向而非简单放大学习率；不能通过等比例调 LR 消除。

## 判定标准

- 当前 long-2 + source-cosine 完整曲线是主判据；只有 full navtrain validation 的 selected
  PDMS、best-of-64、regret 与最终 12,146-scene Navtest 同时接近公开权重，才算复现。
- `2.5e-4` 的短实验没有明显更优，暂不占用完整曲线资源。
- Lightning 已由原始 artifact 锁定为 2.2.1；2.2.1/2.5.1 的短对照只用于量化影响。
- 若 long-2 主跑闭合 proposal ceiling 和 PDMS，则首要根因判为部署 YAML 隐藏了
  私有训练启用的 long target。官方逐 step scheduler 已由 loop state 确认；完整
  曲线与 checkpoint 更新幅度用于判断其是否就是源码 warmup-cosine 及 peak `1e-4`。
- 若 clip=1 相对同布局 clip=0 明显改善 proposal ceiling，再升级为完整曲线；不能
  用上游 ReCogDrive 默认值直接覆盖 DriveVLA 发布配置。
- 若 4.37.2 短训练在 proposal ceiling 或公开 checkpoint 更新方向上显著优于
  同布局 4.48.3，则再决定是否运行 4.37.2 source-cosine 全曲线；否则保留当前
  4.48.3 主跑。
- 若版本对照与 cosine 完整曲线仍不能闭合公开权重，下一优先级才是同一运行时下的
  恒定 `5e-5` 完整对照，然后是 H20/A800 训练 kernel 的不可消除差异；不会回到
  sample 顺序作为主解释。

轻量审计结果保存在：

```text
/mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/
```

大型 checkpoint 只保存在实验目录，不提交 Git。

代码交付验证使用与训练一致的锁定运行时完成：`pytest -q tests` 为
`69 passed, 19 warnings`。默认交互 shell 的旧 `navsim` 环境缺少 `peft`，会在
测试收集阶段报 `ModuleNotFoundError`；这是环境依赖缺口，不是本次测试失败，也未
用于训练进程。
