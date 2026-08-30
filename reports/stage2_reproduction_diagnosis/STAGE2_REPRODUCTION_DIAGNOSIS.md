# DriveVLA-M0 Stage-2 复现差异诊断

审计日期：2026-08-30  
基准分支：`fix/stage2-official-reproduction`  
基准提交：`6e96cf7321b134c42c2cf0fbbc315cd61c925b11`

## 当前结论

本地已完成模型的 Navtest PDMS 为 `0.8998889219`，开源 Base 权重在完全相同的
12,146 个场景和本地评测链路上为 `0.9095938788`，差值为 `-0.0097049569`。

差值不是由样本顺序或本地 evaluator 引起。现有证据按重要性给出下列结论：

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
6. **实际学习率和 scheduler 是剩余最重要的配置歧义。** 论文附录明确报告 base model
   使用 AdamW、学习率 `1e-4` 和 16 张 H20；但发布代码会把 `base_lr` 再乘
   `sqrt(global_batch/base_batch)`。论文没有说明 `1e-4` 是缩放前的配置字段还是
   缩放后的 optimizer-group LR。global batch 为 16、`base_batch_size=64` 时，两种
   解读分别得到实际 LR `5e-5` 和 `1e-4`。发布 YAML 自身的 `base_lr=5e-4` 又会得到
   `2.5e-4`；其未随 launcher 改动的 `agent.batch_size=2,num_gpus=1` 则会得到
   `8.84e-5`。本机 PL 2.6 的 `2.5e-4` 在 1,000 step 得到 `0.8258788`，没有显示
   实质早期优势；`5e-5` 在 1,000 step 欠拟合，但这不能替代 27 epoch 的完整曲线。
   scheduler 同样未锁定：发布 YAML 明确
   `scheduler_args: null`，论文没有报告 warmup/cosine，因此不能把 scheduler 写成
   官方事实；发布源码确实保留了完整的 10% warmup + cosine 分支。开源权重相对
   seed-2 初始化的有效 action-head RMS 位移为 `0.0219354`，本地旧权重相对 seed-0
   初始化为 `0.0416902`，比例为 `0.5262`；逐模块比例也集中在 `0.48--0.66`。
   常数 `5e-5` 与 peak `1e-4` 的 cosine 都约有常数 `1e-4` 一半的累计 LR，单凭
   checkpoint 位移无法区分两者。严格 16×1 的恒定 `1e-4` 已完成一整轮，其位移
   外推明显超过公开权重；结合 DriveVLA-M0/DrivoR/ReCogDrive 的源码沿袭关系，当前
   优先完整验证 peak `1e-4` 的源码 warmup-cosine，恒定 `5e-5` 保留为失败后的首个
   完整对照。这个排序是基于现有证据选择实验，并不把未公开 scheduler 当成官方事实。
7. **冻结 VLM 的 train/eval mode 是次要因素。** 1,000-step A/B 中，eval-mode
   反而比 train-mode 高 `0.00141` PDMS；差异很小且方向不能解释旧本地结果偏低。
8. **Lightning 版本不能被当作一个无关细节，但也不是单独主因。** requirements
   中只留下了注释形式的 `pytorch-lightning==2.5.1`；本机默认是 `2.6.0`。严格
   16×1、seed-2/eager/实际 LR `1e-4` 的 1,000-step 对照中，2.5.1 和 2.6.0
   分别得到 `0.8124804` 和 `0.8077928`，差 `+0.0046877`。两者 action-head
   更新 RMS 几乎相同（`0.003187`/`0.003214`），但更新向量 cosine 只有
   `0.5681`；框架版本会改变优化路径，却不能用这次短跑证明其方向在完整训练中
   恒定有利。此前单机 4×4 对照甚至呈相反方向，说明 1,000-step PDMS 的噪声和
   路径依赖都很强。已确认当前 2.6 安装的关键文件与官方 PyPI 2.6 wheel 逐字节
   一致，不是本地加速补丁；2.5.1/2.6.0 应归类为未锁定的官方运行时语义。由于
   仓库 requirements 唯一留下的版本证据是 2.5.1，完整严格复现选用 2.5.1。

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

因此，现阶段最严格的表述是：**旧本地训练不是官方 Stage-2 的等价复现。
它的最终分数损失主要来自 proposal bank 上界下降；Flash Attention 是已确认的
有害语义改动；8×2 而非 16×1、seed 0 而非 seed 2 也是与公开 checkpoint
证据不符的确定配置错误。剩余需用完整 27 epoch 收口的是实际 LR/scheduler，
其次是梯度裁剪、Transformers/Lightning 运行时及 H20/A800 数值路径。** sample
顺序已按用户要求移出主因验证队列。

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
4. **完整修正 run 已启动。** 当前实验
   `stage2_official_source_cosine_seed2_eager_tf448_pl251_16x1` 使用本机与
   `vla-zt2` 共 16 张 A800，锁定 PyTorch `2.5.1+cu124`、Lightning `2.5.1`、
   Transformers `4.48.3`、eager attention、BF16、seed 2、16 rank × batch 1，
   并启用发布源码中逐 step 等价的 10% linear warmup + cosine decay。完整 27 epoch
   和 Navtest 结果才是 scheduler 假设的最终判据。
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
7. **rl-zt3 对照已改为严格匹配的顺序队列。** 4×1×acc4 虽重放相同全局样本，
   但每 rank 的 dropout/RNG 流与 16×1 不同，不能把单个 4.37 结果直接和主跑归因。
   队列因此依次执行 `TF4.48/PEFT0.10/clip0`、
   `TF4.48/PEFT0.10/clip1`、`TF4.37/PEFT0.10/clip0`，三次使用相同布局、seed、
   cosine schedule 和验证子集。前两次只归因梯度裁剪，第一和第三次只归因
   Transformers；服务恢复后会自动在用户授权的 GPU 3/5/6/7 上顺序运行。
8. **vla-zt2 主跑的完整 epoch-0 验证符合 warmup 预期。** 16×1 主跑在完整
   18,179-scene navtrain validation 上得到 selected PDMS `0.718126`、
   best-of-64 `0.966110`、regret `0.247985` 和 L2 `1.444121`。同期实际 LR 仅
   `3.6997e-5`，所以 scorer 与 GT 拟合明显落后于恒定 `1e-4` 对照是预期现象；
   更关键的是 proposal ceiling 与恒定-LR epoch-0 的 `0.966837` 只差
   `0.000727`。该结果说明 warmup 第一轮没有提前损坏候选覆盖，但不能用来宣称
   已复现最终性能。主判据仍是 warmup 结束附近的 epoch 2--3，以及后续 ceiling
   是否避免旧实验从 epoch 2 到 epoch 26 的持续塌缩。

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

`5e-5` 在 1,000 step 显著欠拟合；`2.5e-4` 也没有获得有意义的早期增益。因此短
实验不支持只修改常数 LR。warmup/cosine 不是“更小常数 LR”：它先达到论文报告的
peak LR，再逐步减小累计更新，必须用完整 27-epoch 曲线验证。

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
runtime: PyTorch 2.5.1+cu124, Lightning 2.5.1, Transformers 4.48.3（当前主跑）
epochs/steps: 27 / 174312
```

Stage-1 基座产物记录 Transformers 4.37.2，但血缘审计证明该字段不能锁定后来
DriveVLA LoRA 的运行时；因此保持 4.48.3 主跑，同时用 rl-zt3 的授权 GPU 先做
clip=0/1、再做 4.37/4.48 的同布局短训练筛选。Lightning 2.6.0 和 2.5.1 的
1,000-step 对照均已在相同两台主机上完成。2.5.1
的验证 PDMS 高 `0.004688`，但更重要的是 requirements 唯一留下的版本证据指向
2.5.1；因此 27 epoch 全曲线锁定 2.5.1。该选择仍是“最有证据的复现假设”，不是
对未发布私有环境的事实声明。

两次短跑从完全相同的 seed-2 初始化开始。训练 1,000 step 后，全部有效
action-head 参数的更新 RMS 分别为 `0.0031865` 和 `0.0032139`，范数比
`0.9915`，但 cosine 仅 `0.5681`、同号比例 `68.58%`。这说明版本差异主要改变
更新方向而非简单放大学习率；不能通过等比例调 LR 消除。

## 判定标准

- 当前 source-cosine 完整曲线是主判据；只有 full navtrain validation 的 selected
  PDMS、best-of-64、regret 与最终 12,146-scene Navtest 同时接近公开权重，才算复现。
- `2.5e-4` 的短实验没有明显更优，暂不占用完整曲线资源。
- Lightning 2.5.1/2.6.0 需在同一 16×1 布局上比较；4×4 结果不用来代替该对照。
- 若 cosine 闭合公开 checkpoint 的更新幅度和 PDMS，主因判为私有 launcher 很可能
  启用了发布源码中存在但 YAML 未启用的 schedule。
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
`51 passed, 16 warnings`。默认交互 shell 的旧 `navsim` 环境缺少 `peft`，会在
测试收集阶段报 `ModuleNotFoundError`；这是环境依赖缺口，不是本次测试失败，也未
用于训练进程。
