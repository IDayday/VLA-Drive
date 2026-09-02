# No-VQA scorer-private 表征改进计划

更新时间：2026-09-02 UTC

## 研究目标

固定本地 No-VQA epoch 35 已收敛的候选生成器，专门提高 scorer 的当前观测
表征与候选选择能力。最终目标是纯 M0 路径、完整 Navtest selected PDMS
超过 `0.93`，不依赖 DrivOR 或其他驾驶模型的表征、scorer 或权重。

本阶段不再把资源用于判断“表征还是 scorer 架构”的模块交换归因。架构只
作为实现表征学习的载体；判定标准是固定候选后的选择能力和最终 PDMS。

## 已知基线与可用上限

No-VQA epoch 35 的严格完整 Navtest 结果：

| 指标 | 数值 |
|---|---:|
| selected PDMS | 0.911493 |
| Best-of-64 oracle | 0.983129 |
| scorer regret | 0.071635 |
| 64 候选平均 PDMS | 0.810235 |

因此达到 `0.93` 需要相对 No-VQA 基线提升约 `0.018507`，等价于回收当前
scorer regret 的约 `25.8%`。候选库上限不是眼前 blocker；主要任务是从
现有 64 条候选中更稳定地识别高质量尾部。

基线 checkpoint：

```text
/mnt/project/DriveVLA-M0-no-vqa/runs/training/
no_vqa_full_ft_seed0_e36/lightning_logs/version_0/checkpoints/
best-epoch=35-step=232416.ckpt
```

SHA256：

```text
72c74a113c557df27c86a320f66d4ff2a79fc1a19e678337d5a142a520359309
```

完整基线证据见
[M0_V8_CORRECTED_NOVQA_COMPARISON.md](M0_V8_CORRECTED_NOVQA_COMPARISON.md)。

## 固定项

以下内容在第一阶段保持不变：

- No-VQA 视觉语言 backbone；
- No-VQA trajectory decoder 与 64 条 proposals；
- 8 个未来 pose、4 秒 horizon；
- 当前图像、当前 ego 状态、历史/导航 prompt 等原模型推理输入；
- official PDM 只生成离线训练标签和最终评价，不进入模型 forward；
- 完整物理日志隔离的 train/validation 划分；
- Navtest 不用于选 epoch、调损失权重或选择架构。

禁止的推理输入：future image、future annotation、future GT trajectory、
MetricCache、official factor/score 和 DrivOR feature/weight。

## 第一优先方案：No-VQA scene-token semantic refiner

输入为 No-VQA 在当前观测上产生的 16 个 scene tokens、ego token、64 条固定
proposals、原 scorer-attention 为每条 proposal 产生的 256 维候选 hidden state、
原六个 factor logits 和原 aggregate score。候选 hidden state 来自同一冻结
No-VQA forward，仅依赖当前观测和候选轨迹；它不是 DrivOR 表征，也不包含
future/MetricCache/PDM 标签。新增分支包含：

1. scorer-private dynamic/static/signal/global query banks；
2. query 对 No-VQA scene tokens 的交叉注意力；
3. proposal geometry encoder；
4. candidate-to-scene interaction；
5. scorer-private candidate feature 与原 No-VQA candidate hidden state 融合；
6. 对原 No-VQA factor logits 和 score 的零初始化 residual。

零初始化要求训练开始时严格复现 No-VQA 的选择。新分支只做 scorer 表征和
排序，不更新 trajectory generator。

训练目标按重要性排序：

1. 六个官方 factor 的 source-equivalent BCE；
2. 同场景、非平分候选对的 RankNet loss；
3. top-set / expected-regret 目标；
4. 当前 actor presence/type/position/velocity/heading/size 的训练期辅助监督；
5. candidate-relative dynamic risk 作为后续消融，不先压过基础排序目标。

当前 actor 标注只作为训练 target；模型推理仍只读当前观测。

## 第二优先方案：No-VQA spatial visual-token refiner

若 16 个 scene tokens 在验证集上仍不足，再从同一 No-VQA 视觉 backbone 缓存
更高空间分辨率的当前四相机 token，并与原 16 个 scene tokens 双流融合。
优先比较 `2×2` 与 `4×4` 每 crop pooling；不引入新视觉模型，也不使用
LoRA。该方案比第一方案开销更大，只在第一方案留下明确表示瓶颈时启动。

## 第三优先方案：scorer-only 深层表征继续训练

若缓存 token refiner 有正向但不足的增益，再考虑为 scorer 复制 No-VQA
Q-Former/末端轻量层并继续训练。proposal 路径必须使用冻结副本，确保 scorer
梯度不会改变候选。不会直接复制或微调整个 2B backbone，也不会让 proposal
和 scorer 的变化重新混在一起。

## 数据与缓存计划

训练集使用完整合法 trainval feature cache：

```text
/mnt/project/DriveVLA-M0-stage2/cache/feature_cache_navtrain_full
```

需要新建两棵严格分离的缓存：

```text
/root/scorer_pdms93_cache/no_vqa_e35_features_full_v1
/root/scorer_pdms93_cache/no_vqa_e35_labels_full_v1
```

feature cache 只含部署时可获得的 proposals、scene/ego tokens、factor logits 和
base scores。label cache 由训练 MetricCache 离线生成，永远不进入模型输入。
两者按 scene token 对齐并分别记录 manifest 与 SHA256。

Navtest 使用独立输出目录和全新候选矩阵，不能把数值略有差异的导出缓存与旧
候选 PDM 矩阵混接。任何 online/cache proposal 差异都必须先审计并记录。

## 预注册选择门槛

模型只依据物理日志隔离 validation 选择。进入完整 Navtest 至少要求：

- validation selected PDMS 高于同一候选 bank 的 No-VQA base；
- 日志级 bootstrap 差值下界不为明显负值；
- Top-1 regret 下降；
- collision/TTC 不出现明显退化；
- candidate order permutation 测试通过；
- current-only、official-score exclusion 和 fixed-proposal 断言通过。

每个满足门槛的预注册方法都做一次完整 FP32 Navtest。最终成功标准：

```text
12,146 scenes
136 segment logs
64 candidates per scene
0 invalid scenes
selected PDMS > 0.93
```

## 当前状态

- No-VQA checkpoint、SHA 和旧完整 Navtest 基线已锁定；
- 8-scene No-VQA trainval feature-export smoke 已通过；
- smoke 通过后并行导出完整 source cache，并同步进行 CPU PDM label 生成；
- 第一版训练直接使用 No-VQA 16 scene tokens，避免先支付高分辨率视觉缓存
  的成本；
- 所有运行命令、manifest、验证和 Navtest 结果将在本文件或独立结果 Markdown
  中追加，避免后续重新检索。

## 2026-09-02 feature-export smoke 证据

第一次 smoke 暴露了真实的 checkpoint/config 身份问题，而不是数据问题：仓库
默认 `episode_drive` 配置会开启 LoRA，但 No-VQA epoch 35 是按训练时保存配置
进行的全量微调，`agent.vlm_config.lora_config.use_lora=false`。因此，不能只把
No-VQA checkpoint 路径覆盖到默认配置上；这样构造的是另一套网络结构，严格
加载会失败。

修复后，导出器显式读取 No-VQA 训练目录保存的 resolved Hydra config：

```text
/mnt/project/DriveVLA-M0-no-vqa/runs/training/
no_vqa_full_ft_seed0_e36/code/hydra/config.yaml
```

其 SHA256 为：

```text
5f70b74293883bebb80fc1feffaf3786556f909645a248374495dfadbf7cd1c3
```

只覆盖推理所需的安全字段：checkpoint 路径、禁用 stage-1 checkpoint、冻结
backbone、`eval` mode、关闭 FlashAttention/gradient checkpointing，并打开
scorer feature 返回；LoRA、action decoder 和其余架构字段保持训练配置原值。

修复后的 8-scene smoke 产物：

```text
/root/scorer_pdms93_cache/no_vqa_e35_source_smoke_v1/
all_shard_000-of-001/
```

实测结果：

- 8 scenes / 8 physical logs；
- checkpoint SHA256 与锁定值
  `72c74a113c557df27c86a320f66d4ff2a79fc1a19e678337d5a142a520359309`
  一致；
- resolved config SHA256 与上述值一致；
- 8 个场景实际前向耗时约 `3.4 s`（不含模型加载）；
- 每个场景导出 64 条 proposal、原 factor logits、原 aggregate score、
  candidate feature、16 个 current-scene token 与 ego token；
- manifest 明确记录 `inference_inputs_only=true`、
  `official_score_present=false`、`future_target_present=false`；
- 推理配置保持 `use_lora=false`、eager attention、冻结 VLM eval。

覆盖 checkpoint 文件名含 `epoch=...-step=...` 的 Hydra quoting 和 resolved
No-LoRA 配置保持性的单元测试均已通过。全量导出必须沿用这一路径；此前用默认
LoRA 配置的失败 smoke 不得作为训练输入，也不得静默忽略。

## 2026-09-02 全量缓存与首轮训练启动

全量流水线已经在本机 8 张 A800 上启动：

```text
launcher PID: 448227
source: /root/scorer_pdms93_cache/no_vqa_e35_features_full_v1
labels: /root/scorer_pdms93_cache/no_vqa_e35_labels_full_v1
logs: /root/scorer_pdms93_logs/no_vqa_e35_cache_full_v1
```

运行拓扑为 8 个 GPU export shards，每卡 batch size 2、两个 dataloader
workers；PDM 标签由 4 个独立 watcher、每个 12 个进程生成，总计占用 48 个
物理 CPU core。机器有 64 个物理 core，其余 16 个留给 8 个 GPU loader。
feature/label 以相同相对 chunk 路径保存，但位于物理分离的目录。

首轮稳定速度为每 shard 每 128 scenes 约 35 秒。首次进度检查时，8 个 shard
均完成 `11/101` chunks（每 shard 1,408 scenes），GPU 利用率为 99–100%；
source/label chunks 分别为 88/64，说明离线标签生成可以跟上 GPU 导出，且没有
出现单卡掉队。

缓存完成后会先运行 `verify_no_vqa_scorer_cache.py`，硬检查：

- 103,288 个唯一 trainval scenes；
- 每场景恰好 64 proposals；
- 每场景恰好 `64 × 256` 个冻结 No-VQA candidate hidden states；
- 8 个 source shards 和 4 个 label workers 全部完成；
- checkpoint/config SHA 完全一致；
- feature/label chunk、token、log 顺序逐行一致；
- Base selected index 等于 `argmax(base_scores)`；
- 全部 tensor finite、PDM valid mask 全真；
- source cache 不含 future/evaluator target。

通过后，8 卡并行运行以下预注册实验；所有实验固定 No-VQA proposals 和基础
VLM/trajectory generator，只训练新增 scorer-private 模块。首轮八个实验均融合
冻结 No-VQA scorer-attention 的 `candidate_features [B,64,256]`，以直接检验
“scorer 需要自己的候选条件 hidden state”这一假设；不同实验只改变下表变量：

| GPU | 实验 | 表征/评分变量 |
|---:|---|---|
| 0 | primary_hybrid_actor050_seed2 | hybrid residual，current-actor auxiliary 0.5，seed 2 |
| 1 | control_hybrid_no_actor_seed2 | 去掉 actor semantic auxiliary |
| 2 | factor_actor050_seed2 | 只由六分项 residual 聚合选择 |
| 3 | direct_actor050_seed2 | 只由直接 utility residual 选择 |
| 4 | primary_hybrid_actor050_seed11 | primary 的第二随机种子 |
| 5 | primary_hybrid_actor050_seed23 | primary 的第三随机种子 |
| 6 | hybrid_actor050_deep_seed2 | private/residual Transformer 各加深到 3 层 |
| 7 | hybrid_actor050_future025_seed2 | 增加 shared logged-future actor auxiliary 0.25，仅作训练 target |

共同配置为完整 85,109 train / 18,179 validation scenes、物理日志完全隔离、
8 epochs、batch size 32、eager deterministic attention。current actor 和
logged future 均只作为训练期 auxiliary target，forward signature 不接受这些
字段。

对应 watcher：

```text
PID: 455360
script: local_stage2/watch_no_vqa_e35_cache_and_train_scene_token_scorers.sh
```

同一 watcher 会并行离线评分与新 FP32 No-VQA scene-token cache 数值完全匹配的
完整 Navtest 64-proposal bank。旧 No-VQA candidate matrix 与这次 feature
cache 的 proposal 存在可测数值差异，因此明确禁止跨 bank 拼接标签。

训练后另外生成每个 artifact 的保守校准版本。61 条 held-out 物理日志被固定
种子平衡拆成 calibration/promotion 两半：前一半只选择 residual scale、非 Base
候选切换惩罚和预测安全门，后一半独立计算 bootstrap 区间。原始版本与校准版本
分别按各自的 held-out 证据晋级；所有正区间 artifact 都必须完整跑 Navtest，
不能根据 Navtest 结果回调门限。

授权的 `training-vla-zt2` 八卡为空闲状态，因此第二波在完全相同的冻结 No-VQA
cache 上预注册两类更保守实验：`m0_candidate_only` 只用原 scorer-attention 的
候选 hidden state 与 Base factor context 学 residual；另一类保留完整 private
scene/candidate interaction，但只在 Base top-4、top-8 或 top-16 内重排。这样可
分别检验“专用候选 hidden state”以及“all-64 过度换轨”两个高优先级假设。
第二波同样做日志二分保守校准，且只有 held-out bootstrap 下界为正的原始/校准
artifact 才允许进入完整 Navtest。

## Scene-token 在线部署路径修复

旧的 `M0NativePrivateScorerAgent` 只支持第二次运行 M0 vision tower 来提取四相机
空间 token，无法忠实部署第一优先的 16-token semantic refiner。现在 adapter、
packager 和 cached Navtest evaluator 已增加：

```text
private_observation_source = source_checkpoint_current_scene_tokens
```

该模式直接消费同一次冻结 No-VQA forward 返回的 `language_feature [B,16,256]`、
`ego_feature [B,1,256]` 和 `scorer_candidate_features [B,64,256]`，不会再运行
第二个视觉分支。打包时强制检查训练 source checkpoint SHA 与基础 No-VQA
checkpoint SHA 相同，并声明
`private_vision_config=null`。相关打包、零初始化 Base 等价性与候选置换等变性
路径已纳入测试。

## 2026-09-02 全量缓存验收与三波并行实验

全量 No-VQA epoch-35 replay 缓存已经完成并通过 fail-closed 验收：

```text
scene_count: 103288
unique_scene_count: 103288
candidate_count: 64
source shards: 8
source chunks: 808
label workers: 4
invalid scenes: 0
future_or_official_input_present: false
status: PASS
```

机器可读证据为
`reports/m0_independent_scorer_representation/no_vqa_e35_scene_token_v1/CACHE_VERIFICATION.json`。
缓存同时锁定 No-VQA checkpoint SHA256
`72c74a113c557df27c86a320f66d4ff2a79fc1a19e678337d5a142a520359309`
和 resolved config SHA256
`5f70b74293883bebb80fc1feffaf3786556f909645a248374495dfadbf7cd1c3`。

在用户确认 `vla-zt`、`vla-zt2` 已空闲并授权 `rl-zt4` 后，实验扩为三波：

1. 本机 8 卡：private current-scene representation、direct/factor/hybrid、
   current-actor auxiliary、三随机种子、深层网络和 shared-future auxiliary；
2. `training-vla-zt2` 8 卡：冻结 No-VQA candidate hidden 的低容量对照，及
   Base top-4/top-8/top-16 保守重排；
3. `training-rl-zt4` 仅使用启动时确认空闲的 GPU 1/3/5/6/7：
   `SharedFutureFactorized` 候选相对重计算、训练集预注册的 safety-negative
   权重 1/5 对照，以及 candidate-only factor 对照。GPU 0/2/4 未被本实验占用。

第三波的 factorized 路径预测一次当前 ego frame 下的共享 actor future，然后
用同一个可微 relabeler 对 64 条候选重算 clearance、soft collision、soft TTC、
corridor occupancy 和相对 actor state。logged future 只生成训练 target；模型
forward 不接收 future、MetricCache 或 official score。对应可复现入口为：

```text
local_stage2/run_no_vqa_e35_scene_token_wave3_remote.sh
local_stage2/launch_no_vqa_e35_wave3_on_rl_zt4.sh
local_stage2/watch_no_vqa_e35_wave3_and_run_navtest_remote.sh
```

三波均先按 101/61 条物理日志的固定 train/validation split 训练，再把 61 条
held-out logs 确定性拆成 calibration/promotion 两半。只有 promotion 半的日志级
bootstrap 下界为正的 raw 或 calibrated artifact 才进入完整 Navtest；所有晋级
artifact 都必须做 12,146-scene FP32 Navtest 和四场景 online/cache parity。

本轮新增与相关回归测试当前为 `148 passed`；完整项目测试将在实验产物稳定后
再次运行并单独记录。

## 2026-09-02 Wave-6/7：原生四视角空间表征与路径局部注意力

Wave-5 已证明 Base-relative conservative gate 可以把测试退化压到接近零，但
16 个冻结 Q-Former scene token 上的最好 Navtest 仅为 `0.911499`，相对 Base
只有 `+0.000006` 且只改变 1 个 scene。验证提升最大的模型在 Navtest 仍翻转为
负值。因此下一步不再增加相同 scene-token head 的容量，而是保留同一个 No-VQA
checkpoint，导出当前时刻 `CAM_F0/L0/R0/B0` 的 80 个空间 token。

Wave-6 的八个配置使用已有 query-bank 压缩路径。Wave-7 在其上增加一个严格
可选、默认关闭的 scorer-private 路径：每条 proposal 的 8 个轨迹点直接查询同一
份未压缩当前观测 memory，再在时间维聚合为候选特征。它满足：

- 当前观测 memory 每个 scene 只计算一次，候选不能改变共享视觉表征；
- candidate permutation 时输出同步置换；
- 新路径标量门初始化为 0，关闭时对既有 scorer 是精确 no-op；
- forward 不接收 future、PDM、MetricCache 或 DrivOR 表征；
- Base generator、Q-Former、trajectory decoder 与原 scorer 全部冻结。

Wave-7 在本机 GPU 1--7 预注册七个与 Wave-6 配对的设置，GPU 0 留给既有任务；
这些设置在读取 Wave-6/7 validation 与 Navtest 前已经锁定。入口为：

```text
local_stage2/run_no_vqa_e35_multiview_point_attention_wave7_local.sh
local_stage2/watch_no_vqa_e35_multiview_point_attention_wave7_and_run_navtest_local.sh
```

新增路径及 refit 兼容性相关回归当前为 `108 passed`。完整多视角 trainval/Navtest
cache 通过 `.complete` 和 lineage 验收后，Wave-6 与 Wave-7 才会分别自动启动。

## 2026-09-02 Wave-8：全覆盖当前 actor 表征监督

旧 Gate-C current-actor table 只覆盖 45,378/103,288 个 No-VQA trainval scene，
使 actor-localization auxiliary 在多数 batch 中没有监督。现已用 Scene 当前帧
annotations 构造新的 103,288/103,288 全覆盖表；它只读取 current frame，明确
不依赖 logged future、MetricCache、proposal 或 PDM，且只作为训练 target。

Wave-8 预先固定六个与 Wave-6/7 一一匹配的 conservative-reference 配置：

- pooled raw-token：combined Top-16/Top-32、private Top-16、context+combined Top-16；
- path-local attention：combined Top-16/Top-32；
- actor loss weight 保持 `0.5`，其余 split、seed、epoch、优化器、候选、标签、
  gate 和推理输入均不变。

该实验只回答 actor 监督覆盖是否改善 scorer-private perception，不把 actor
annotation 加入验证或推理。训练入口和自动严格 Navtest 入口分别为：

```text
local_stage2/run_no_vqa_e35_full_current_actor_wave8_remote.sh
local_stage2/watch_no_vqa_e35_full_current_actor_wave8_and_run_navtest_remote.sh
```
