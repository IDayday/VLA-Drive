# M0 V8、Stage2 修正版与 No-VQA 训练/评测对比

更新时间：2026-09-02

本文档固定记录三次本地 M0 训练的训练语义与完整 Navtest 结果，避免后续重复检索和把不同口径的数字混在一起。

## 1. 结论摘要

1. **No-VQA 是三版本中完整 Navtest 最好的一版：0.911493**，高于 V8 的 0.905593，也高于 M0 开源权重的 0.909594，但仍未达到当前研究目标 0.93。
2. **Stage2 修正版不是最终性能改进版。** 它把 Best-of-64 上限从 V8 的 0.968203 提高到 0.975643，却把实际选中 PDMS 降到 0.889563。问题主要表现为 scorer 无法从更分散、更困难的候选中选对。
3. **V8 的候选平均质量最高但多样性最低。** 它产生了一组更集中、更保守的候选，因此 scorer regret 较小，但 Best-of-64 上限受限。
4. **No-VQA 的关键结构差异是：规划损失会更新当前观测 VLM 表征。** V8 和修正版都冻结 VLM，只训练随机初始化的 action head；No-VQA 从基础 InternVL3-2B 出发，联合训练 VLM 表征和 action head。
5. 三次训练同时改变了表征更新、长轨迹目标、学习率、训练时长、FlashAttention、采样器和随机种子，因此这不是严格单变量因果实验。当前证据把“scorer-private 表征学习/持续联合校准”提升为首要假设，但仍需固定同一候选库的模块交换实验确认。

## 2. 严格评测身份与验收

三次结果都采用同一严格口径：

- split：Navtest
- 场景：12,146
- 日志：136
- 每场景候选：64
- 无效场景：0
- GPU 推理精度：FP32
- 评分：候选生成和选择完成后，离线调用官方 PDM 路径
- batch candidate scoring 与单轨迹 scoring 最大误差：0
- `best_of_64_pdms` 仅表示离线候选库 oracle 上限，不是可部署模型成绩

三个审计目录均通过：

```bash
/root/.codex/skills/navsim-scorer-evaluation/scripts/validate_audit.sh <audit_dir>
```

验收结果均为 `passed=true`、`scene_count=12146`、`candidate_count=64`、`invalid_scene_count=0`。

### V8

- checkpoint：`/mnt/project/DriveVLA-M0-stage2/runs/training/stage2_full_seed0_pipeline_v8_restart/lightning_logs/version_0/checkpoints/last.ckpt`
- SHA256：`f5df88d824f977b814ff7ea6c778f82f2b00769c7ac440ee08d99cab5b3a9f21`
- 审计：`/mnt/project/DriveVLA-M0-stage2/runs/ke_candidate_audit/v8_last_resume_gate_fp32_full_20260901_v1/summary.json`
- 被测 checkpoint：epoch 26 训练结束的 `last.ckpt`

### Stage2 修正版

- checkpoint：`/root/drivevla_checkpoints/stage2_official_pl221_tf448_eager_seed2_long2_source_cosine_16x1/resume-20260901T141855Z/best-epoch=25-step=167856.ckpt`
- SHA256：`883169975c3e06929c9187a9b15a915ec282c6901e23cbf0f06aa9404e1e73f6`
- 审计：`/mnt/project/DriveVLA-M0-stage2/runs/ke_candidate_audit/long2_final_epoch25_fp32_export_20260902_v1/cpu_score/summary.json`
- 被测 checkpoint：按本地 validation callback 选择的 epoch 25 best checkpoint
- 注意：这是根据残留 checkpoint、源码分支和步数反推的 Stage2 重建，不是已经获得官方私有 launcher 后的精确复现。

### No-VQA

- checkpoint：`/mnt/project/DriveVLA-M0-no-vqa/runs/training/no_vqa_full_ft_seed0_e36/lightning_logs/version_0/checkpoints/best-epoch=35-step=232416.ckpt`
- SHA256：`72c74a113c557df27c86a320f66d4ff2a79fc1a19e678337d5a142a520359309`
- 审计：`/mnt/project/DriveVLA-M0-stage2/runs/ke_candidate_audit/no_vqa_e35_resume_gate_fp32_full_20260901_v1/summary.json`
- 被测 checkpoint：epoch 35，亦为第 36 个 epoch 结束时 checkpoint
- “No-VQA”表示不使用 Stage-1 VQA checkpoint/辅助 VQA loss，并不表示删除导航语言输入。

## 3. 完整 Navtest 主结果

| 指标 | V8 | Stage2 修正版 | No-VQA |
|---|---:|---:|---:|
| 实际选中 PDMS | 0.905593 | 0.889563 | **0.911493** |
| Best-of-64 oracle | 0.968203 | 0.975643 | **0.983129** |
| Scorer regret | **0.062610** | 0.086080 | 0.071635 |
| 64 候选平均 PDMS | **0.829958** | 0.818789 | 0.810235 |
| 64 候选中位 PDMS | **0.859814** | 0.845341 | 0.849457 |
| Top-5 oracle 平均 | 0.959433 | 0.962964 | **0.973493** |
| 候选 PDMS >= 0.8 | **84.319%** | 81.065% | 79.751% |
| 候选 PDMS >= 0.9 | 64.829% | **66.659%** | 64.129% |
| 选中分数减候选均值 | 0.075635 | 0.070774 | **0.101259** |
| 平均 pairwise ADE | 1.058569 m | **1.595737 m** | 1.226459 m |
| 平均 endpoint 距离 | 2.430984 m | **3.857105 m** | 2.858682 m |

### No-VQA 8 卡训练墙钟时间

No-VQA 使用单机 8 张 A800、每卡 batch size 2、global batch size 16，完整
训练 36 epochs / 232,416 optimizer steps。

从 TensorBoard event 的首个训练 scalar 到最后一个 validation scalar：

```text
first train scalar: 2026-08-27 16:39:31.872 UTC
last validation scalar: 2026-08-31 19:33:58.906 UTC
elapsed: 98 h 54 min 27 s
```

从 launcher 日志开始到最终 checkpoint 落盘：

```text
launcher: 2026-08-27 16:37:28 UTC
checkpoint: 2026-08-31 19:34:09 UTC
elapsed: 98 h 56 min 41 s
```

因此实测应表述为：**8 卡完整 No-VQA 训练约 99 小时，即 4 天 3
小时**。按 TensorBoard 各 epoch 的训练 scalar 区间估算，其中约 `93.4`
小时用于 train step，约 `5.2` 小时用于逐 epoch validation/checkpoint，另有
少量 DDP 初始化与 epoch 切换开销。平均墙钟约 `2 小时 45 分/epoch`；最后
一个 epoch 的训练阶段约 `2:31:20`，随后 validation 约 `8.5` 分钟。

证据文件：

```text
/mnt/project/DriveVLA-M0-no-vqa/runs/training/no_vqa_full_ft_seed0_e36/train.log
/mnt/project/DriveVLA-M0-no-vqa/runs/training/no_vqa_full_ft_seed0_e36/
lightning_logs/version_0/events.out.tfevents.1787848738.training-rl-zt4-worker-0.777222.0
/mnt/project/DriveVLA-M0-no-vqa/runs/training/no_vqa_full_ft_seed0_e36/
lightning_logs/version_0/checkpoints/best-epoch=35-step=232416.ckpt
```

`93.4 / 5.2` 小时是根据每 10 step scalar 与 epoch-end validation 时间戳
分解得到的近似值；`98:54:27` 和 `98:56:41` 是直接时间戳差值。

解释：

- `scorer_regret = Best-of-64 - selected`。它只在同一候选库内部衡量选择损失；跨候选库直接比较时还会受候选难度分布影响。
- V8 的低 regret 部分来自候选更集中，而不是已证明 scorer 表征更强。
- No-VQA 的候选均值低于 V8，但 oracle 高 0.014926，说明它牺牲了“每条都不错”的集中性，换来了更高的 best-of-K 上限。
- No-VQA 的 `selected - mean` 比 V8 高 0.025624，说明它的 scorer 在自身候选库中提供了更大的筛选增益。

## 4. 日志级配对置信区间

以下区间使用 136 个日志作为 cluster、20,000 次 bootstrap；差值定义为前者减后者。

### No-VQA 相对 V8

| 指标 | 差值 | 95% CI |
|---|---:|---:|
| 实际选中 PDMS | **+0.005900** | **[+0.000598, +0.011122]** |
| Best-of-64 | **+0.014926** | **[+0.010568, +0.019404]** |
| Scorer regret | +0.009025 | [+0.004075, +0.014237] |
| 候选平均 PDMS | -0.019724 | [-0.030081, -0.009481] |
| Pairwise ADE | +0.167890 m | [+0.145960, +0.190006] |
| Endpoint 距离 | +0.427699 m | [+0.380150, +0.473568] |

No-VQA 对 V8 的最终提升在日志级配对统计上为正，但其绝对 regret 更大。这并不矛盾：No-VQA 的候选库更困难、oracle 也明显更高。

### Stage2 修正版相对 V8

| 指标 | 差值 | 95% CI |
|---|---:|---:|
| 实际选中 PDMS | **-0.016030** | **[-0.022006, -0.010166]** |
| Best-of-64 | **+0.007440** | **[+0.002615, +0.012320]** |
| Scorer regret | **+0.023471** | **[+0.018414, +0.028826]** |
| 候选平均 PDMS | -0.011170 | [-0.019154, -0.003035] |
| Pairwise ADE | +0.537168 m | [+0.495161, +0.582202] |
| Endpoint 距离 | +1.426121 m | [+1.312278, +1.544290] |

这是最明确的现象：修正版确实扩大了候选覆盖并提高 oracle，但 scorer 选择恶化的幅度更大，最终 PDMS 显著下降。

### No-VQA 相对 Stage2 修正版

| 指标 | 差值 | 95% CI |
|---|---:|---:|
| 实际选中 PDMS | **+0.021931** | **[+0.015018, +0.028964]** |
| Best-of-64 | +0.007486 | [+0.004101, +0.011212] |
| Scorer regret | **-0.014445** | **[-0.020516, -0.008254]** |
| 候选平均 PDMS | -0.008554 | [-0.017564, +0.000029] |

## 5. 选中轨迹分项

| 分项 | V8 | Stage2 修正版 | No-VQA |
|---|---:|---:|---:|
| No-at-fault collision | 0.984439 | 0.971349 | **0.987650** |
| Drivable area compliance | **0.978264** | 0.957517 | 0.978100 |
| Driving direction compliance | **0.977976** | 0.968837 | 0.977235 |
| TTC within bound | 0.953400 | 0.916351 | **0.954141** |
| Ego progress | 0.856145 | **0.885521** | 0.867344 |
| Comfort | 1.000000 | 1.000000 | 1.000000 |

修正版的进度更高，但 NOC、DAC、DDC 和尤其 TTC 同时下降。它不是“轨迹整体更差”，而是 scorer 倾向选择更激进但风险更高的候选。

No-VQA 相比 V8 的最终增益来自：

- NOC：+0.003211
- TTC：+0.000741
- progress：+0.011199
- DAC：-0.000165
- DDC：-0.000741

## 6. 三版训练设置

三版共享的基础结构和监督：

- ActionDecoder 输出 64 条候选，每条 8 个未来 pose，4 秒 horizon。
- 轨迹与 scorer 联合训练，不是先冻结候选再单独训练 scorer。
- 轨迹损失：对 64 条候选取最小的 waypoint L1。
- Scorer 不直接回归最终 PDMS；它预测 NOC、DAC、TTC、progress、DDC、comfort 六个 factor。
- scorer loss 是上述六个 factor 的 BCE 之和；训练标签由训练 MetricCache 和当前候选通过官方风格的 PDM 评分路径同步构造。
- 推理聚合权重相同：`noc=1, dac=1, ddc=0, ttc=5, ep=5, comfort=2`。
- 训练均为 `bf16-mixed`，最终 Navtest 审计均重新用 FP32 推理。

### 配置总表

| 设置 | V8 | Stage2 修正版 | No-VQA |
|---|---|---|---|
| 初始化 | 从公开合并 checkpoint **只加载 VLM** | 同 V8 | 从基础 InternVL3-2B `from_pretrained` |
| Action head 初始化 | 随机 | 随机 | 随机 |
| VLM 参数 | 全冻结 | 全冻结 | vision/projector/language 全量训练，lm_head 冻结 |
| 实际可训练参数 | action head 21,643,390 | action head 21,643,390 | action head 21,643,390 + VLM 约 1.856B |
| LoRA | 配置开启，但随后随 backbone 全冻结 | 配置开启，但随后随 backbone 全冻结 | 不使用 LoRA |
| 冻结 VLM 的 train/eval mode | eval | train，dropout 保持活动 | train |
| 轨迹监督 | 标准 4 秒 target | 4 秒 target + long-2 target | 标准 4 秒 target |
| GPU × 每卡 batch | 8 × 2 | 16 × 1（2 节点） | 8 × 2 |
| 全局 batch | 16 | 16 | 16 |
| 随机种子 | 0 | 2 | 0 |
| sampler | 默认 DDP sampler，padding | 重建的 official-stage2 sampler，不 padding | 默认 DDP sampler，padding |
| FlashAttention | 开 | 关，eager | 开 |
| Gradient checkpointing | 关 | 关 | 开 |
| Optimizer | AdamW | AdamW | AdamW，betas=(0.9, 0.95) |
| Action-head LR | 1e-4 | 1e-4 | 1e-4 |
| Action-head scheduler | 无，常数 LR | 10% linear warmup + cosine 到 0 | 常数 LR（`action_head_min_lr_ratio=1.0`） |
| VLM LR | 不更新 | 不更新 | vision/language 1e-5，projector 2e-5 |
| VLM scheduler | 不更新 | 不更新 | 3% warmup + cosine，最低为初始 LR 的 10% |
| Weight decay | action head 1e-4 | action head 1e-4，并显式 decay norm/bias | action head 1e-4；VLM 0.05 |
| Gradient clipping | 0 | 0 | norm 1.0 |
| Epoch / optimizer steps | 27 / 174,312 | 27 / 174,312 | 36 / 232,416 |
| 实测训练墙钟 | **38 h 49 m 48 s** | 未在本表锁定 | **98 h 56 m 41 s** |
| 训练样本 | 103,296（由真实集合 padding） | 103,288 | 103,296（由真实集合 padding） |
| 验证样本 | 18,192（由 18,179 padding） | 18,179 | 18,192（由 18,179 padding） |
| 运行时锁定 | 旧环境，未在日志中完整锁定 | PL 2.2.1 / Transformers 4.48.3 / Torch 2.5.1 | 旧环境，未在日志中完整锁定 |

V8 墙钟时间来自原始 launcher 日志中的
`2026-08-28T03:01:49Z` 到 `2026-08-29T17:51:37Z`；日志同时记录
`resume=none`，所以目录名虽含 `restart`，该数值覆盖完整 27-epoch run，而非
仅覆盖断点续训片段。训练主体从 `03:03:47` 开始，和 launcher 墙钟只差约两分钟。

### V8 的具体语义

配置：

`/mnt/project/DriveVLA-M0-stage2/runs/training/stage2_full_seed0_pipeline_v8_restart/code/hydra/config.yaml`

关键点：

- `stage1_checkpoint_path` 指向 M0 公开合并 checkpoint。
- 本地 `_load_stage1_backbone()` 只加载 `backbone`，不会加载 Compress/Q-Former、轨迹 decoder 或 scorer；action head 保持随机初始化。
- `freeze_backbone=true` 最终把 VLM 和已经构造的 LoRA 参数全部冻结。训练日志中的 optimizer 只有 action-head group。
- `scheduler_args=null`，action-head LR 在 27 个 epoch 中保持 1e-4。
- 冻结 VLM 使用默认 `eval` mode，输入给 action head 的表征稳定且无 dropout。
- `long_trajectory_additional_poses=-1`，只学习标准 4 秒 GT 轨迹。

### Stage2 修正版的具体语义

配置：

`/mnt/project/DriveVLA-M0-stage2/runs/training/stage2_official_pl221_tf448_eager_seed2_long2_source_cosine_16x1/code/hydra/config.yaml`

关键点：

- 和 V8 一样，只从公开 checkpoint 恢复 VLM，action head 随机初始化且 VLM 全冻结。
- `long_trajectory_additional_poses=2`：额外读取 10 个 logged-future pose，用 cubic spline 重采样成 8 个、终点约 5 秒的 target，并在普通 4 秒 min-of-64 L1 之外再加一个 long-target min-of-64 L1。
- 这项修改扩大了候选几何覆盖，Navtest Best-of-64 相对 V8 提高 0.007440。
- 使用 16 卡 × 1、seed 2、重建 sampler、PL 2.2.1 和 Transformers 4.48.3。
- FlashAttention 关闭。
- action head 使用 10% warmup 后 cosine 衰减到 0；轨迹生成器和 scorer 属于同一 action-head optimizer group，因此二者一起衰减。
- 冻结 VLM 被显式置于 train mode，参数不更新但 dropout 活动。

### No-VQA 的具体语义

配置：

`/mnt/project/DriveVLA-M0-no-vqa/runs/training/no_vqa_full_ft_seed0_e36/code/hydra/config.yaml`

关键点：

- `checkpoint_path=null`、`stage1_checkpoint_path=null`，不加载 M0/VQA 阶段权重。
- `initialize_from_config=false`，从基础 InternVL3-2B pretrained 权重初始化。
- VLM vision、projector 和 language model 全量可训练；lm_head 被跳过并冻结。
- 不使用 LoRA。epoch 0 checkpoint 审计已经确认 vision、projector 和 language 权重均发生变化。
- VLM 由和轨迹/scorer 相同的规划损失端到端更新，因此当前观测 hidden state 会为候选生成和 factor scoring 共同适配。
- action head 始终使用 1e-4 常数 LR；VLM 使用更小 LR 和 3% warmup/cosine。
- 训练 36 个 epoch，比另外两版多 9 个 epoch。
- 使用 FlashAttention 和 gradient checkpointing；其高分说明 FlashAttention 不是跨所有训练设置都必然造成低分的根因。

## 7. 当前最可信的解释

### 已由数据直接证明

1. 修正版的 long target **提高了候选 oracle 和几何多样性**。
2. 修正版的 scorer **没有利用这个更高上限**，而且选择误差显著扩大。
3. No-VQA 能在本地代码和数据链路上达到 0.911493，因此“本地实现天然只能到 0.89”已经被排除。
4. No-VQA 的 VLM current-observation representation 确实接受规划梯度并发生更新；V8/修正版没有。
5. 三版使用同一 scorer/action-head 架构和同类 PDM factor 标签，架构本身不是三版差异的主要来源。

### 强但尚未完成单变量验证的推断

1. **当前观测表征的任务适配是 No-VQA 优于两个 frozen-VLM 版本的重要原因。** No-VQA 同时提高候选上限和 `selected - candidate mean`，符合“表征同时帮助生成和评分”的预期。
2. **修正版把 scorer LR 和 trajectory LR 一起 cosine 到零，可能使 scorer 无法持续跟随仍在变化的候选分布。** No-VQA 和 V8 都保持 action-head LR 为 1e-4；但这仍需单变量实验。
3. **冻结 VLM 却保持 train mode/dropout 活动可能增加 scorer 输入噪声。** 这是修正版相对 V8 的独有变化之一，但还没有与其他改动完全解耦。

### 当前不能据此断言

- 不能把 No-VQA 的全部 +0.005900 归因给 VLM 表征，因为它还改变了训练时长、初始化、optimizer、scheduler 和 gradient clipping。
- 不能说 FlashAttention 是普遍有害的；表现最好的 No-VQA 恰好开启了 FlashAttention。
- 不能继续把 `long_trajectory_additional_poses=2` 称为最终复现差距的首要修复。它是候选上限修复，不是最终选择修复。
- 不能用三套各自不同的候选库直接判断哪个 scorer 模块本身最好；必须固定 proposals 和 PDM labels 后交换 representation/attention/factor head。

## 8. 对后续 scorer-private 实验的直接含义

下一阶段不应继续只替换 frozen hidden state 上的小 scorer head，而应：

1. 固定同一 M0 64 候选库，交换并审计：current-observation representation、scorer attention、factor heads。
2. 训练只服务 scorer 的当前观测表征分支，保持 M0 轨迹生成器冻结，避免候选分布继续漂移。
3. scorer-private 分支使用独立 optimizer/scheduler；不能让 scorer 因 trajectory scheduler 到零而停止适配。
4. 使用日志级 held-out validation 决策，所有有效方案再做完整 12,146-scene FP32 Navtest。
5. 目标仍是纯 M0、不依赖 DrivOR 表征/权重，完整 Navtest selected PDMS > 0.93。

## 9. 关键源码证据

- VLM-only Stage-1 恢复与 action head 随机初始化：
  `navsim/agents/EpisodeDrive/drivevla_base_agent.py::_load_stage1_backbone`
- VLM 冻结和 train/eval mode：
  `navsim/agents/EpisodeDrive/drivevla_base_agent.py::train`
- long-2 target 构造：
  `navsim/agents/EpisodeDrive/drivevla_features.py::TrajectoryTargetBuilder.compute_targets`
- 标准/long min-of-64 轨迹损失以及 factor BCE：
  `navsim/agents/EpisodeDrive/layers/losses/episode_drive_loss.py::EpisodeDriveLoss`
- No-VQA epoch-0 权重变化审计：
  `/mnt/project/DriveVLA-M0-no-vqa/runs/training/no_vqa_full_ft_seed0_e36/first_checkpoint_audit.log`

以后引用本对比时，以上述 checkpoint SHA、审计目录和配置文件为准。
