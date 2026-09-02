# No-VQA epoch-35 scorer-private campaign results

更新时间：2026-09-02 UTC

本文档持续记录固定 No-VQA epoch-35 proposal bank 上的 scorer-private
实验。Navtest 只用于通过日志级 validation promotion gate 后的最终审计，不用于
选择 epoch、shortlist、残差比例或安全阈值。

## 锁定输入与评测边界

- Base checkpoint：
  `/mnt/project/DriveVLA-M0-no-vqa/runs/training/no_vqa_full_ft_seed0_e36/lightning_logs/version_0/checkpoints/best-epoch=35-step=232416.ckpt`
- Base SHA256：
  `72c74a113c557df27c86a320f66d4ff2a79fc1a19e678337d5a142a520359309`
- FP32 Navtest feature/proposal cache SHA256：
  `7d01a3d03f3d8b5fee24e596eca27b0e302f1d120609ca237aa61a06439aaa05`
- matching candidate score matrix SHA256：
  `1801678e76aa89877259517742d934833a130423100610525fc1a830d8ac3363`
- Navtest：12,146 scenes、136 segment logs、64 candidates、0 invalid；
- 推理只读当前 No-VQA scene/ego/candidate hidden、候选轨迹、Base factor
  logits 和 Base scorer 值；official PDM candidate matrix 在选定 index 冻结后
  才离线 join；
- future annotation、future image、official score 和 DrivOR/external-model
  representation/weight 均不进入模型推理。

matching candidate matrix 已通过严格 validator：batch/single parity error 为
`0`，regret identity error 为 `3.2e-16`。

## Base 排名下不同 shortlist 的离线候选上限

以下是先按 No-VQA Base scorer 排序，再在前 K 条内用真实离线 PDM 取最优的
候选库上限。它们不是可部署模型成绩。

| Base shortlist | offline oracle PDMS | 相对 Base headroom | 包含全 64 oracle 的场景比例 |
|---:|---:|---:|---:|
| 1 | 0.911493 | 0.000000 | 2.73% |
| 2 | 0.919678 | 0.008184 | 4.80% |
| 4 | 0.929307 | 0.017814 | 8.35% |
| 8 | 0.940399 | 0.028906 | 13.61% |
| 16 | 0.952347 | 0.040854 | 25.08% |
| 32 | 0.965845 | 0.054352 | 48.79% |
| 64 | 0.983129 | 0.071635 | 100.00% |

因此 Top-4 即使使用不可部署 oracle 也不能超过 `0.93`；Top-8 是满足目标的
最小候选范围，但实际 scorer 需要回收至少约 `64%` 的 Top-8 headroom 才能从
`0.911493` 超过 `0.93`。

## 首个独立 promotion：Top-8 conservative residual

冻结 wave-2 的当前最佳 Top-4/Top-8 权重后，将预先锁定的 61 条 validation
物理日志平衡拆成 30-log calibration 和 31-log promotion 两半。calibration
只选择 inference scale、switch penalty 和预测安全 gate；promotion 半完全不
参与这些选择。

| 模型 | promotion PDMS | Base | delta | log-bootstrap 95% CI | 结论 |
|---|---:|---:|---:|---:|---|
| calibrated Top-4 | 0.933767 | 0.931716 | +0.002052 | [-0.000150, +0.004000] | 不晋级 |
| calibrated Top-8 | 0.934350 | 0.931716 | +0.002634 | [+0.000059, +0.005095] | 晋级 |

Top-8 选择的部署 policy 为 residual scale `0.5`、`factor_all` safety gate、
safety floor `0.95`、相对 Base tolerance `0.02`、switch penalty `0`。

- 原始冻结 ranker SHA256：
  `7a87bb5ab18e65fe0a29e9e9bddec877a029c21d9916970b0e6bbd73481750e4`
- calibrated ranker SHA256：
  `e6b8c51e3d99067907d6bfa21a0e73504af97db2ec5a4db018f287130565087d`
- packaged online artifact SHA256：
  `174d34392e043311857596fbc58dfe98f1e9b9e8613419643226d140ca2f8d4c`

## 首个完整 Navtest：验证到测试发生负向反转

严格完整结果目录：

```text
/root/scorer_pdms93_navtest/no_vqa_e35_wave2_interim_top8_calibrated_v2
```

该目录通过完整 validator：12,146 scenes、64 candidates、0 invalid，
batch/single parity error `0`，selected/oracle/regret 均可由 NPZ/CSV 重构。

| 指标 | No-VQA Base | calibrated Top-8 | delta |
|---|---:|---:|---:|
| selected PDMS | 0.911493 | **0.909580** | **-0.001913** |
| NOC | 0.987650 | 0.984604 | -0.003046 |
| DAC | 0.978100 | 0.974560 | -0.003540 |
| DDC | 0.977235 | 0.974560 | -0.002676 |
| TTC | 0.954141 | 0.946731 | -0.007410 |
| progress | 0.867344 | 0.875899 | +0.008555 |
| comfort | 1.000000 | 0.999918 | -0.000082 |
| scorer regret | 0.071635 | 0.073549 | +0.001913 |

配对差值的物理日志 bootstrap 95% CI 为
`[-0.003780, -0.000087]`。场景计数为 4,836 wins、264 losses、7,046 ties；
少量安全退化的损失幅度压过了大量小进度收益。因此该模型是明确负结果，不能
作为 >0.93 或任何测试集改进声明。

## 由负结果触发、但未读取 Navtest 标签调参的训练修复

源码审计显示初始 M0-private loss 有两个可独立验证的问题：

1. 训练会计算并报告 Top-1 regret，却没有使用仓库已经实现的
   `top_regret_rank_loss`；all-pairs loss 中真正决定最终选择的 oracle-vs-rest
   对只占很小比例。
2. 部署在 Base Top-K 内重排，但 factor、factor-rank 和 relative-safety loss
   默认在全部 64 条 proposal 上平均，容易让 shortlist 内的稀有安全错误被
   大量无关候选稀释。

实现保持旧默认行为不变，新增 `--top-regret-weight`、
`--top-regret-minimum-delta` 和 `--factor-loss-scope {all,topk}`。wave-4 在
rl-zt4 GPU 0/2/4 上做三项固定对照：safety-5 baseline、只加 top-regret、
同时加 top-regret 与 Top-K factor supervision。所有方案仍使用相同 No-VQA
proposals、train/validation logs、current-only 输入和完整 Navtest promotion
规则。

当前结论：已经证明现有 conservative Top-8 residual 不能泛化；尚未证明
M0 自有 scorer-private 表征可以把 Navtest 提升到 `>0.93`。目标保持未完成。

## wave-2 最终 8-epoch campaign

早期快照之后，wave-2 的 8 个训练、8 个保守校准和所有 validation-effective
artifact 已全部结束。promotion manifest 一共包含 11 个 raw/calibrated
artifact；11/11 均完成严格完整 Navtest 和四场景在线/缓存一致性，coverage gate
为 `PASS`。所有在线 parity 的 proposal error 为 `0`，score 最大误差不超过
`3.58e-7`。

| 排名 | artifact | validation delta | Navtest PDMS | Navtest delta | 95% CI |
|---:|---|---:|---:|---:|---:|
| 1 | combined Top-4 calibrated | +0.003710 | **0.910824** | -0.000669 | [-0.002237,+0.000650] |
| 2 | candidate-hidden-only Top-8 calibrated | +0.004559 | 0.910437 | -0.001057 | [-0.003357,+0.001272] |
| 3 | combined Top-4 raw | +0.002931 | 0.909887 | -0.001607 | [-0.003556,+0.000158] |
| 4 | combined Top-8 calibrated | +0.002634 | 0.909580 | -0.001913 | [-0.003792,-0.000130] |
| 5 | candidate-hidden-only Top-16 calibrated | +0.006809 | 0.909137 | -0.002357 | [-0.004540,-0.000073] |
| 6 | candidate-hidden-only Top-8 raw | +0.002605 | 0.909059 | -0.002435 | [-0.005698,+0.000744] |
| 7 | combined Top-16 seed-11 calibrated | +0.002761 | 0.908562 | -0.002932 | [-0.005012,-0.000922] |
| 8 | combined Top-16 seed-2 calibrated | +0.003249 | 0.908549 | -0.002944 | [-0.004603,-0.001208] |
| 9 | candidate-hidden-only Top-64 calibrated | +0.004614 | 0.908316 | -0.003177 | [-0.005479,-0.001017] |
| 10 | combined Top-8 raw | +0.002146 | 0.908021 | -0.003472 | [-0.006498,-0.000534] |
| 11 | combined Top-16 factor calibrated | +0.003864 | 0.906954 | -0.004539 | [-0.007195,-0.001881] |

机器可读汇总位于
`reports/m0_independent_scorer_representation/no_vqa_e35_scene_token_wave2_v1/`
下的 `CAMPAIGN_RESULTS.json/.csv/.md`。最终事实是：11 个方法在 validation
上全部为正，但 Navtest 点估计全部为负，且 7 个置信区间已完全低于零；没有
任何模型超过 matching Base，更没有模型超过 `0.93`。因此固定单一 held-out
split 上的约 `+0.002` 到 `+0.007` 不能被解释为可泛化 scorer 改进。

## scene-token wave-1、wave-3、wave-4 与全日志 refit 汇总

所有下列结果均使用同一份不可变 No-VQA FP32 proposal/candidate matrix，完整
覆盖 12,146 个 Navtest scene、136 个 segment log、每场景 64 候选、0 invalid；
每个 validation promotion artifact 都完成了 Navtest 与真实 agent/cache parity。

| campaign | 晋级/测试 | 最佳 Navtest PDMS | 相对 Base | 95% CI | 结论 |
|---|---:|---:|---:|---:|---|
| wave-1：scene-token + current actor/future auxiliary | 7/7 | 0.909441 | -0.002053 | [-0.004087,-0.000272] | 明确退化 |
| wave-2：Top-K/frozen M0 candidate hidden | 11/11 | 0.910824 | -0.000669 | [-0.002237,+0.000650] | 未改善 |
| wave-3：shared-future factorization/candidate-only | 9/9 | 0.909577 | -0.001916 | [-0.003807,-0.000154] | 明确退化 |
| wave-4：Top-1 regret/shortlist factor scope | 5/5 | **0.911414** | **-0.000080** | [-0.002089,+0.001740] | 与 Base 持平，未改善 |
| wave-3 架构的 162-log refit | 3/3 | 0.910348 | -0.001145 | [-0.003195,+0.000996] | 损失缩小但未翻正 |
| wave-4 Top-regret 的 162-log refit | 1/1 | 0.911192 | -0.000302 | [-0.001954,+0.001243] | 未改善 |

对应机器可读结果位于：

```text
reports/m0_independent_scorer_representation/no_vqa_e35_scene_token_v1/
reports/m0_independent_scorer_representation/no_vqa_e35_scene_token_wave2_v1/
reports/m0_independent_scorer_representation/no_vqa_e35_scene_token_wave3_v1/
reports/m0_independent_scorer_representation/no_vqa_e35_scene_token_wave4_v1/
reports/m0_independent_scorer_representation/no_vqa_e35_all_log_refit_wave3_v1/
reports/m0_independent_scorer_representation/no_vqa_e35_all_log_refit_wave4_topregret_v1/
```

这些实验共同排除了以下解释：只增加 scorer MLP/Transformer 容量、冻结 M0
candidate hidden、Base Top-K、shared-future auxiliary、Top-1 ranking loss、
validation-only residual calibration，或在选定 epoch 上使用全部 162 条物理日志
refit，均不足以在 Navtest 上超过 Base。wave-4 Top-regret 的全日志 refit
也从 validation promotion 的 `+0.005398` 翻转为 Navtest `-0.000302`；完整
validator、12,146/136/64/0 覆盖与四场景在线/cache parity（最大 score error
`2.38e-7`）均通过。最接近的仍是未 refit wave-4 的 `-0.000080`，距离
`0.93` 仍差 `0.018586`。

## wave-5：Base-relative conservative policy-improvement head

wave-1 至 wave-4 的主要失败模式是大量小 progress 收益被少量大安全损失抵消。
因此 wave-5 不再给每条候选自由叠加绝对 residual，而是把 Base-selected proposal
定义为精确零收益回退，并预测候选相对 Base 的 q10/q50 gain、NOC/DAC/TTC
退化概率和 safe-improvement 概率。只有 gain 与两类安全 gate 同时通过时才切换。

八个变体在 Navtest 前固定，比较 Base Top-8/16/32、q10/q50、完整
scorer-private scene representation、只用冻结 M0 candidate hidden 的容量对照，
以及 strict/balanced gate。推理仍只使用当前观测、候选、Base factor logits 和
Base score；训练标签、future 和 PDM 不进入推理。完整 152-test 回归已通过。
该 wave 正在 `training-vla-zt2` 八卡运行；只有 held-out physical-log bootstrap
下界大于 0 的 artifact 才会自动进入完整 Navtest。

## wave-6：No-VQA 自有四视角 scorer-private 表征

前四个 scene-token wave 的 private encoder 仍以冻结后的 16 个 Q-Former scene
token 为输入，无法证明更丰富的 scorer-specific perception 是否有效。为进行
真正的表征实验，已从同一个 No-VQA epoch-35 checkpoint 的 InternVL vision
encoder 导出当前时刻 `CAM_F0/L0/R0/B0` 的空间 token：每相机动态切片、每 crop
做 `2 x 2` 池化，总计 80 token、宽度 1536。4-scene smoke 已验证 checkpoint
SHA 为 `72c74a...9309`，resolved No-VQA config 可严格加载，且 manifest 明确
`current_observation_only=true`、`future_or_evaluator_input=false`、
`drivor_checkpoint_or_representation_used=false`。

全量 103,288-scene trainval cache 正在本机八卡导出；完成后同一 pipeline 自动
导出独立的 12,146-scene Navtest current-observation cache。wave-6 已预注册八个
模型，比较普通 residual、Top-regret、conservative-reference、raw-private-only、
raw + released-context 双流、current-actor auxiliary 与 no-actor。其设计冻结于
wave-5 Navtest 之前，因此不会按测试集结果修改。
