# M0 scorer 实验与证据索引

更新时间：2026-09-02 UTC

本文档是 scorer 研究的统一入口。后续回答训练设置、权重身份、Navtest
分数、候选上限和版本差异时，应优先引用这里列出的稳定 Markdown，不再从
聊天记录、终端历史或分散缓存重新推断。

## 正式结果的记录标准

一个数字只有同时记录以下内容，才可作为正式结论：

1. checkpoint 路径与 SHA256；
2. code commit、配置和实际运行命令；
3. split、场景数、日志数、候选数与无效场景数；
4. 推理精度、候选来源、selector 和离线官方 PDM 的调用边界；
5. 机器可读结果或原始审计目录；
6. 明确区分已验证事实、证据支持的解释和待验证推断。

完整 Navtest 的统一验收口径是 FP32、12,146 scenes、136 segment logs、
K=64、0 invalid scenes。`Best-of-64` 只表示离线候选库 oracle 上限，不是
可部署成绩。Navtest 只用于预先确定方案的最终审计，不用于选 epoch 或调参。

## 稳定主报告

| 主题 | 稳定 Markdown | 主要用途 |
|---|---|---|
| V8、Stage2 修正版、No-VQA | [M0_V8_CORRECTED_NOVQA_COMPARISON.md](M0_V8_CORRECTED_NOVQA_COMPARISON.md) | 三次本地训练的配置、权重哈希、完整 Navtest、置信区间与事实/推断边界 |
| No-VQA scorer 表征改进 | [NO_VQA_SCORER_REPRESENTATION_PLAN.md](NO_VQA_SCORER_REPRESENTATION_PLAN.md) | 当前主路线、冻结项、训练输入、验证门槛与产物身份 |
| No-VQA epoch-35 scorer 实测结果 | [NO_VQA_E35_SCORER_CAMPAIGN_RESULTS.md](NO_VQA_E35_SCORER_CAMPAIGN_RESULTS.md) | Base-Top-K oracle、独立 promotion、完整 Navtest 正/负结果与后续 loss 修复 |
| M0 与 DrivOR 原生 64 候选 | [NATIVE_PROPOSAL_BANK_REPORT.md](NATIVE_PROPOSAL_BANK_REPORT.md) | 各自原生候选的 selected、mean、oracle、regret、分项与几何差异 |
| Public 与 epoch-3 固定候选模块交换 | [M0_SCORER_MODULE_SWAP_EPOCH3_VALIDATION.md](M0_SCORER_MODULE_SWAP_EPOCH3_VALIDATION.md) | 历史 calibration 诊断；不是当前主路线，也不是 Navtest 成绩 |
| 独立冻结 DINO scorer | [LOWRES_DINO_NAVTEST_REPORT.md](LOWRES_DINO_NAVTEST_REPORT.md) | 已完成的严格 Navtest 负结果与分项分析 |
| 当前运行状态 | [RUN_STATUS.md](RUN_STATUS.md) | 任务、checkpoint、验证 gate 与完整 Navtest promotion 的滚动状态 |

M0/DrivOR 的逐对机器可读对比位于：

- [m0_vs_drivor_original_native64_v1/COMPARISON.md](m0_vs_drivor_original_native64_v1/COMPARISON.md)
- [m0_vs_drivor_scaled_native64_v2/COMPARISON.md](m0_vs_drivor_scaled_native64_v2/COMPARISON.md)
- [drivor_original_vs_scaled_native64_v1/COMPARISON.md](drivor_original_vs_scaled_native64_v1/COMPARISON.md)

DrivOR 只用于外部差距分析，不得成为 M0 新方法的输入、初始化或成绩来源。

## 当前数字快照

| 系统 | 完整 Navtest selected PDMS | Best-of-64 | Regret |
|---|---:|---:|---:|
| M0 public | 0.909594 | 0.984112 | 0.074518 |
| 本地 V8 | 0.905593 | 0.968203 | 0.062610 |
| 本地 Stage2 修正版 | 0.889563 | 0.975643 | 0.086080 |
| 本地 No-VQA | **0.911493** | **0.983129** | 0.071635 |
| DrivOR original，仅分析 | 0.936907 | 0.993342 | 0.056436 |
| DrivOR scaled，仅分析 | 0.945829 | 0.994094 | 0.048265 |

这些数字对应不同原生 proposal bank；跨行不能只凭 regret 判断完整系统优劣。
全部 checkpoint、SHA 和审计目录见各自稳定主报告。

## 后续文档模板

每个新实验至少包含：

```text
# 实验名
## 研究问题与预注册假设
## 固定项与训练变量
## 数据、日志、候选和 checkpoint 身份
## code commit / config / SHA256 / command
## validation 选择规则
## 完整指标与日志级置信区间
## 分项差异与失败样本
## 泄漏、精度、顺序和在线/缓存一致性审计
## 已证实结论
## 不能由本实验推出的结论
## 是否进入完整 Navtest
```

新的关键结论必须先写入对应 Markdown，再更新本索引和 `RUN_STATUS.md`。
