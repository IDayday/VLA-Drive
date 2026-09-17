# Flow-GRPO validation — work in progress

**NOT_READY：最终 action-only 初始化权重仍在下载，尚未完成该目标的真实模型集成、短训、恢复、导出和评估。** 这是当前验收记录，不是最终交付结论；下载与开发继续进行。

## 状态定义

- IMPLEMENTED：代码已实现，不能等同验证通过。
- TESTED：相应测试实际执行，有保存的结果。
- NOT_RUN：尚未运行；不得写成 PASS。
- BLOCKED：明确缺少外部条件，当前无法执行。网络下载中单独记录，不宣称权重已可用。

## 已执行证据

`action_contract_tests.log`：31 passed，涵盖 Flow SDE 对锁定 reference 的数值对照、Torch Normal/KL、边界/方差/梯度符号、参数对象覆盖、观测隔离、真实数据 worker/prefetch 恢复、实际 NAVSIM 串行与 spawn 评分及错误/timeout、轻量双进程 DDP 缩放。此日志运行在增加 action-only scene-key RNG 之前；后续修改需追加回归结果。

`baseline_regression_tests.log`：36 passed，原框架兼容性回归。随后 `scene_seed_regression.log` 11 passed；`replay_wrapper_test.log` 1 passed，直接验证生产 loss wrapper 中 replay 场景只计算一次、G/K/分块加权和 RNG 恢复。

`action_architecture.json`：两套构造分别与 989/995 个 checkpoint 元数据 tensor 名称/形状一致。frozen 源/actor trainable 均为 2,233,120,260；unfrozen 源为 2,640,077,316，应用用户的视觉冻结后为 2,233,120,260。此项仅检查结构和精确源码绑定，不代替真实权重内容校验。

原官方完整 SFT checkpoint 的真实回归已完成：独立 SFT/ODE 等价、ratio=1、KL=0、独立 reference、RL 与辅助分支梯度、冻结哈希、精度检查、单卡 update、双卡 50 updates（25 保存/恢复到 50）及原协议评估。详细报告见 `integration_deterministic/`、`integration_precision/`、`full_short_parameter_updates.json`、`eval_sft/`、`eval_full_sft_rl/`。这些是通用训练器和旧完整 SFT 兼容性的证据，**不代替用户最后指定的 action-only checkpoint 验收**。

原完整 SFT 的固定 16 场景验证集 EPDMS：SFT 0.9631009044960672，RL-50 0.9670846566240842。这既非 Navtest 全量结果，也非 action-only 模型性能结论，不据此宣称 GRPO 普遍提升。

旧非确定性连续/恢复逐位比较失败，证据保留在 `resume_nondeterministic_comparison.json`。开启确定性后的连续/恢复完整状态比较已 TESTED：模型、两个 optimizer 分片和两个 rank RNG/stream 文件均逐位一致，见 `resume_deterministic_comparison.json`。比较器已修正 DeepSpeed LossScaler 对象身份与序列化字段状态的区别；所有 tensor 容差仍为零。共享盘顺序读取替换慢速 mmap 扫描后完成全量重跑，未复用未完成结果。

## 25 项验收矩阵

| # | 合同 | 当前 action-only 验证状态 | 证据/剩余工作 |
|---|---|---|---|
| 1 | 原 ODE 等价 | NOT_RUN | 精确 f9449d55 源码 oracle 已接入，待真实权重 |
| 2 | 原 SFT loss 等价 | NOT_RUN | 原 source forward/head 独立对照已实现 |
| 3 | 投影一次/token 语义 | NOT_RUN | 真模型 hook gate 已实现 |
| 4 | export 原 evaluator 输出 | NOT_RUN | verify-export 实现逐 tensor 和原 VLAAgent 输出对照 |
| 5 | 上游 mean/std/logp | TESTED | test_math.py 锁定上游源 sampler |
| 6 | Normal/reduction/KL | TESTED | FP64 数学对照 |
| 7 | sqrt(dt)/边界/noise=0 | TESTED | 显式边界测试 |
| 8 | 经验采样均值方差 | TESTED | Gaussian 统计测试 |
| 9 | old/current ratio 与扰动 | NOT_RUN | 旧完整 SFT 通过；新权重待测 |
| 10 | 完整 reference KL/扰动 | NOT_RUN | 同上 |
| 11 | manifest/optimizer | IMPLEMENTED | alias/覆盖/唯一视觉例外单元测试通过；真实名单待测 |
| 12 | RL/SFT/reference/aux 梯度 | NOT_RUN | action-only 无辅助分支；原完整 SFT aux 回归通过 |
| 13 | detached/current-ref 负例 | NOT_RUN | 真模型 gate 已实现 |
| 14 | 优势梯度方向/全等组 | TESTED | 低维严格基准 |
| 15 | G/K/microbatch/accum 缩放 | TESTED（数学） | 真 action-only 单卡累积/双卡对照待运行 |
| 16 | rank 不等有效数/全等组 | TESTED | 实际双进程 Gloo 与单进程全局基准；生产错误批次整体失败 |
| 17 | 原官方 reward 一致 | TESTED | 真实 NAVSIM cache、逐条/串行/多进程/顺序/候选组合 |
| 18 | 真零分与异常/timeout | TESTED | 缺失 cache、异常及 bounded worker timeout |
| 19 | action normalization/时间 | TESTED | 原 1225 normalization 与 8×0.5s 轨迹检查 |
| 20 | future 隔离/数据划分 | TESTED（数据） | 真实模型 future condition 对照待新权重 |
| 21 | 单卡/多卡真实更新 | NOT_RUN | 旧完整 SFT 已测，新目标待权重 |
| 22 | checkpoint/分块梯度 | NOT_RUN | 原 full checkpoint gate 通过；action-only 新分块 gate 待测 |
| 23 | BF16 网络/FP32 概率 | NOT_RUN | 原 full 真网络对 FP64 通过；新目标待测 |
| 24 | 连续/恢复完整状态 | NOT_RUN | 不能用旧非确定性失败实验充作通过 |
| 25 | frozen/ref 不变及 actor 更新 | NOT_RUN | 新目标需完整 hash/delta 审计 |

## 资源与数据

本机 8×A800 80GB；使用现有 ddp 环境，未升级依赖。原完整 SFT 双卡短测峰值约 50–57 GiB/卡，单卡 optimizer CPU offload 短测约 26.82 GiB；最终 action-only 资源数字待实测。

当前 navtrain 10,000 场景，与 12,146 个 navtest token 不相交。token hash 固定留出 16 个 validation，训练 9,984；noise calibration 使用预先固定的 16 个训练场景。输入、reward、seed、候选选择规则不因分数改变。

可复制命令、源合同和公式分别在 `docs/ddp_flow_grpo/README.md`、`sft_contract.md`、`algorithm.md`。所有实际运行须记录完整命令、开始/结束时间、代码 SHA 和配置；最终 readiness 与 PDMS 趋势分开判定。
