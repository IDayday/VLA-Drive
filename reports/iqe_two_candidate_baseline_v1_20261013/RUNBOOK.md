# 70k two-candidate baseline — execution runbook

源码起点：`5ef80979dd59c8bd9278b1cae5d463e2011aade0`。工作区为 `/mnt/project/DriveVLA-M0-iqe-two-candidate-baseline-v1`，分支 `feature/iqe-two-candidate-baseline-v1`。本文件依据 `iqe.two_candidate --help`、实际子命令和当前配置编写；最终提交与运行结果由交付报告另行记录。

本轮只训练一个专家和一个随机初始化 Scorer。最新用户授权的有效预算为专家连续 10,000 updates、Scorer 连续 20,000 updates；第 5,000 步为 Scorer 内部阶段切换，不暂停或重启。扩充通过独立 `TRAINING_BUDGET_AMENDMENT.json` 生效，原 YAML 的 1000/2000/500 保留为历史初始绑定。专门真实恢复 probes 已取消，下面的恢复入口仅供意外中断。文档编写时缓存和 CPU 评分仍在运行，不表示训练或选轨收益已经完成。

**固定身份与权限。**

- Experiment：`two_candidate_base70k_seed31415_v1`。
- Config：`/mnt/project/DriveVLA-M0-iqe-two-candidate-baseline-v1/configs/iqe/two_candidate_baseline/base70k_v1.yaml`。
- Output：`/mnt/project/iqe-two-candidate-baseline-20261013/base70k_v1`。
- Base：`/mnt/project/iqe-remote-rolling-cache-20261012/continuation/step_070000.pt`。
- Base SHA256：`30517484efcc8740c331f42f3f2ba2d1896a4f95cf70e2bf662e0d6c4484e8b0`。
- 已存在 `BASE_BINDING.json`，绑定时间 2026-10-13 02:20:10 UTC。后续阶段不再查找更新的 Base。
- 当前 Base 和其动态缓存生产保持运行；本实验不训练 Base，不修改 22k 等已有身份，不启动 Navtest，不更新 ACTIVE。
- 用户授权使用 `training-vlawm-tsinghua-zt`、`training-vlawm-tsinghua-zt2` 的合适资源。每次派发仍读取 `/mnt/project/server_dispatch_policy.json` 并核验真实 GPU/PID/共享 IO；不能停止其他训练或评测。以下 `cuda:0` 指进程内已分配的可见 GPU，不代表可以直接抢占宿主物理 0 号卡。

如果同一输出已有活跃 `WORKFLOW_LIVE.json` 所指的真实进程，保持该唯一 workflow owner。不要在它运行时并发执行这些恢复/阶段命令。文件中的历史 RUNNING 字样本身不等于实时进程存在。

在已分配资源且可见相同 project 挂载的容器内设置：

```bash
cd /mnt/project/DriveVLA-M0-iqe-two-candidate-baseline-v1
PYTHON=/root/miniconda3/envs/ddp/bin/python
CONFIG=/mnt/project/DriveVLA-M0-iqe-two-candidate-baseline-v1/configs/iqe/two_candidate_baseline/base70k_v1.yaml
RUN_ROOT=/mnt/project/iqe-two-candidate-baseline-20261013/base70k_v1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

本 CLI 没有虚构的 `--mode`、通用 `--max-steps` 或通用 `--roles` 参数。样本预算由固定 YAML 决定，完整更新数由经校验的独立预算 amendment 覆盖初始值；不要手改 YAML 或 Base/context 身份。头部训练默认单 GPU：global batch 128，expert microbatch 32，Scorer microbatch 16；分别累积 4/8 次，学习率不因 microbatch 改变。CPU 官方评分并发为 16、每次调用最多尝试 3 次；不在 GPU batch loss 中启动仿真。

**1. 初始化与固定划分。**

当前绑定已经存在，以下命令为可复用入口，不创建更新的 Base 身份：

```bash
"$PYTHON" -m iqe.two_candidate init-baseline --config "$CONFIG"
"$PYTHON" -m iqe.two_candidate build-baseline-splits --config "$CONFIG"
```

`init-baseline` 核对完整 checkpoint、来源、Query 架构与契约，保存固定 Base、场景和规则。`build-baseline-splits` 读取并校验已锁定角色，不按结果重新划分。当前实际角色如下：

| 角色 | 场景 | logs/groups | 允许用途 |
|---|---:|---:|---|
| D_E / expert_fit | 16,000 | 832 | 专家失败挖掘、GT 审计、H/N 训练 |
| D_S / scorer_fit | 4,000 | 205 | 专家冻结后的 Scorer 训练 |
| stage_val | 1,024 | 53 | 专家和 Scorer checkpoint 选择 |
| selector_cal | 1,024 | 62 | 可选单 margin，继承固定集合 |
| dev_report | 1,696 | 16 | 固定完整开发结果 |

D_E/D_S 的 log/source-parent 连通组隔离；D_S 不进入专家的困难或普通 bucket。根据本 Base 训练清单，D_E/D_S 已被 Base 见过；不能称为整个 VLA 未见数据。stage_val/cal/dev 不在该 run 的训练清单中。`final_test/navtest` 不进入本实验。

在两个头部尚未启动前，显式记录本轮已授权的预算扩充：

```bash
"$PYTHON" -m iqe.two_candidate amend-training-budget \
  --config "$CONFIG" \
  --budget-config configs/iqe/two_candidate_baseline/training_budget_v2.yaml \
  --reason 'Explicit user extension on 2026-10-13 before head training'
```

预算文件只有 `expert_updates: 10000`、`scorer_updates: 20000`、`pointwise_updates: 5000` 三个字段。该命令保存不可变 `TRAINING_BUDGET_AMENDMENT.json`，绑定原 config、Base 和数据划分 hash。相同目标重复执行为幂等读取；不同目标或任一头部已开始更新时拒绝改预算。已有相同 amendment 时保持它的原始时间与理由；不要手改文件。两个 trainer 各自记录有效 `TRAINING_BUDGET.json`，checkpoint/恢复依赖包含 amendment hash。除更新上限和 Scorer 阶段边界外，所有配方和身份保持原值。

**2. 缓存当前 Base 的规划条件和 W/A。**

前置依赖：固定绑定和场景清单、真实图像/ego/GT 可读。完整固定清单为 23,744 场景。单进程入口如下；同一个 shard 的同一输出只允许一个 writer：

```bash
"$PYTHON" -m iqe.two_candidate cache-baseline-context \
  --config "$CONFIG" --device cuda:0 --shard 0 --shards 1
```

已有并行缓存任务继续完成，不重复启动。需要安排新的分片时，入口支持 `--shards 8 --shard 0` 至 `--shard 7`，由调度记录给两台 tsinghua 容器分配互不重复的 shard 和空闲 GPU；各 worker 仍使用自己的进程内 `cuda:0`。最多 8 个全局 shards。`--max-samples` 只适用于明确的小范围诊断，不作为完整缓存的完成标志。

输出为 `$RUN_ROOT/features/` 中的 tensor 与完成 receipt；每个消费者核验 checksum 和 Base/runtime/observation 身份。规划 F_plan 与原始 W/A 来自同一次 VLM 前向；Scorer 可训练投影不缓存。这里不使用持续 Base 训练的动态视觉池，也不复用 22k 特征。

**3. Base 失败、GT 审计和唯一专家 manifest。**

前置依赖：D_E 与 stage_val 所需缓存完整，官方 metric contexts 可读。下面三个名称是同一个幂等阶段的别名；正常运行只需执行 `build-expert-data` 一次：

```bash
"$PYTHON" -m iqe.two_candidate build-expert-data --config "$CONFIG"
```

也可通过 `score-baseline-base --config "$CONFIG"` 或 `audit-baseline-targets --config "$CONFIG"` 进入同一阶段，它们不会只做名字中的一个子步骤。本阶段生成 Base 标签、F/H/F_V/H_V、GT 审计及固定 H/N manifest；不使用 D_S 结果指导专家。候选和 GT 各自独立走官方单轨迹 reference backend，GT 不成为部署候选。评分临时失败保存证据，修复后重跑同一阶段；有效结果按身份复用，失败不写为有效零分。

规则固定为 Base <0.8 或有效 NC/DAC 非满分；H 要求 GT≥0.8、GT−Base≥0.05、GT NC/DAC 满分及输入/轨迹合法。普通 N 固定最多 2,048 个。训练 sampler 80% H +20% N，loss 无第二次比例加权。无合法 H 或 N 时该阶段报错，不降门槛制造训练数据。

主要产物：`BASE_FIT_VAL_SCORES.json`、`HARD_SETS_BEFORE_EXPERT.json`、`EXPERT_MANIFEST.json`、`EXPERT_TRAIN_MANIFEST.json`、`TARGET_AUDIT_QUEUE.json`。

**4. 连续训练唯一专家。**

前置依赖：专家 manifest、H/N 缓存和固定 stage_val 标签完整；没有同 run 的其他 expert trainer。只复制和更新 Query/Decoder/轨迹头，原 Base 冻结。

```bash
"$PYTHON" -m iqe.two_candidate train-baseline-expert \
  --config "$CONFIG" --device cuda:0
"$PYTHON" -m iqe.two_candidate evaluate-baseline-expert --config "$CONFIG"
```

预算 amendment 生效后连续执行 10,000 updates，不设置 `--stop-after`。AdamW `lr=1e-4`、`wd=0.01`、warmup 50 后恒定 lr、clip 1.0。每 250 步固定 stage_val 官方验证，直至第 10,000 步；按 Oracle@2 增益、H_V 直接收益、较早 step 选 checkpoint。`evaluate-baseline-expert` 读取完成后的选择记录，不重新训练，也不以 D_S/dev 选 checkpoint。

待生成的真实路径：

- `/mnt/project/iqe-two-candidate-baseline-20261013/base70k_v1/expert/best.pt`：完成固定验证选择后才可使用。
- 同目录 `result.json`、`step_*.pt.COMPLETE.json`、`validation_*.json`。
- `$RUN_ROOT/EXPERT_FREEZE_AUDIT.json`：参数/buffer/raw/physical 保持证据。

这里不预先指定“best 是第几步”，不声称专家已经专精成功。

**5. 冻结专家后构建真实两候选银行。**

前置依赖：专家训练 COMPLETE，best hash 匹配，目标角色缓存齐全。

```bash
"$PYTHON" -m iqe.two_candidate build-pair-bank \
  --config "$CONFIG" --device cuda:0
```

完整 D_S、stage_val、selector_cal、dev_report，以及固定 D_E 的 H/N/F 诊断清单，各场景仅包含绑定 Base 与最终专家。使用 scene/hash join；输出 `PAIR_BANK_SCENES.json`、`PAIR_BANK.json`、`PAIR_BANK.npz`、`PAIR_BANK.COMPLETE.json` 与统计。D_E 的附加诊断行不会进入 Scorer 拟合。GT 和中间专家 checkpoint 不加入主银行。

**6. 从零连续训练内置 Scorer。**

前置依赖：完整、合法的两候选银行与 D_S W/A 缓存；不自动加载其他实验的 Scorer。

```bash
"$PYTHON" -m iqe.two_candidate train-baseline-scorer \
  --config "$CONFIG" --device cuda:0
```

全部新增参数由 seed 31415 随机初始化，记录 `SCORER_INITIALIZATION.json`。预算 amendment 生效后连续 20,000 updates，AdamW `lr=1e-4`、`wd=0.01`、warmup 100 后恒定 lr、clip 1.0。前 5,000 步为 128 自然场景；后 15,000 步为 64 自然 +64 gap 分布比较场景。5,000 步处不停止程序或重启学习率。

自然流只使用 D_S，用于 SmoothL1(beta0.1) 与有效分项 BCE；比较流按 `q=abs(gap)/sum(abs(gap))` 采样，用普通 pair softplus 均值，不再乘 gap。总权重为 value1、component0.2、ranking0.2。每 250 步由固定 stage_val 原始 argmax 的真实分数、较少新增 NC/DAC 违规、较早 step 选择。

待生成路径为 `/mnt/project/iqe-two-candidate-baseline-20261013/base70k_v1/scorer/best.pt`；只有同目录 `result.json` 为 COMPLETE 且 hash 一致时才可进入评估。`COMPARISON_DISTRIBUTION.json` 保存 D_S 的固定 q、有效 pair 数、分组和重复采样信息。

**7. 单 margin、完整开发结果与研究导出。**

前置依赖：固定 Scorer best 与合法银行；不因 dev 结果返回修改阈值/配方。

```bash
"$PYTHON" -m iqe.two_candidate calibrate-baseline-margin \
  --config "$CONFIG" --device cuda:0
"$PYTHON" -m iqe.two_candidate evaluate-two-candidate-baseline \
  --config "$CONFIG" --device cuda:0
"$PYTHON" -m iqe.two_candidate export-research-bundle \
  --config "$CONFIG" --device cuda:0
```

margin 只在 selector_cal 从 `{0,0.01,0.02,0.05,0.1}` 中选，预算为相对 Base 零新增 NC/DAC 违规、零高分转零分。没有有益可行配置则明确 always-Base；原始 argmax 仍单独报告。`evaluate` 会核验/幂等复用同一校准协议，并在完整 1,696 场景 dev 上报告 Base、Expert alone、Oracle@2、scratch Scorer argmax、margin。

结果重点是 H_train/H_val 直接收益、Oracle 互补空间、selected 相对 Base 和较好单策略的收益、A/B/R 恒等式、错误替换代价和安全变化，不只看 Oracle 非负。输出 `FOUR_WAY_RESULTS.csv`、`RESULTS.json`、`VERDICT.json` 和报告；不可把等待输入写成科学失败或成功。

研究导出路径为 `$RUN_ROOT/bundle/`，包含本实验专家和 Scorer 及绑定 Base 契约，不包含 GT/metric-cache 推理依赖；不会更新 ACTIVE。

**8. 已训练模型的在线接线与时延。**

这些入口使用本实验固定数据与研究 bundle，不触发 Navtest。前置依赖为已训练 best；benchmark 还要求上述 bundle 完成。

```bash
"$PYTHON" -m iqe.two_candidate audit-online --config "$CONFIG" --device cuda:0
"$PYTHON" -m iqe.two_candidate benchmark --config "$CONFIG" --device cuda:0
```

检查真实图像、标签独立性、Base/K=1 一致性、缓存数值和 VLM 调用数。时延使用 5 次预热、30 次正式重复，分别测真实图像 Base、两候选和两候选+Scorer。不要把缓存头部训练耗时当线上总耗时。无需安排人为中断的恢复资格实验。

**9. 一键流程与意外中断。**

初始化与缓存已经完成、且不存在同输出活跃 workflow 时，可以由唯一 owner 执行：

```bash
"$PYTHON" -m iqe.two_candidate run-two-candidate-baseline \
  --config "$CONFIG" --device cuda:0 --wait-cache-seconds 0
```

该入口复用相同阶段，并连续完成专家和 Scorer；`--wait-cache-seconds` 只控制有限的缓存等待（0 至 10800），不构建缓存。它保存 workflow 状态与代码身份，拒绝第二个同输出 owner。只有完整银行已通过验证时，`--from-stage scorer` 才能跳过已完成的专家阶段。不要同时运行一键流程和独立训练命令。

仅当进程意外中断并确认已退出时，使用以下恢复入口；保持同一配置、依赖、设备拓扑及数据身份。`auto` 校验 checksum/receipt，选择最新兼容的完整 checkpoint，恢复 optimizer/scheduler/RNG 与实际消费游标，不从头训练：

```bash
IQE_RECOVERY_STAGE=train-baseline-expert
# 如果实际中断的是 Scorer，将上一行改为 train-baseline-scorer。
"$PYTHON" -m iqe.two_candidate "$IQE_RECOVERY_STAGE" \
  --config "$CONFIG" --device cuda:0 --resume auto
```

不得为了验证该命令主动暂停本轮 10,000/20,000 步主训练。现有恢复单测不代替本轮未执行的真实恢复实验；不能虚报 probe 已通过。

**10. 换 Base 必须新建实验。**

以下是未来独立实验的创建命令，当前任务不执行。只在指定合法 Base run 里解析一次最新完整快照，新的 output/experiment ID 不复用本轮缓存或标签：

```bash
NEW_ROOT=/mnt/project/iqe-two-candidate-baseline-20261013/base_latest_rebind_v1
"$PYTHON" -m iqe.two_candidate init-baseline \
  --config "$CONFIG" \
  --experiment-id two_candidate_latest_rebind_seed31415_v1 \
  --output-root "$NEW_ROOT" \
  --base-run /mnt/project/iqe-remote-rolling-cache-20261012/continuation \
  --base-policy latest_complete
NEW_CONFIG="$NEW_ROOT/EXPERIMENT_CONFIG.yaml"
"$PYTHON" -m iqe.two_candidate build-baseline-splits --config "$NEW_CONFIG"
"$PYTHON" -m iqe.two_candidate amend-training-budget \
  --config "$NEW_CONFIG" \
  --budget-config configs/iqe/two_candidate_baseline/training_budget_v2.yaml \
  --reason 'Explicitly retain the documented 10000/20000/5000 budget for this new Base experiment.'
"$PYTHON" -m iqe.two_candidate cache-baseline-context \
  --config "$NEW_CONFIG" --device cuda:0 --shard 0 --shards 1
"$PYTHON" -m iqe.two_candidate run-two-candidate-baseline \
  --config "$NEW_CONFIG" --device cuda:0 --wait-cache-seconds 0
```

`NEW_ROOT` 是该未来命令明确创建的输出，不是声称已经存在的实验。初始化后始终使用生成的 `EXPERIMENT_CONFIG.yaml`，后续阶段读取其 `BASE_BINDING.json`，不再次刷新 latest。显式 checkpoint 使用 `--base-checkpoint` 替代 `--base-policy latest_complete`，路径必须属于指定 `--base-run`；两参数不能同时传。

本轮 70k 的原权重、持续 Base 训练和其他锁定实验均不因此改变。
