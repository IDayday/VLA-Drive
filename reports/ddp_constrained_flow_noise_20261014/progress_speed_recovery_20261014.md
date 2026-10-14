# 训练速度与 C 原生恢复

两组仍按训练 seed42、完整 RL_TRAIN 两次遍历、每组 203,184 次筛选前场景曝光运行；G16/K10、两个 inner epochs 和正式单候选 ODE seed2101 不变。耗时估计使用完整 collection 的墙钟，包含采样、评分、更新和 IO，不把已经重叠的 CPU/GPU 计时相加。

## 当前速度与完成估计

截至 **2026-10-14 18:49 UTC**，两组各在一台八卡主机运行。[八 rank 完整更新与吞吐原始汇总](performance/live_speed_20261014T1849Z.json)、[C 正式激活与验收前缀一致性](performance/C_repair_production_activation.json)。

| 训练线 | 完整更新 / 原始场景曝光 | 两遍预算完成比例 | 当前原始场景/秒 | 优化前窗口 | 观测吞吐倍数 | 剩余纯训练 | 含保存、评价和汇总的安排 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| N_FLOW_NOISE | 3,212 / 40,086 | 19.73% | 2.52 | 1.063 | 约 2.37× | 约18小时 | 约20–24小时 |
| C_SDE，继承1850恢复点 | 1,888 / 23,648 | 11.64% | 1.45 | 0.737 | 约 1.97× | 约34.4小时 | 暂估36–40小时 |

N 最近16/32/64个完整 collection 分别为2.519/2.516/2.520 raw scenes/s；64组窗口为1,600场景 / 635秒。C 刚恢复，当前只有16组稳态窗口，为392场景 / 270秒，已排除启动和最初 collection。C 的短窗口估计仍需后续观察，但不能拿 N 的速度代替 C 预测。

每场景仍生成并评分16条真实候选。当前 N / C 分别约40.3 / 23.2候选/秒，原始场景覆盖与有效组筛选规则没有减少。原窗口参考：[N 初期部署比较](performance/N_production_throughput.json)、[C/N 原正式训练分解](performance/training_cost_20261014T1705Z.json)。不同时间窗和主机负载使这些倍数属于观察结果，不是同场景配对的因果保证。

两组并行的完整交付暂按较慢 C 的 **36–40小时**安排。按当前进度，C 距首个注册 `raw25398` DEV 观察点约20分钟训练，另加实际评测；N 距下一 `raw50796` 点约71分钟训练。DEV 上一次实测约5.3分钟；这两个预计观察时间不等于最终完成时间。

初期 N 的短窗口曾出现更新阶段从约 2 秒变为 5 秒的波动；现在已回到约 1.94 秒。没有通过修改学习目标或数值检查来消除该波动，根因尚未定位。此前报告的批量评分 warm ABBA 2.68× 是评分模块结果；不能直接作为整段训练加速倍数。此前批量版最初 16 个 collection 仅提升 1.9% 的记录仍保留，不因后续速度回升而删除。

纯训练 ETA 为 `(203184 - 已完成原始曝光) / 当前曝光吞吐`。完整任务还包含原子 checkpoint、固定 TRAIN 监测、剩余全量 DEV、锁定后的 Navtest 和汇总。两台机器并行工作，整体完成时间取较慢的一组，不相加。时间估计依赖当前吞吐持续，不能提前承诺后续安全规则不会触发。

## C 已恢复，无需从 SFT 重训

原 C 在 update1850 后，因为一个 fresh transition 的 flat/B1 数值差异触发原 1e-4 检查。此前已从 update1024 精确重放到1850，恢复完整模型、FP32 Adam、RNG、dual、KL 和数据游标；本次继续使用这个完整恢复点，保留 23,184 次原始场景曝光。

修复只处理被原误差检查发现的转移：在第一个 inner epoch、current 与 old behavior 参数相同的时刻，检测 flat 重算与保存 old logprob 的差异。仅对超过原 1e-4 的转移，用原采样 B1 计算在**同一保存状态**重新计算可微 mean/std/logprob；保留真实 old 值，不裁剪 ratio。选中的转移索引在第二个 inner epoch 固定，作为每 rank 状态随原子 checkpoint 保存。其余转移仍使用原 flat 实现。若 B1 仍不能通过原检查，继续报错；若一组超过8个转移需要回退，也停止并保留证据。

实际故障组只回退第159个转移。该组 fresh ratio 最大误差从 `0.00010013580322265625` 降至 `0.00004762411117553711`，低于**未放宽的**原阈值。回退位置的原 B1 density 和全部359个动作头参数梯度精确一致。

[登记](performance/C_probability_layout_registration.json)、[真实故障值与梯度](performance/C_probability_layout_values_gradients.json)、[八卡真实更新/恢复](performance/C_probability_layout_acceptance.json)、[验收与重试成本](performance/C_probability_layout_cost.json)。

验收从真实1850完整状态开始，连续四次原生更新与第一 inner 更新后中断再恢复逐值比较：actor、八个 FP32 optimizer 分片、每 rank RNG、dual及其 optimizer、KL、数据游标、回退选择与保存候选完全一致；359个 FP32 主参数均实际更新。第一组采样链、old mean/std/logprob、评分与故障前保存值一致。另有5项针对性测试验证选择冻结、恢复、真正概率错配及非有限值不能被回退掩盖。

这是经过新验证的数值执行修订。原1851更新已中止，不存在可用的旧完成结果，因此不声称修复后的 optimizer 轨迹与一个不存在的旧终点逐位一致，也不将该修复列为第三种 RL 算法。

正式续训从原1850状态重新启动；验收的1854权重没有进入正式初始化。正式新完成的1851–1854更新与验收连续分支在八 rank 的PG、KL、replay、ratio、梯度norm、dual、LR及数据进度逐值一致；截至本记录已实际完成1888更新。原失败 run、826步精确恢复记录和所有 checkpoint 均保留。新输出目录为 `/mnt/project/constrained-flow-noise-20261014/formal_C_SDE_seed42_v1_probability_repair`，继承同一个科学 run identity，单独记录执行身份 `e2e904b3efd7c6dd4b51fa69492322904455574a6111e08f0a9c29e12b0a012f`，候选 behavior version 同时记录该身份。

## 已执行的命令

以下从本任务 worktree 和已授权 Tsinghua 主机执行，输出目录拒绝覆盖。

```bash
PY=/mnt/project/flow-safe-improvement-20261013/venv/bin/python
RUN=/mnt/project/constrained-flow-noise-20261014
CP="$RUN/C_SDE_ratio_failure_20261014T1609Z/replay_from_update1024/checkpoints/update001850_raw023184"

$PY -m scripts.analysis.constrained_flow_noise.qualify_probability_layout \
  --checkpoint "$CP" \
  --failure "$RUN/formal_C_SDE_seed42_v1/failures/trainer_rank3.pt" \
  --output "$RUN/C_probability_layout_values_gradients_v2.json"

$PY -m pytest -q tests/constrained_flow_noise/test_probability_layout.py

$PY -m scripts.analysis.constrained_flow_noise.qualify_probability_training \
  --root "$RUN" --output "$RUN/probability_layout_qualification_v1" \
  --checkpoint "$CP" --values-proof "$RUN/C_probability_layout_values_gradients_v2.json"

$PY -m scripts.analysis.constrained_flow_noise.resume_repaired_c \
  --root "$RUN" --checkpoint "$CP" \
  --acceptance "$RUN/probability_layout_qualification_v1/acceptance.json" \
  --output "$RUN/formal_C_SDE_seed42_v1_probability_repair" \
  --receipt "$RUN/C_probability_repair_deploy_v1/resume.json"

python -m scripts.analysis.constrained_flow_noise.summarize_training_speed \
  --run "$RUN/formal_N_FLOW_NOISE_seed42_v1" \
  --run "$RUN/formal_C_SDE_seed42_v1_probability_repair" \
  --budget "$RUN/formal_budget.json" \
  --output "$RUN/live_speed_20261014T1849Z.json"
```

后续中断恢复沿用 `scripts.analysis.constrained_flow_noise.train_stable_probability` 的原生 `--resume`，携带同一 `--layout-acceptance` 和原科学参数；完整实参保存在实际部署 receipt。已通过的 inner 中断验收执行的正是这个入口，恢复时保留回退选择、固定旧链和 optimizer，不重置 Adam 或重新解释旧优势。

本次只修复执行与恢复。新的正式规划效果仍以各自注册曝光点的完整 DEV、原单候选 ODE 评价为准；训练提速不能作为 PDMS 或安全改善证据。
