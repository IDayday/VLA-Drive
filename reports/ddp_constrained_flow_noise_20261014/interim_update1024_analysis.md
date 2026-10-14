# 第 1024 次更新：部署收益、约束缺口与采样效率

这是用户要求提前进行的诊断观察。两组都从真实、完整的正式训练检查点导出，
没有使用验收权重。C_SDE / N_FLOW_NOISE 分别消耗 12,760 / 12,744 次原始场景曝光，
均完成 512 次 behavior collection、1024 次 optimizer update；原始曝光相差 16，
约为各自正式总预算的 6.3%。不能把它们称为最终训练结果。

评价使用完整 RL_DEV 1696 场景、16 logs，原 VLAAgent 当前观测入口、原 10 步 ODE、
每场景单候选、seed2101、canonical v1.1。部署没有 noise_head、搜索、GT 选择或新 scorer。
SFT 可能在预训练中见过 DEV；这里只能称 RL 更新未见。未运行本轮 Navtest。

## 实际部署结果

分数均为 0–100；违规和零分为场景计数。

| 模型 | PDMS | 相对 SFT | NC | DAC | TTC | EP | 零分 | TTC 违规 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 固定 SFT | 93.6192 | — | 100 | 99.2335 | 99.9410 | 85.8065 | 13 | 1 |
| C_SDE update1024 | 94.4840 | +0.8647 | 100 | 99.7052 | 99.7642 | 87.4573 | 5 | 4 |
| N_FLOW_NOISE update1024 | 94.4822 | +0.8630 | 100 | 99.5873 | 99.4104 | 87.9720 | 7 | 10 |

PDMS 配对 log-cluster bootstrap 95% 区间：C−SFT 为 [+0.5819,+1.0700] 分，
N−SFT 为 [+0.6396,+1.0957] 分。N−C 为 −0.0017 分，区间 [−0.1577,+0.2086]。
这些区间只反映固定训练 seed42 / 推理 seed2101 下的 log 抽样不确定性，
不包含训练或推理 seed 不确定性，不把 G16 当作训练复现。

| 转移，相对固定 SFT | C_SDE | N_FLOW_NOISE |
| --- | ---: | ---: |
| 原非零 → 零分 | 1 | 2 |
| 原零分 → 非零 | 9 | 8 |
| DAC 新增违规 / 恢复 | 1 / 9 | 2 / 8 |
| TTC 新增违规 / 恢复 | 4 / 1 | 10 / 1 |
| 联合 NC/DAC/TTC 新增违规 / 恢复 | 5 / 9 | 12 / 8 |
| 当前联合违规总数 | 9 | 17 |

两组 NC 与 Comfort 均值保持，DAC 改善，TTC 下降；不能概括成“安全全部下降”。
两组都未满足登记的安全择优规则。N 相对 C 多 6 个 TTC 违规，且没有任何
“C 违规、N 恢复”的 TTC 场景；违规率差的 log-bootstrap 95% 区间为
[+0.0843,+0.5603] 个百分点。

可见行为改变主要包括：C / N 的实际模拟前进行程均值增加 0.251 / 0.415 米，
平均速度增加 0.066 / 0.114 m/s，终点速度增加 0.107 / 0.380 m/s。
请求轨迹二阶位置差的均值改变为 −0.0022 / +0.0300 米。
二阶差只是几何平滑性描述，不能当官方 Comfort 或事故概率。
这些结果支持“更积极的前进与 TTC 退化同时出现”，尚未证明前进改变导致违规，
也不能证明已学会场景相关让行。

N 的 TTC 违规率相对 SFT 增加 0.531 个百分点，区间 [+0.0912,+0.8084]。
这一次尚未达到单次 +2 个百分点暂停条件，但已构成连续退化规则的第一次观察。
独立只读监测器会在下一次登记的完整 DEV 后应用同一规则：若再次至少 +0.5
个百分点且区间下界大于零，向本任务八个训练 rank 发送原生 SIGUSR1 保存停止请求。
不自动恢复、不调整 lambda/LR、不把额外观察加入 checkpoint 择优列表。

[C 完整报告](interim_update1024_C_SDE_ODE.json)、
[N 完整报告](interim_update1024_N_FLOW_NOISE_ODE.json)、
[N−C 配对报告](interim_update1024_N_minus_C.json)、
[真实评价成本](interim_update1024_evaluation_costs.json)。

## 有效组与真实消耗

下表读取同样前 512 个已保存、经 checksum 验证的真实 behavior collections。
它是训练过程中变化策略的记录，不是冻结模型在整个训练集上的统一评价。

| 组统计 | C_SDE | N_FLOW_NOISE |
| --- | ---: | ---: |
| 原始场景组 | 12,760 | 12,744 |
| 实际进入 PG | 8,190（64.19%） | 8,192（64.28%） |
| 最终优势全零 | 3,281（25.71%） | 3,204（25.14%） |
| 有效但补样超过上限，未用于 PG | 1,289（10.10%） | 1,348（10.58%） |
| PDMS 全部相同 | 4,485 | 4,349 |
| 16 条均为满分 | 4,472 | 4,337 |
| 16 条均为零分 | 3 | 4 |
| 其余相同且介于 0 和 1 | 10 | 8 |

“99.7% 满分”以 **PDMS 等分组** 为分母，不能改写成整个训练集 99.7% 满分。
上述全满分组占各自原始场景曝光约 35.05% / 34.03%。单次 G16 全通过
不保证所有噪声均安全，也不保证不同轨迹的物理安全余量相同。

零最终优势组中，3,003 / 2,954 组的全部六个反馈分量相同；278 / 250 组
仅在当时 lambda=0 的成本分量上有差异。没有发现非零活跃分量恰好互相抵消的组。
部分零优势组仍参加 first16 reference 保持和未筛选 dual 统计，不能说其全部计算毫无用途。

两组总计 408,064 次候选评分，其中 145,952 次未进入 PG。这些未进入 PG 的候选
消耗约 6.52 CPU-hours 评分，以及 9.86 小时累加的 worker rollout 墙钟时间；
后者不是独占 GPU-hours，更不能直接当成可兑现的时间节省。

现有补样按最多 8 组执行。对保存序列做只读的“达到第 16 个有效组立即停止”
计算，尾部共有 1,712 / 1,785 组，本次记录内相当于 13.42% / 14.01% 的评分。
这只是可避免尾部的记录内上界；小批补样可能降低八卡并行效率，改变后续数据边界，
必须实际 profile 后才能承诺 wall-clock 加速。

有效组也不等于有益驾驶更新：约 53.06% / 52.65% 的接受组只有 reward 分量有差异，
只有约 10.81% / 12.37% 的接受组有官方 binary 安全成本差异。
优势 RMS 小于 0.1 的接受组为 522 / 490 个，阈值只用于描述，不是新增训练过滤器。
N 的优势 RMS 中位数更高（0.432 对 0.296），但 ODE 没有比 C 更高的均分，TTC 更差。
因此不能把增加非零优势比例或优势幅度当成方法有效性的终点。

乘子不是没有工作：第 1024 次更新后，C / N 的 TTC lambda 为 1.644 / 2.567，
object-margin lambda 为 2.644 / 2.718。未筛选 first16 的累计 TTC 成本为
0.00833 / 0.01244，共同预算为 0.00586；object-margin 约 0.0211，共同预算 0.01118。
两组仍有约束缺口。N 的噪声参数存在实际梯度和更新，不能归因为 noise_head 未训练。
这些训练成本与 DEV ODE 违规分别统计，不跨 sampler 混成同一个约束满足结论。

[完整信号、乘子与成本统计](interim_update1024_signal_efficiency.json)。

## 全训练集与可借鉴方法

本轮 RL_TRAIN 为 101,592 场景 / 1176 logs。旧 RL 的完整日志虽有 36.204% 等分组，
但使用旧 v2 reward、不同人口和变化的 behavior，不能当本轮 v1.1 全量冻结模型的匹配统计。
用户要求的统一全量统计现已启动：固定 F-SFT、原 C_SDE、G16/K10、同一 canonical v1.1
和几何成本，对每个 TRAIN 场景保存全部候选反馈。它是 SFT 的全量难度诊断，
不是更新后 N_FLOW_NOISE 的全量部署成绩。完整结果发布前明确标为未完成。

这个 census 额外需要 1,625,472 次候选评分。不能在讨论采样加速时忽略这项一次性成本；
仅两遍训练时，预评分开销未必能回收。后续实际提效优先复用已有训练历史。

建议按问题分开处理，保持当前两组正式配方不变：

1. **先处理补样过量。** 在新运行身份下验证按剩余缺额组织补样，保留同一 behavior、
   first16 均匀 dual 人口和组内全部负样本；以实际评分数和墙钟时间比较。
   这是最明确的工程损耗，不能把它和解决满分饱和混成一个贡献。
2. **满分且所有有效安全成本相同：降低重复探索频率，同时保持复查与任务能力。**
   [GRESO](https://infini-ai-lab.github.io/GRESO/) 在生成前依据历史零方差次数进行概率过滤，
   并保留最低探索概率，比当前生成后才丢组更直接地减少 rollout。
   驾驶适配必须保留均匀安全审计、GT replay、参考保持和历史过期机制；不永久删除
   满分场景，不把一轮满分当永久安全。优先级应保留六个分量的历史统计，不能仅按
   PDMS 是否满分或旧 lambda 下 A=0 删除：乘子变化后，同一成本差异可能重新产生信号。
   其语言推理加速倍数不能当驾驶承诺。
3. **全零但安全/几何成本不同：利用已有真实分项差异；所有反馈都相同：承认没有排序信号。**
   [Prism-GRPO](https://arxiv.org/abs/2608.17423) 用执行质量区分同结果的 VLA 轨迹，
   是相关先例。本轮已经使用道路/对象余量分项，不能重新包装成首次加入；也不能
   为制造方差任意增加“越远越好、越慢越好”的奖励。持续完全同反馈的难例需要
   可行行为覆盖或已有 GT replay 的能力支撑，GT 不能补进 GRPO 旧链冒充 on-policy。

[DAPO](https://dapo-sia.github.io/) 的动态补样已经实现，但生成后过滤仍支付生成与评分成本。
[AERO](https://arxiv.org/html/2602.14338v1) 进一步研究自适应 rollout 和后验基线；
其按正负结果筛选、修改等分组基线的做法改变了本轮整组样本与优势合同，不能直接搬入
当前运行，也不能以“变成非零梯度”证明获得了有益驾驶信号。

当前结论：原 ODE 可以吸收部分真实策略梯度收益，当前平均收益有匹配证据；
学习噪声的额外价值尚未成立，TTC 安全保持与采样效率是明确未解决的问题。
不据此宣布整个 Flow RL 无效，也不以中途均分超过 92 宣布任务成功——匹配 SFT
在这个 DEV 上本来就已经超过 92。

## 已执行入口与续接

在本任务 worktree、已记录的 Python 环境中执行：

```bash
python -m scripts.analysis.constrained_flow_noise.export_checkpoint --checkpoint CHECKPOINT --output EXPORT
python -m scripts.analysis.constrained_flow_noise.evaluate_policy --export EXPORT --population POPULATION --output EVALUATION --baseline-scores BASELINE_SCORES --evaluation-override OVERRIDE --cache-root DEV_V1_CACHE --workers 16
python -m scripts.analysis.constrained_flow_noise.diagnose_signal_efficiency --run-root /mnt/project/constrained-flow-noise-20261014 --collections 512 --output NEW_SIGNAL_REPORT
python -m pytest tests/constrained_flow_noise/test_temporary_guard.py -q
python -m scripts.analysis.constrained_flow_noise.audit_train_population --registration CENSUS/registration.json --output CENSUS --shard SHARD
python -m scripts.analysis.constrained_flow_noise.summarize_population_census --root CENSUS --wait
```

精确参数保存在对应私有 launch receipts，实际资产路径见
[观察登记](interim_update1024_registration.json) 和 [census 启动记录](full_TRAIN_census_start.json)。
临时服务器上的 audit 命令必须由 `temporary_guard` 包装并传入本次 `--lease`；
长期授权节点的恢复重用同一 registration、output、shard，逐个验证并跳过已完成记录。
不会换噪声键、丢候选或将失败记成零分。未提交原始数据、候选内容、权重或缓存。
