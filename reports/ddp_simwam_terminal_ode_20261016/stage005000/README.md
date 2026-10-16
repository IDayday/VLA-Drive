# LAST_ODE 固定5k：候选更安全，尚未转化为单候选 ODE 改善

2026-10-16。已完成预登记的完整 DEV1696 / 16 logs 和定向 TRAIN512 / 343 logs；两种十步 ODE 网格均使用原始观测入口、seed2101、canonical v1.1、单候选。没有搜索、GT 选择或 best-of-N。该阶段是20k计划的中间观察，不是 Navtest，也不是固定终点结论。

**前5k匹配训练的随机候选均分与安全比例提高，但固定5k的两种 ODE 部署读出均低于同期 Canonical。** 不能将“末端噪声损害候选”的有限反事实直接升级为“去掉末端随机就能提高RL部署”。本次只分析已登记的评价及已保存链；不改变LR、anchor、窗口、奖励或训练预算。

## 1. 比较身份和有效预算

| 项目 | 合同 |
|---|---|
| 匹配控制 | R_CANONICAL_seed42_v1，固定5k |
| 变体 | R_CANONICAL_LAST_ODE_seed42_v1，固定5k |
| 唯一科学配置因素 | sde_exclude_last_step=true；随机窗口[7,8,9]→[6,7,8]，第9步普通Euler ODE；不叠加X1 |
| 共同设置 | 同一F-SFT、初始LoRA、困难清单与scene顺序、canonical reward、原anchor、LR、G8/K10、inner4 |
| 每组截至该阶段 | 5,000次optimizer更新，1,250次fresh全局rollout，10,000次scene曝光，80,000条新候选 |
| 正式训练源码 | 8b4dda71ea299e3eef53a8c0b613385d752b2907 |
| 合并导出SHA256 | 6a98c0b862cf30ce4079e9d5c78cbf45beb10cd317a0e3cc712c8ed2e1993a85 |

输入顺序、初始LoRA与condition身份通过匹配检查，第一份collection的root及共同六步ODE前缀逐位一致。后续模型与随机链不同，因此比较的是整个已命名窗口配方的效果；不是只替换同一链上的最后一个epsilon。正式采样与训练源码不因本次文档修改而变化。

## 2. 单候选 ODE 主表

所有PDMS均以0—100表示。区间为log聚类bootstrap的95%区间，不含训练seed或推理seed重复的不确定性；DEV只有16个logs。TRAIN是预登记定向诊断人口，不能当完整训练人口无偏均值。

| 人口 / 网格 | SFT | Canonical5k | LAST_ODE5k | LAST−SFT | LAST−Canonical / 95%区间 |
|---|---:|---:|---:|---:|---|
| DEV / E_NATIVE | 93.6192 | 94.3121 | 93.5608 | -0.0584 | -0.7513 [-1.1896, -0.4787] |
| DEV / E_SHIFT5 | 93.5747 | 94.3214 | 93.4772 | -0.0976 | -0.8442 [-1.3836, -0.4590] |
| TRAIN_DIAGNOSTIC / E_NATIVE | 80.3316 | 84.4715 | 79.6007 | -0.7309 | -4.8708 [-7.6046, -2.2766] |
| TRAIN_DIAGNOSTIC / E_SHIFT5 | 79.1099 | 83.9018 | 76.8505 | -2.2594 | -7.0513 [-9.8489, -4.3456] |

相对SFT的DEV两种网格区间均跨零；TRAIN原网格区间跨零，shift5为负。相对同期Canonical，四项配对区间均为负。这里没有用较高的网格替换主目标，也没有将DEV的93分解释为Navtest达到92。

### 分项、低分与安全人口

| 人口 / 网格 / 模型 | NC | DAC | TTC | EP | Comfort | DDC | 零分 / 联合安全场景数 |
|---|---:|---:|---:|---:|---:|---:|---|
| DEV / E_NATIVE / SFT | 100.0000 | 99.2335 | 99.9410 | 85.8065 | 99.8821 | 98.8797 | 13 / 1683 |
| DEV / E_NATIVE / Canonical | 100.0000 | 99.5283 | 99.9410 | 87.1155 | 99.8821 | 98.8797 | 8 / 1687 |
| DEV / E_NATIVE / LAST | 100.0000 | 99.2925 | 99.9410 | 85.6427 | 99.8821 | 98.8502 | 12 / 1683 |
| DEV / E_SHIFT5 / SFT | 100.0000 | 99.1745 | 99.7642 | 85.9945 | 99.9410 | 98.9092 | 14 / 1678 |
| DEV / E_SHIFT5 / Canonical | 100.0000 | 99.5873 | 99.7642 | 87.2557 | 99.8231 | 98.8797 | 7 / 1685 |
| DEV / E_SHIFT5 / LAST | 100.0000 | 99.2925 | 99.6462 | 85.7131 | 99.9410 | 98.9092 | 12 / 1678 |
| TRAIN_DIAGNOSTIC / E_NATIVE / SFT | 100.0000 | 88.6719 | 95.8984 | 72.6396 | 99.8047 | 99.0234 | 58 / 434 |
| TRAIN_DIAGNOSTIC / E_NATIVE / Canonical | 100.0000 | 92.3828 | 95.7031 | 77.5754 | 99.8047 | 98.8281 | 39 / 452 |
| TRAIN_DIAGNOSTIC / E_NATIVE / LAST | 100.0000 | 87.8906 | 96.0938 | 71.5885 | 99.8047 | 99.0234 | 62 / 432 |
| TRAIN_DIAGNOSTIC / E_SHIFT5 / SFT | 100.0000 | 87.5000 | 95.3125 | 71.9342 | 99.8047 | 99.1211 | 64 / 425 |
| TRAIN_DIAGNOSTIC / E_SHIFT5 / Canonical | 100.0000 | 91.7969 | 95.3125 | 77.2237 | 99.8047 | 99.0234 | 42 / 448 |
| TRAIN_DIAGNOSTIC / E_SHIFT5 / LAST | 100.0000 | 85.3516 | 95.3125 | 69.3241 | 99.8047 | 99.1211 | 75 / 415 |

### 配对新增违规、恢复与正负收益

表中“对照”分别为同一网格的固定SFT或同期Canonical；每种固定safe人口独立报告。新增/恢复按分项完全通过定义，NC=0.5不算完全通过。G+/G−为场景原始0—1差的正/负部分求和，不能称为百分点。

| 人口 / 网格 / 对照 | NC 新增/恢复 | DAC 新增/恢复 | TTC 新增/恢复 | 固定safe新增联合违规 / 人口 | 非零→零 / 零→非零 | G+ / G− |
|---|---|---|---|---|---|---|
| DEV / E_NATIVE / SFT | 0/0 | 1/2 | 1/1 | 2/1683 | 1/2 | 2.0559/3.0467 |
| DEV / E_NATIVE / Canonical | 0/0 | 6/2 | 1/1 | 7/1687 | 6/2 | 2.2894/15.0310 |
| DEV / E_SHIFT5 / SFT | 0/0 | 1/3 | 2/0 | 3/1678 | 1/3 | 3.2007/4.8559 |
| DEV / E_SHIFT5 / Canonical | 0/0 | 8/3 | 2/0 | 10/1685 | 8/3 | 3.4446/17.7624 |
| TRAIN_DIAGNOSTIC / E_NATIVE / SFT | 0/0 | 12/8 | 4/5 | 15/434 | 12/8 | 9.6179/13.3601 |
| TRAIN_DIAGNOSTIC / E_NATIVE / Canonical | 0/0 | 33/10 | 6/8 | 37/452 | 33/10 | 12.3237/37.2623 |
| TRAIN_DIAGNOSTIC / E_SHIFT5 / SFT | 0/0 | 17/6 | 4/4 | 20/425 | 17/6 | 7.3918/18.9600 |
| TRAIN_DIAGNOSTIC / E_SHIFT5 / Canonical | 0/0 | 41/8 | 6/6 | 46/448 | 41/8 | 9.8093/45.9120 |

TRAIN原ODE相对Canonical的主要差距是DAC和EP，TTC均值反而略高；不能把全部分项说成都下降。DEV原ODE的NC/TTC几乎饱和，无法据此证明已解决Navtest的TTC困难。

## 3. 匹配前5k训练随机策略：改善与信号损失同时存在

下面汇总各自不断更新中的随机策略，在相同10,000次scene曝光上的全部80,000条候选。它不是固定5k端点的独立随机策略评价，也不是单候选ODE。全组满分/零分不是部署中场景零分的同一事件。

| 读出 | Canonical | LAST_ODE |
|---|---:|---:|
| canonical PDMS | 82.1064 | 83.5382 |
| NC | 99.7694 | 99.8981 |
| DAC | 94.8312 | 97.5400 |
| TTC | 97.5962 | 98.6200 |
| EP | 66.9977 | 65.4555 |
| 联合安全候选% | 92.6725 | 96.2000 |
| 有效PG组% | 99.62 | 98.40 |
| 全满分组 | 8 | 9 |
| 全零分组 | 16 | 96 |
| 其他等分组 | 14 | 55 |
| progress-only组 | 7044 | 9288 |
| 安全分项有差异的组 | 2325 | 515 |

随机PDMS提高1.4318分，联合安全候选比例提高3.5275个百分点，但EP下降1.5422分。progress-only组由70.44%增至92.88%，安全分项有差异的组由23.25%降至5.15%。这支持“随机样本更干净，但可供安全排序的组更少”，尚不能证明哪个梯度机制导致部署收益不足。

按相同scene曝光配对，90组从非全零变成全零，10组从全零退出；1,840组失去安全分项差异，30组新出现差异。两策略各自的状态与权重不同，此处是整个训练前缀配方的现象，不是同一轨迹被末步噪声直接修复的因果计数。[组事件转移](group_flag_transitions.json)

### 更新和clipping

| inner | Canonical PPO clip% | LAST PPO clip% | Canonical总梯度norm | LAST总梯度norm |
|---|---:|---:|---:|---:|
| 1 | 0.0000 | 0.0000 | 0.046273 | 0.009138 |
| 2 | 0.3479 | 0.0300 | 0.040293 | 0.008594 |
| 3 | 2.1742 | 0.2758 | 0.030121 | 0.007349 |
| 4 | 5.3742 | 0.7204 | 0.023817 | 0.006242 |

这里训练日志的`gradient_norm`来自`total.backward()`后的已归约LoRA梯度，包含PG和beta50 anchor，**不是单独PG norm**。第一inner的最大ratio误差与clipfraction两组都为0；后续clipping来自真实更新，不能据此认定概率错配。第一inner总梯度约为控制的19.75%，不等于参数位移也精确缩小相同比例，Adam状态仍影响实际更新。

## 4. 保存batch逐步梯度：有限解释，不替代全程判断

只读取5k checkpoint和collection1250的既有8组链；没有新采样、评分或optimizer更新。此checkpoint在四次inner之后，保存old behavior在4996，因此下表clipping不是第一inner数值噪声。控制复用原报告已有5k数据，不新增GPU对照。两变体随机时间位置不同，共同步骤的输入状态也不同。

| 变体 / 随机步 | PG norm | beta50 anchor norm | cosine | PPO clip% |
|---|---:|---:|---:|---:|
| Canonical / 7 | 0.029014 | 0.002386 | -0.0986 | 0.0000 |
| Canonical / 8 | 0.034137 | 0.006647 | -0.2750 | 6.2500 |
| Canonical / 9 | 0.060697 | 0.037023 | -0.2812 | 45.3125 |
| LAST_ODE / 6 | 0.018443 | 0.000388 | -0.1045 | 1.5625 |
| LAST_ODE / 7 | 0.011826 | 0.000905 | -0.1690 | 4.6875 |
| LAST_ODE / 8 | 0.002142 | 0.003193 | -0.0544 | 12.5000 |

LAST的本批末随机步8上anchor norm大于PG norm，但cosine仅约−0.0544；不能据此把全程无收益归结为“anchor压死”。该批两组正优势候选均没有违规，因此本批也不代表全训练安全梯度。步骤/逻辑通道的均值与velocity偏移、有效LoRA权重变化见[实际梯度](saved_batch_gradients.json)，控制来源见[复用说明](control_gradient_reused.json)。

## 5. 行为描述

| 人口 / 原ODE模型 | 请求前向终点/m | 模拟前向终点/m | 请求平均分段速度/(m/s) | 请求jerk RMS/(m/s³) |
|---|---:|---:|---:|---:|
| DEV / SFT | 17.3461 | 17.4757 | 4.4918 | 0.9538 |
| DEV / Canonical | 17.4581 | 17.6510 | 4.5212 | 1.4262 |
| DEV / LAST | 17.3266 | 17.4361 | 4.4891 | 0.9986 |
| TRAIN_DIAGNOSTIC / SFT | 15.9079 | 15.9927 | 4.1904 | 0.9526 |
| TRAIN_DIAGNOSTIC / Canonical | 16.0438 | 16.1998 | 4.2236 | 1.4387 |
| TRAIN_DIAGNOSTIC / LAST | 15.8903 | 15.9543 | 4.1872 | 1.0135 |

LAST的原ODE前向行程与速度更接近SFT，未吸收同期Canonical的进度变化；请求轨迹平滑性也更接近SFT。jerk为请求点差分描述量，不是官方Comfort。各自轨迹“走过自身路径90%所用时间”不能当作到达同一目标点的时间；完整诊断保留在JSON，不据此声称更快完成任务。

## 6. 实际成本与当前状态

| 已完成任务 | 逻辑单候选评分 | 实际物理推理（含入口核验） | 累计worker秒 | 单项job跨度秒 |
|---|---:|---:|---:|---:|
| DEV，两网格分别报告 | 3392 | 3400 | 3277.95 | 411.99 |
| TRAIN_DIAGNOSTIC，两网格分别报告 | 1024 | 1032 | 1836.03 | 230.57 |
| 保存链梯度 | 0 | 0新轨迹；有重算forward/backward | 31.03 | 32.94 |

评价合计4,416次逻辑单候选评分、4,432次物理轨迹推理，差额16次是每worker一次原部署入口核验，不产生额外正式评分。16个worker均`ASSIGNED_COMPLETE`，new/reused场景为2208/0；没有以部分分母汇总。每个场景的两个网格分表，不是best-of-two。

DEV累计加载/前向/评分等待为256.24/2472.60/210.00秒；TRAIN为758.67/842.46/80.50秒。两项并行运行，不能把job跨度简单相加为端到端时间；共享worker时间也不是独占GPU计算小时。梯度重算峰值为6.8630GiB torch allocated，非整个GPU占用。CPU分析/管理和合并导出的独立wall未完整计时，不填成0。训练前缀中的`mean_update_seconds_per_rank`仅为更新步骤计时，不能拿它推算整轮含采样、评分、保存的总训练效率。

实际进程核对时间为2026-10-16T14:03:09.103768+00:00。LAST_ODE合法checkpoint为5400更新；另一个已登记的SAFETY25为900更新。两个训练launcher的PID/start ticks、cwd和命令匹配；四名既有controller匹配；各24个identity内source文件hash一致。本轮没有重启或热改任何训练，LAST_ODE最终Navtest仍关闭，等待固定20k及DEV阶段。[快照](runtime_snapshot.json)

## 7. 当前研究判断

- 已排除的简单解释：这5k不是没有真实更新，也没有第一inner ratio数值噪声；98.40%的组仍有PG。不能再把主要问题归为“没有优势”。
- 支持的现象：候选安全均值提高，安全差异组减少、progress-only占比提高，单候选ODE没有同步受益。提高候选质量和提高学习价值需要分开验证。
- 尚未确定的原因：窗口时间位置、未反传的确定性后缀、source global std、LoRA共享参数及anchor的相互作用仍混在这个配方因素内。现有单batch梯度不足以做因果归因。
- 本阶段的负结果不推翻X1已完成的Navtest正收益，也不能据此宣布整个Flow Matching路线无效。没有将当前阶段结果当作固定20k终点。
- 继续已经锁定的10k/15k/20k观察和固定终点；不根据本次结果换reward、加LR、降anchor、扩大噪声或增加训练seed。SAFETY25保持独立，不与LAST_ODE混合。

## 8. 可复查入口

机器可读完整分项、转移、CI、低尾和成本见[results.json](results.json)；实际训练曝光与候选统计见[matched_training_prefix.json](matched_training_prefix.json)。公开仅含聚合值、代码路径及身份hash，不包含场景token、图像、轨迹、权重或cache。

以下是本次实际执行过的只读入口。复查时使用新的输出文件，保留旧摘要；命令不会运行新策略或评分。

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m scripts.analysis.simwam_recipe.compare_training_prefix \
  --control /mnt/project/simwam-recipe-20261015/runs/R_CANONICAL_seed42_v1 \
  --candidate /mnt/project/simwam-terminal-ode-20261016/runs/R_CANONICAL_LAST_ODE_seed42_v1 \
  --updates 5000 --output "$NEW_TRAINING_PREFIX_SUMMARY"

CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m scripts.analysis.simwam_recipe.analyze_evaluation \
  --evaluation /mnt/project/simwam-terminal-ode-20261016/evaluations/R_CANONICAL_LAST_ODE_update005000_DEV \
  --reference /mnt/project/simwam-recipe-20261015/evaluations/R_CANONICAL_update005000_DEV \
  --source-compatibility reports/ddp_simwam_x_anchor_20261015/evaluation_recovery_20261016/compatibility.json \
  --output "$NEW_DEV_COMPARISON"
```

TRAIN使用同一命令将两个目录的DEV替换为TRAIN_DIAGNOSTIC。本次日志分析和保存batch分析源码锁定361ab35f6a1dee1d511ec34a61fc0875e2e2dee4；评价汇总源码及所有输入文件hash另见results.json。文档数字和本地链接核验见[validation.json](validation.json)。
