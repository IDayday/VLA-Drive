# LAST_ODE 固定10000更新：完整单候选ODE阶段评价

**10k的部署结果仍未改善：原ODE DEV为93.4190，比同期Canonical低1.0828分；定向TRAIN为80.5337，低6.2177分。** 随机训练候选更安全，却没有同步转化为单候选ODE收益。5k与10k方向一致，但本报告不替代固定20k终点，也不据此中途调参或扩展实验矩阵。

2026-10-16T15:52:19.108190+00:00。DEV1696/16 logs、固定定向TRAIN512/343 logs；canonical v1.1、原始观测、单候选、seed2101。没有搜索或GT选择。

这是已登记20k计划中的固定观察点，不是Navtest。主目标仍是原十步均匀ODE的完整Navtest提升及安全保持；不能把DEV超过92写成目标已实现。两组训练seed均42，按log聚类的区间不包含训练或推理seed重复的不确定性。DEV仅保证RL更新未见，SFT可能见过；TRAIN是固定定向诊断集，不是总体无偏估计。

唯一科学配置差异为sde_exclude_last_step=true：三个随机步骤从[7,8,9]前移到[6,7,8]，第9步为普通Euler ODE。SFT、LoRA、困难场景顺序、reward、原anchor和optimizer不变，不叠加X1。窗口改变同时影响采样状态与被优化时间位置，不能归因于只删除最后一次epsilon。

| 人口 / 网格 | SFT | 同阶段Canonical | LAST_ODE | 对SFT差值 / 95%区间 | 对Canonical差值 / 95%区间 |
|---|---:|---:|---:|---|---|
| DEV / E_NATIVE | 93.6192 | 94.5018 | 93.4190 | -0.2003 [-0.4502, -0.0421] | -1.0828 [-1.4674, -0.7157] |
| DEV / E_SHIFT5 | 93.5747 | 94.3856 | 93.3719 | -0.2029 [-0.4280, +0.1127] | -1.0137 [-1.3836, -0.4653] |
| TRAIN_DIAGNOSTIC / E_NATIVE | 80.3316 | 86.7514 | 80.5337 | +0.2021 [-1.3470, +1.7945] | -6.2177 [-9.0633, -3.5799] |
| TRAIN_DIAGNOSTIC / E_SHIFT5 | 79.1099 | 86.9100 | 78.0743 | -1.0357 [-2.9725, +0.7613] | -8.8357 [-11.9529, -5.8120] |

## 分项与安全转移

| 人口 / 网格 / 模型 | NC | DAC | TTC | EP | Comfort | DDC | 零分 / 联合安全场景数 |
|---|---:|---:|---:|---:|---:|---:|---|
| DEV / E_NATIVE / SFT | 100.0000 | 99.2335 | 99.9410 | 85.8065 | 99.8821 | 98.8797 | 13 / 1683 |
| DEV / E_NATIVE / Canonical | 100.0000 | 99.7642 | 99.7642 | 87.4174 | 99.8821 | 98.8502 | 4 / 1688 |
| DEV / E_NATIVE / LAST_ODE | 100.0000 | 99.1745 | 99.8231 | 85.5027 | 99.9410 | 98.9092 | 14 / 1680 |
| DEV / E_SHIFT5 / SFT | 100.0000 | 99.1745 | 99.7642 | 85.9945 | 99.9410 | 98.9092 | 14 / 1678 |
| DEV / E_SHIFT5 / Canonical | 100.0000 | 99.7052 | 99.6462 | 87.3391 | 99.8821 | 98.9092 | 5 / 1685 |
| DEV / E_SHIFT5 / LAST_ODE | 100.0000 | 99.1156 | 99.7052 | 85.6727 | 99.8821 | 98.9092 | 15 / 1676 |
| TRAIN_DIAGNOSTIC / E_NATIVE / SFT | 100.0000 | 88.6719 | 95.8984 | 72.6396 | 99.8047 | 99.0234 | 58 / 434 |
| TRAIN_DIAGNOSTIC / E_NATIVE / Canonical | 99.9023 | 94.5312 | 96.2891 | 79.5336 | 99.8047 | 98.7305 | 28 / 466 |
| TRAIN_DIAGNOSTIC / E_NATIVE / LAST_ODE | 100.0000 | 88.6719 | 96.4844 | 72.3433 | 99.8047 | 99.0234 | 58 / 438 |
| TRAIN_DIAGNOSTIC / E_SHIFT5 / SFT | 100.0000 | 87.5000 | 95.3125 | 71.9342 | 99.8047 | 99.1211 | 64 / 425 |
| TRAIN_DIAGNOSTIC / E_SHIFT5 / Canonical | 99.9023 | 95.1172 | 95.3125 | 80.0704 | 99.8047 | 98.8281 | 25 / 464 |
| TRAIN_DIAGNOSTIC / E_SHIFT5 / LAST_ODE | 100.0000 | 86.7188 | 94.9219 | 70.9330 | 99.8047 | 99.1211 | 68 / 419 |

新增/恢复按每个分项完全通过定义；NC=0.5不算完全通过。固定safe人口由各参照分别定义。G+/G−是场景原始0—1分差的正/负部分总和，不是百分点。

| 人口 / 网格 / 参照 | NC 新增/恢复 | DAC 新增/恢复 | TTC 新增/恢复 | 固定safe新增联合违规/人口 | 非零→零/零→非零 | G+ / G− |
|---|---|---|---|---|---|---|
| DEV / E_NATIVE / SFT | 0/0 | 2/1 | 2/0 | 4/1683 | 2/1 | 1.3417/4.7379 |
| DEV / E_NATIVE / Canonical | 0/0 | 12/2 | 1/2 | 12/1688 | 12/2 | 3.1030/21.4667 |
| DEV / E_SHIFT5 / SFT | 0/0 | 2/1 | 2/1 | 4/1678 | 2/1 | 1.8657/5.3066 |
| DEV / E_SHIFT5 / Canonical | 0/0 | 12/2 | 2/3 | 14/1685 | 12/2 | 3.5770/20.7696 |
| TRAIN_DIAGNOSTIC / E_NATIVE / SFT | 0/0 | 8/8 | 1/4 | 8/434 | 8/8 | 9.5576/8.5229 |
| TRAIN_DIAGNOSTIC / E_NATIVE / Canonical | 0/1 | 41/11 | 6/7 | 44/466 | 41/11 | 13.3685/45.2031 |
| TRAIN_DIAGNOSTIC / E_SHIFT5 / SFT | 0/0 | 15/11 | 3/1 | 18/425 | 15/11 | 11.2721/16.5746 |
| TRAIN_DIAGNOSTIC / E_SHIFT5 / Canonical | 0/1 | 52/9 | 6/4 | 58/464 | 52/9 | 11.0097/56.2487 |

## 匹配训练窗口与更新方向

同一场景顺序、同一初始化；每个5k窗口均为10,000次scene曝光和80,000条新候选。这里汇总变化中的随机策略，不是独立端点评价。完整前10k为2,500份fresh rollout、20,000次scene曝光、160,000条候选。

| 窗口 / 配方 | 随机PDMS | 联合安全候选% | 有效PG组% | 安全差异组% | progress-only组% | 全零组 | 第一inner总梯度norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1—5000 / Canonical | 82.1064 | 92.6725 | 99.62 | 23.25 | 70.44 | 16 | 0.046273 |
| 1—5000 / LAST_ODE | 83.5382 | 96.2000 | 98.40 | 5.15 | 92.88 | 96 | 0.009138 |
| 5001—10000 / Canonical | 82.7486 | 93.0888 | 99.72 | 22.38 | 71.33 | 11 | 0.047205 |
| 5001—10000 / LAST_ODE | 83.6204 | 96.3912 | 98.57 | 4.68 | 93.68 | 99 | 0.008913 |

后5k中，LAST_ODE安全差异组只有4.68%，progress-only为93.68%；虽然98.57%的组仍有PG，但有效信号主要是进度排序。总梯度norm含PG和beta50 anchor，不能当作纯PG梯度或Adam参数位移。所有已检查第一inner的ratio误差和clipfraction均为0；后续inner的clipping是实际更新。

前10k的随机PDMS相对Canonical高1.1518分，联合安全率高3.415个百分点，EP低2.0266分。候选平均更安全与更多全零组同时存在；不能只根据均分推断所有难例都改善。[完整窗口、组事件转移及账本](matched_windows.json)

### 固定10k保存batch梯度

只重算collection2500原有8组/64候选，checkpoint在四次inner之后；控制复用已有Canonical10k诊断。两个变体的输入状态及随机步骤不同，不是同状态、同时间的一因素梯度干预。

| 配方 / 随机步 | PG norm | beta50 anchor norm | cosine | 源PPO clip% |
|---|---:|---:|---:|---:|
| Canonical / 7 | 0.013356 | 0.000907 | +0.0813 | 0.0000 |
| Canonical / 8 | 0.023930 | 0.002247 | -0.1801 | 0.0000 |
| Canonical / 9 | 0.033746 | 0.015697 | -0.0402 | 31.2500 |
| LAST_ODE / 6 | 0.004315 | 0.000250 | -0.1304 | 0.0000 |
| LAST_ODE / 7 | 0.004684 | 0.000602 | -0.1784 | 0.0000 |
| LAST_ODE / 8 | 0.007512 | 0.001587 | +0.2354 | 1.5625 |

LAST本batch的三步anchor norm都低于PG，末随机步cosine为正；5k的末随机步则是anchor norm较高。现有证据不支持“anchor在所有阶段都压死学习”。两组本batch正优势均无违规，不能代表总体安全梯度；LAST有15个正优势候选属于微小进度差异重叠标签。[重算结果](saved_batch_gradients.json)与[复用控制](control_gradient_reused.json)。

## 阶段结论与未排除因素

- 主部署收益不足持续到10k：相对Canonical，两种网格、两个人口的配对区间均为负。DEV原ODE相对SFT为−0.2003分，区间[−0.4502,−0.0421]；TRAIN原ODE相对SFT为+0.2021分但区间跨零。
- 分项并非全部下降：原ODE相对Canonical，DEV的DAC新增/恢复为12/2，TTC为1/2；TRAIN的DAC为41/11，TTC为6/7、NC为0/1。主要差距包含DAC与EP，不能将它简写成全部安全分项变差。
- 在固定SFT-safe人口上仍有新违规：DEV原ODE4/1683，TRAIN原ODE8/434；shift5的TRAIN为18/425。所有分项/零分转移保留在主表。
- 窗口前移同时改变采样轨迹、学习时刻和确定性后缀；现有结果说明这个完整窗口变体的候选收益没有顺利被部署ODE吸收，不能单独证明某一个机制是原因。
- DDP与SimWAM的FM线性路径可等价转换，但原动作SFT时间目标和条件网络不同；新的源码对照见[FM设计与可复用边界](../fm_design_comparison/README.md)。这不是证明FM公式错误。
- 继续已登记15k/20k和固定终点，不根据中期差值更改LR、reward、anchor、sigma、窗口或seed。

## 分析验收与成本边界

新通用阶段汇总入口复用了原SAFETY25汇总；在已经发布的SAFETY25固定5k上，全部评价数据、配对结果、权重身份和成本逐项一致。只扩展了已登记变体名、匹配控制路径和报告说明，不调用actor/scorer。

新增窗口分析的首个CPU尝试因历史Canonical日志没有chain_identity字段而拒绝，失败证据保留。随后明确区分覆盖：LAST的20,000个group摘要可核对四次inner的chain digest；Canonical的20,000个旧group没有此字段，不能宣称已通过同一摘要检查。两者token、behavior、reward/exploration复用及完整update序列均核对，第一份实际链/root仍逐位匹配。没有给旧日志补造链身份。[失败与修正边界](window_analysis_failure.json)

新增GPU诊断job wall为30.0453秒，函数体为28.0407秒，峰值torch allocated约6.8630GiB；共享GPU，不是独占计算小时。未运行新采样、评分或optimizer。窗口分析CPU函数体为20.7694秒；此前CPU失败尝试及旧入口完整前缀重读没有独立wall记录，明确为未知，不填0。原始行为和正式源码均未改写。

既有命令已实际执行：

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m scripts.analysis.simwam_recipe.compare_training_prefix \
  --control /mnt/project/simwam-recipe-20261015/runs/R_CANONICAL_seed42_v1 \
  --candidate /mnt/project/simwam-terminal-ode-20261016/runs/R_CANONICAL_LAST_ODE_seed42_v1 \
  --updates 10000 --window-updates 5000 --output "$NEW_MATCHED_WINDOWS"
```

GPU保存batch诊断命令及锁定源码、PID/start ticks和实际时间见[invocation](gradient_invocation.json)，范围见[登记](analysis_registration.json)。只在仍授权且容量合格的主机复查。

## 评价成本与复查

| 人口，两网格分表 | 逻辑单候选评分 | 实际推理（含核验） | 累计worker wall秒 | 单项job跨度秒 |
|---|---:|---:|---:|---:|
| DEV | 3392 | 3400 | 2891.32 | 362.54 |
| TRAIN_DIAGNOSTIC | 1024 | 1032 | 1606.58 | 203.28 |

两个人口并行，不能把各job跨度相加当作端到端wall；共享worker时间也不是独占GPU计算小时。合并导出及本报告聚合未单独恢复完整wall，不伪造为0。详见[完整结果与成本](results.json)。不重新运行模型或评分，只读取控制器已完成并校验的摘要。

实际可执行汇总入口，输出必须使用新目录：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m scripts.analysis.simwam_recipe.summarize_followup_stage \
  --variant R_CANONICAL_LAST_ODE --update 10000 \
  --matched-comparison-dir /mnt/project/simwam-terminal-ode-20261016/stage010000 \
  --output "$NEW_PUBLIC_STAGE_OUTPUT"
```
