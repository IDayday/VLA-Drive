# 首2500更新：安全难例曝光是否产生更多有用训练信号

2026-10-16，分析执行至14:50 UTC。两组均使用各自已封存的`update002500_inner0`，每组625份fresh全局rollout、5,000次scene曝光、40,000条G8候选。**这是已有训练记录分析，尚不是SAFETY25的单候选ODE评价。**

主要发现：相同候选预算下，安全分项有差异的组从21.84%增加到35.86%，progress-only组从72.86%降至59.64%。有效PG组仍分别为99.48%和98.54%，当前问题不是大量零优势。新人口增加了安全排序机会，同时增大global reward std并重复访问少数失败场景；这些机会能否转化为原ODE收益，要由既定DEV与终点Navtest确认。

## 证据、来源与比较边界

- [registration.json](registration.json)：分析前锁定窗口、两组identity、checkpoint封存摘要及最多500个待详查组/实验。
- [results.json](results.json)：完整聚合结果、八rank日志摘要、checkpoint transaction摘要、源码hash。
- [execution.json](execution.json)：两次实际只读执行、成本和结果一致性；没有新rollout、评分或参数更新。
- [validation.json](validation.json)：分母、证据覆盖、重加权、链接及正式源码hash复核；不是新增算法有效性测试。
- [progress_snapshot.json](progress_snapshot.json)：14:56 UTC实际训练及controller身份核查。LAST_ODE/SAFETY25完成四inner边界分别7,788/3,712，固定终点各20,000；最近封存窗口估算剩余训练4.72/5.15小时，评价另计，共享负载可能改变耗时。
- 实际入口：[compare_safety_exposure_prefix.py](../../../scripts/analysis/simwam_recipe/compare_safety_exposure_prefix.py)。主要复用现有日志、数据流和保存链验证函数。

对照是已完成的`R_CANONICAL_X1`与正在运行的`R_CANONICAL_X1_SAFETY25`，不是SOURCE与CANONICAL。两组SFT、初始LoRA、condition、评分、优化器和其他recipe相同，允许差异只有变体名、数据顺序及固定安全分层曝光。正式SAFETY25源码仍锁定`12c46a5bafd1e37a14435cd1e34aeb175731ca2f`，本次只读分析没有热修改它。

每次fresh rollout的全部八rank、四次inner均检查run、scene、collection、behavior、chain/old-probability/advantage摘要和global std归属。两组场景顺序不同，策略随训练演化，不能称为逐场景、逐候选的配对效果实验。安全标签来自固定SFT一次单候选评分，不意味着所有噪声下都安全。

## 信号与人口

下表组比例统一以各自5,000次scene曝光为分母；重复场景算重复曝光，不算新独立场景。progress-only使用原已登记统计定义，不是安全差异组的严格补集。

| 指标 | X1 | SAFETY25 |
|---|---:|---:|
| 有效PG组 | 4,974 / 99.48% | 4,927 / 98.54% |
| 安全分项有差异的组 | 1,092 / 21.84% | 1,793 / 35.86% |
| progress-only组 | 3,643 / 72.86% | 2,982 / 59.64% |
| 全满分组 | 3 / 0.06% | 30 / 0.60% |
| 全零分组 | 13 / 0.26% | 28 / 0.56% |
| 其他等分组 | 10 / 0.20% | 15 / 0.30% |
| 全部等分、零PG组 | 26 / 0.52% | 73 / 1.46% |
| 同组既有安全又有不安全候选 | 1,086 / 21.72% | 1,775 / 35.50% |
| 全部候选未联合通过安全指标 | 22 / 0.44% | 51 / 1.02% |
| SFT联合安全场景曝光 | 4,757 | 3,750 |
| SFT安全失败场景曝光 | 243 | 1,250 |
| 实际独立场景 / logs | 5,000 / 880 | 4,796 / 883 |
| 独立SFT安全失败场景 | 243 | 1,046 |

SAFETY25已访问全部1,046个固定失败场景，并产生204次重复失败场景曝光。类型曝光为：纯TTC 95→444、纯DAC 140→766、NC+TTC 1→25、DAC+TTC 5→11、纯NC 2→4。碰撞相关独立场景仍少，不能从总安全信号增加推断已学会稳定碰撞修复。

同组所有候选均未安全通过不等于全零分：例如TTC失败仍可得到合法的非零PDMS。全满分和全零分也不是所有等分组。

## 随机训练读出：不能混作部署成绩

下表分数转为0—100；联合安全为NC/DAC/TTC均完全通过的候选比例。

| 训练读出 | X1 | SAFETY25 |
|---|---:|---:|
| 实际不同人口上的随机PDMS | 81.5980 | 76.8358 |
| 实际人口上的联合安全比例 | 92.9700% | 84.4625% |
| 固定SFT-safe分层中的随机PDMS | 83.0109 | 83.1036 |
| 固定SFT-unsafe分层中的随机PDMS | 53.9392 | 58.0323 |
| 固定SFT-unsafe分层中的联合安全比例 | 44.3930% | 50.3400% |
| 按原困难人口比例描述性加权的随机PDMS | 81.6297 | 81.9125 |

最后一行仅把各自安全/失败分层均值按原困难清单的20,971:1,046重新加权。它没有校正层内scene、重复、策略年龄或训练历史差异，没有importance correction。原始均分低4.7622分不能直接判为策略退化；描述性加权高0.2827分也不能判为RL收益。上述均值都来自训练过程中多个behavior，不能当作2500检查点的固定策略评价。

## Global std、真实更新与裁剪

| 指标 | X1 | SAFETY25 |
|---|---:|---:|
| 每fresh rollout的raw global reward std均值 | 0.176146 | 0.267936 |
| 首inner总LoRA gradient norm均值 | 0.039539 | 0.051230 |
| 首inner ratio最大绝对误差 / clip比例 | 0 / 0 | 0 / 0 |
| 第二inner平均PPO clip比例 | 0.2942% | 0.5908% |
| 第三inner平均PPO clip比例 | 1.6192% | 2.7317% |
| 第四inner平均PPO clip比例 | 4.0500% | 5.2842% |

global std均值增加52.11%，相同绝对组内reward差的优势尺度因此可能缩小，但各组分布和参数梯度同时变化。实测总梯度没有消失；日志gradient norm包含PG与anchor，不能把它解释为单独PG贡献或仅凭loss标量认定anchor阻碍学习。

第一inner尚未更新时无ratio偏差，后续inner的clipping随真实更新出现，不构成放宽0.005 clip的依据。global std检查覆盖跨rank、四次inner和选中保存组的一致性；本次没有独立重读全部未选中候选来重新构造每次global std。

## 不安全正优势是否把安全候选排在后面

详查全部存在不安全正优势候选的保存组：X1为22组/176条记录，SAFETY25为84组/672条记录，保留每组全部G8。全部原始chain/old-probability/advantage摘要核验通过，没有按好坏丢弃组内成员。

| 指标 | X1 | SAFETY25 |
|---|---:|---:|
| 正优势但仍不安全候选 | 57 | 220 |
| 其中同组存在安全候选 | 38 | 167 |
| 其中分数高于或等于同组最佳安全候选 | 0 | 0 |
| 同组没有安全候选的正优势不安全候选 | 19 | 53 |
| 不安全正优势类型 | 57 TTC | 217 TTC，3 NC+TTC |

在这些有安全替代项的组内，不安全正优势候选分数全部低于最佳安全候选，未发现该范围的最佳安全/不安全排序颠倒。组中心化仍可能给“比更差失败好，但未完全安全”的候选正优势，这与奖励排序正确可以同时发生。它不等于策略保证安全，参数共享和ODE迁移仍可能产生新失败。

选中组中的优势总量不能当作全训练梯度比例；本次没有通过这些计数修改reward、anchor、学习率或采样边界。

## 执行、成本与下一观察点

实际执行两次：第一次源码`b703e75`，第二次`a57fc8e`增加checkpoint摘要与预登记值的显式一致检查。第一次摘要也匹配；两次`runs`结果逐字段完全相同。两次都成功，没有重算模型、仿真或训练。

| 成本 | 首次 | 加强身份检查后 |
|---|---:|---:|
| SSH invocation端到端秒数 | 12.3008 | 12.5142 |
| 分析主体秒数 | 10.4830 | 10.1852 |
| 保存链文件长度合计 | 53,895,228 bytes | 53,895,228 bytes |
| 新候选 / 新评分 / optimizer更新 | 0 / 0 / 0 | 0 / 0 / 0 |

累计端到端24.8150秒，读取并hash相同链文件的物理开销两次均计入。该字节统计不包含全部日志/元数据I/O，wall不是独占CPU利用时间。未使用GPU。原始私有scene/候选文件留在运行资产，不上传仓库。

已执行入口如下；复核时使用新的输出目录，程序拒绝覆盖已有分析：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.simwam_recipe.compare_safety_exposure_prefix \
  --registration /mnt/project/simwam-safety-exposure-20261016/analysis/prefix2500_registration.json \
  --output "$NEW_READONLY_ANALYSIS_OUTPUT"
```

执行工作目录为本分支worktree，实际运行主机为授权Tsinghua zt。固定5k/10k/15k/20k、DEV和终点Navtest队列不变。下一步先看SAFETY25的5k原ODE与shift5分别对照同阶段X1，以及NC/DAC/TTC新增违规、恢复和进度；不因本次信号更丰富就增加训练预算或宣布部署改进。
