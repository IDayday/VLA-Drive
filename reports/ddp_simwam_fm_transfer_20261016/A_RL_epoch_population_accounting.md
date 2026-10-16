# A组RL跨epoch人口核算

2026-10-16 23:04 UTC完成只读核算。它使用已封存的12,000步恢复点和八rank有效日志，**不是新增模型评价观察点**。最新部署分数仍见[固定10k报告](interim_A_RL_update010000.md)，后续评价仍为15k/20k；没有修改正式训练、人口或预算。

| 12k已提交前缀 | 实际数量 |
|---|---:|
| 全局optimizer updates | 12,000 |
| fresh全局rollouts | 3,000 |
| 场景曝光 | 24,000 |
| 新G8候选 | 192,000 |
| 独立场景 / logs | 21,367 / 1,064 |
| 首次访问之后的重复曝光 | 2,633 |
| 其中：源even-batch末批补齐 | 1 |
| 其中：下一epoch的正常重访 | 2,632 |

固定困难清单共有21,367场景，已经完整访问一次；这不等于全体101,592个RL_TRAIN场景都参与了RL。源场景流将每个epoch补齐到21,368，保持每卡一个scene。第一个epoch为21,367个真实成员加1次补齐；第二个epoch在此边界已经访问2,632场景。18,734场景访问一次，2,633场景访问两次。

补齐发生在collection2671的rank7，位置为epoch0/index21367；该collection由update10680的冻结behavior生成，完成四次inner后到达10684。补齐scene重复了人口中的已有观测，但使用新的当前策略链和概率。八rank日志共24,000个不同的完整链摘要，没有把旧链复用计为fresh collection；同一collection的四次inner始终保留相同scene、behavior、chain、old logprob与advantage摘要。

因此，末批补齐仅占这段场景曝光的1/24,000，不是当前效率瓶颈。正常跨epoch重访是登记的固定困难人口循环；不能把这些曝光再写成新的独立scene，也不能将四次inner计算成四倍新候选。192,000是有效已提交链的逻辑候选数；失败/回滚的物理成本仍由独立invocation账本统计。

## 验证范围与限制

分析直接复用运行身份中hash一致的`SceneStream`和源`ResumableEpochSampler`，按seed42+epoch重放**场景索引**，没有重新生成策略轨迹。每个已提交scene/rank的实际token与预期顺序逐一比较；零优势与安全/失败结果不参与筛选。

12k checkpoint全部原生文件hash、八rank状态、transaction和run身份一致，阶段为READY/inner0。额外读取collection1、2、2670、2671、2672、2673、3000的56份保存behavior，核对实际stream position、token/log、behavior版本与chain/old-logprob/advantage摘要。全部通过；并非重新读取24,000份behavior张量。behavior生成invocation与first-inner执行invocation分别保留，合法恢复可能跨invocation，本分析不取代已有恢复账本。

新增14项CPU测试通过，覆盖world1/2/8、整除/补齐/跨epoch、四次inner分母、错误顺序、跨run/跨rank behavior、重复旧链、旧概率变化、日志缺失/重复和未提交尾行。分析本身main耗时4.591秒、进程CPU3.673秒，不含Python导入与最终JSON发布；0次GPU推理、0次新评分、0次optimizer更新。

机器证据：[完整核算与边界payload摘要](A_RL_update012000_population.json)。源码为[只读人口分析入口](../../scripts/analysis/fm_transfer/analyze_rl_population.py)，测试为[人口与复用计数测试](../../tests/rl/test_fm_transfer_rl_population.py)。正式训练源码仍为`225b9af660cb0210208e2c46fc4216d0f151275b`，本次不修改运行快照或控制器。

## 已执行命令

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/rl/test_fm_transfer_rl_population.py

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m scripts.analysis.fm_transfer.analyze_rl_population \
  --run /mnt/project/fm-transfer-20261016/runs/A_DDP_TIME_R_SOURCE_seed42_v1 \
  --manifest /mnt/project/fm-transfer-20261016/hard_manifests/A_DDP_TIME_SFT_update020000.json \
  --updates 12000 \
  --output /mnt/project/fm-transfer-20261016/analysis/A_DDP_TIME_RL_update012000_population.json
```

使用既有`/mnt/project/flow-safe-improvement-20261013/venv/bin/python`实际执行。输出已存在时拒绝覆盖。该入口可在后续合法完整边界复用，但当前没有15k/20k或B正式RL人口结果；不能将预期计数写成已完成证据。
